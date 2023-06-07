import gc
import logging
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch_geometric.utils import subgraph
from torchmetrics.functional import retrieval_reciprocal_rank as mrr
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


# def relabel(src, dst, neg_dst):
#     src_dst_idx = torch.cat([src, dst, neg_dst]).unique()
#     num_nodes = torch.max(src_dst_idx) + 1
#     node_idx = torch.zeros(num_nodes, dtype=torch.long)
#     node_idx[src_dst_idx] = torch.arange(src_dst_idx.size(0))
#     return node_idx[src], node_idx[dst], node_idx[neg_dst]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, att_mask, src, dst, neg_dst):
        super().__init__()
        self.input_ids, self.att_mask = input_ids, att_mask
        self.src, self.dst, self.neg_dst = src, dst, neg_dst
        # auxiliary
        self.num_nodes = self.input_ids.size(0)

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, index):
        src, dst, neg_dst = self.src[index], self.dst[index], torch.randint(self.num_nodes, (1,))
        all_idx = torch.tensor([src, dst, neg_dst], dtype=torch.long)
        data = dict(input_ids=self.input_ids[all_idx], att_mask=self.att_mask[all_idx])
        return data


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, att_mask):
        super().__init__()
        self.input_ids, self.att_mask = input_ids, att_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, index):
        return dict(input_ids=self.input_ids[index], att_mask=self.att_mask[index])


class InnerTrainer(HugTrainer):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        eps = 1e-15
        input_ids, att_mask = inputs.pop("input_ids"), inputs.pop("att_mask")
        pos_out, neg_out = model(input_ids=input_ids.cuda(), att_mask=att_mask.cuda())
        pos_loss = -torch.log(pos_out + eps).mean()
        neg_loss = -torch.log(1 - neg_out + eps).mean()
        loss = pos_loss + neg_loss
        if return_outputs:
            outputs = dict(pos_out=pos_out, neg_out=neg_out)
        return (loss, outputs) if return_outputs else loss


class LinkLMTrainer(Trainer):
    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test"]
        src, dst = self.split_idx[mode]["source_node"], self.split_idx[mode]["target_node"]
        if mode == "train":
            neg_dst = torch.randint(0, self.data.num_nodes, (src.size(0),))
        else:
            neg_dst = self.split_idx[mode]["target_node_neg"]
        dataset = Dataset(self.data.input_ids, self.data.attention_mask, src=src, dst=dst, neg_dst=neg_dst)
        return dataset

    def _prepare_dataset(self):
        train_set, valid_set = self._get_dataset("train"), self._get_dataset("valid")
        all_set = InferenceDataset(self.data.input_ids, self.data.attention_mask)
        return train_set, valid_set, all_set

    def compute_metrics(self, eval_pred):
        pos_out, neg_out = eval_pred[0]
        pos_out, neg_out = torch.from_numpy(pos_out), torch.from_numpy(neg_out)
        pos_target, neg_target = torch.ones_like(pos_out, dtype=torch.bool), torch.zeros_like(neg_out, dtype=torch.bool)
        predict = torch.cat([pos_out, neg_out])
        target = torch.cat([pos_target, neg_target])
        return {"mrr": mrr(predict, target)}

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.batch_size * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            seed=self.args.random_seed,
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            # eval_accumulation_steps=10,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="mrr",
            dataloader_drop_last=True,
            gradient_accumulation_steps=self.args.accum_interval,
            label_smoothing_factor=self.args.label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            disable_tqdm=False,
            num_train_epochs=self.args.epochs,
            local_rank=self.rank,
            dataloader_num_workers=8,
            ddp_find_unused_parameters=False,
            label_names=[],
        )
        return InnerTrainer(
            data=self.data,
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def inference(self, dataset, embs_path):
        x_embs_name = "x_embs.pt"
        emb_handler = EmbeddingHandler(embs_path)
        if self.args.use_cache and emb_handler.has([x_embs_name]):
            x_embs = emb_handler.load(x_embs_name)
            if isinstance(x_embs, np.ndarray):
                x_embs = torch.from_numpy(x_embs)
        else:
            eval_output = self.trainer.predict(dataset)
            x_embs = eval_output.predictions
            x_embs = torch.from_numpy(x_embs)
            emb_handler.save(x_embs, x_embs_name)
            logger.info(f"save the hidden features of {self.args.lm_type} to {osp.join(embs_path, x_embs_name)}")
        return x_embs

    def _evaluate(self, x_embs):
        evaluator = Evaluator(name="ogbl-citation2")

        def test_split(split):
            source = self.split_idx[split]["source_node"].to(self.rank)
            target = self.split_idx[split]["target_node"].to(self.rank)
            target_neg = self.split_idx[split]["target_node_neg"].to(self.rank)

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), 10000):
                src, dst = source[perm], target[perm]
                # pos_preds += [self.model.link_predict(x_embs[src], x_embs[dst]).squeeze().cpu()]
                pos_preds = torch.dot(x_embs[src], x_embs[dst])
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.view(-1)
            for perm in DataLoader(range(source.size(0)), 10000):
                src, dst_neg = source[perm], target_neg[perm]
                # neg_preds += [self.model.link_predict(x_embs[src], x_embs[dst_neg]).squeeze().cpu()]
                neg_preds += torch.dot(x_embs[src], x_embs[dst_neg])
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            mrr = evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})
            logger.info(mrr)
            return mrr["mrr_list"].mean().item()

        return dict(valid_mrr=test_split("valid"), test_mrr=test_split("test"))

    def inference_and_evaluate(self, dataset):
        embs_path = osp.join(self.args.output_dir, "cached_embs")
        x_embs = self.inference(dataset, embs_path)
        results = self._evaluate(x_embs)
        logger.critical("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))
        gc.collect()
        torch.cuda.empty_cache()
        return results  # logits_embs is None

    def train(self, return_value="valid"):
        self.prepare()
        assert self.args.mode in ["train", "test"]
        if self.args.mode == "train":
            self.train_once()

        logger.warning(f"\n*************** Start inference and testing ***************\n")
        results = self.inference_and_evaluate(self.all_set)
        gc.collect()
        torch.cuda.empty_cache()
        return results["test_mrr"], results["valid_mrr"]
