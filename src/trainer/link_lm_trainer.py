import gc
import logging
import os
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
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

NUM_NEG_PER_SAMPLE = 1000


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
        src, dst = self.src[index], self.dst[index]
        if self.neg_dst is None:  # train set
            neg_dst = torch.randint(self.num_nodes, (1,))
        else:
            # perm = torch.randint(NUM_NEG_PER_SAMPLE, (NUM_NEG_PER_SAMPLE,))  # for approxiamate evalation
            neg_dst = self.neg_dst[index]
        all_idx = torch.tensor([src, dst], dtype=torch.long)
        all_idx = torch.cat([all_idx, neg_dst])
        labels = torch.tensor([1] + [0] * NUM_NEG_PER_SAMPLE, dtype=torch.long)
        data = dict(
            input_ids=self.input_ids[all_idx],
            att_mask=self.att_mask[all_idx],
            labels=labels,
        )
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
        labels = inputs.pop("labels")
        input_ids, att_mask = inputs.pop("input_ids"), inputs.pop("att_mask")

        if len(input_ids.shape) == 2:
            x_embs = model(input_ids=input_ids.cuda(), att_mask=att_mask.cuda())
            return (None, x_embs)

        if return_outputs:  # evaluation
            pos_preds, neg_preds = None, []
            for perm in DataLoader(
                torch.range(2, NUM_NEG_PER_SAMPLE), batch_size=10, shuffle=False
            ):
                perm = torch.cat((torch.tensor([0, 1]), perm)).long()
                batch_input_ids, batch_att_mask = (
                    input_ids[:, perm, :],
                    att_mask[:, perm, :],
                )
                with torch.no_grad():
                    pos_pred, neg_pred = model(
                        batch_input_ids.cuda(), batch_att_mask.cuda()
                    )
                pos_preds = pos_pred
                neg_preds.append(neg_pred)
            pos_out, neg_out = pos_preds, torch.cat(neg_preds, dim=-1)
        else:
            pos_out, neg_out = model(
                input_ids=input_ids.cuda(), att_mask=att_mask.cuda()
            )

        pos_loss = -torch.log(pos_out + eps).mean() / (input_ids.size(1) - 2)
        neg_loss = -torch.log(1 - neg_out + eps).mean()
        loss = pos_loss + neg_loss
        if return_outputs:
            outputs = dict(pos_out=pos_out, neg_out=neg_out)
        return (loss, outputs) if return_outputs else loss


class LinkLMTrainer(Trainer):
    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test"]
        src, dst = (
            self.split_idx[mode]["source_node"],
            self.split_idx[mode]["target_node"],
        )
        # use the fist 100 for validation
        neg_dst = None if mode == "train" else self.split_idx[mode]["target_node_neg"]
        if mode == "valid":
            logger.warning("Use 0.01 of the validation set for fast evaluation")
            num_samples = src.size(0) // 100
            src, dst, neg_dst = (
                src[:num_samples],
                dst[:num_samples],
                neg_dst[:num_samples],
            )
        dataset = Dataset(
            self.data.input_ids,
            self.data.attention_mask,
            src=src,
            dst=dst,
            neg_dst=neg_dst,
        )
        return dataset

    def _prepare_dataset(self):
        train_set, valid_set = self._get_dataset("train"), self._get_dataset("valid")
        all_set = InferenceDataset(self.data.input_ids, self.data.attention_mask)
        return train_set, valid_set, all_set

    def _prepare_model(self):
        model_class = get_model_class(self.args.model_type, self.args.task_type)
        model = model_class(self.args)
        n_params = sum(p.numel() for p in model.parameters())
        logger.warning(f"Model: {self.args.model_type}, Num of Params: {n_params}")
        if self.args.use_cache:
            logger.warning(f"Loading model checkpoint from {self.ckpt_path}")
            model.load_state_dict(torch.load(self.ckpt_path))
        return model

    def compute_metrics(self, eval_pred):
        evaluator = Evaluator(name="ogbl-citation2")
        pos_out, neg_out = eval_pred[0]
        pos_out, neg_out = torch.from_numpy(pos_out), torch.from_numpy(neg_out)
        results = evaluator.eval(
            {"y_pred_pos": pos_out.view(-1), "y_pred_neg": neg_out}
        )
        return {f"eval_{key[:-5]}": value.mean() for key, value in results.items()}

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = (
            self.world_size * self.args.batch_size * self.args.accum_interval
        )
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            seed=self.args.random_seed,
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            eval_delay=self.args.eval_delay * eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            # eval_accumulation_steps=10,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_mrr",
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
            label_names=["labels"],
            deepspeed=self.args.deepspeed,
            fp16=self.args.fp16,
            resume_from_checkpoint=True,
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
            logger.info(
                f"save the hidden features of {self.args.lm_type} to {osp.join(embs_path, x_embs_name)}"
            )
        return x_embs

    def _evaluate(self, x_embs):
        evaluator = Evaluator(name="ogbl-citation2")

        @torch.no_grad()
        def test_split(split):
            source = self.split_idx[split]["source_node"].contiguous()
            target = self.split_idx[split]["target_node"].contiguous()
            target_neg = self.split_idx[split]["target_node_neg"].contiguous()

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), 5000):
                src, dst = source[perm], target[perm]
                x_src, x_dst = x_embs[src].to(self.rank), x_embs[dst].to(self.rank)
                pos_preds += [
                    self.trainer.model.link_predict(x_src, x_dst).squeeze().cpu()
                ]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.reshape(-1)
            for perm in DataLoader(range(source.size(0)), 5000):
                src, dst_neg = source[perm], target_neg[perm]
                x_src, x_dst_neg = x_embs[src].to(self.rank), x_embs[dst_neg].to(
                    self.rank
                )
                neg_preds += [
                    self.trainer.model.link_predict(x_src, x_dst_neg).squeeze().cpu()
                ]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            results = evaluator.eval(
                {"y_pred_pos": pos_pred.view(-1), "y_pred_neg": neg_pred}
            )
            return {key[:-5]: value.mean().item() for key, value in results.items()}

        return test_split("valid"), test_split("test")

    def inference_and_evaluate(self, dataset):
        embs_path = osp.join(self.args.output_dir, "cached_embs")
        x_embs = self.inference(dataset, embs_path)
        valid_results, test_results = self._evaluate(x_embs)
        logger.info(
            f"Valid Metrics:: "
            + "".join("{}: {:.4f} ".format(k, v) for k, v in valid_results.items())
        )
        logger.info(
            f"Test Metrics:: "
            + "".join("{}: {:.4f} ".format(k, v) for k, v in test_results.items())
        )
        gc.collect()
        torch.cuda.empty_cache()
        return valid_results, test_results  # logits_embs is None

    def train(self, return_value="valid"):
        dist.barrier()
        self.prepare()
        assert self.args.mode in ["train", "test"]
        if self.args.mode == "train":
            self.train_once()

        logger.warning(
            f"\n*************** Start inference and testing ***************\n"
        )
        valid_results, test_results = self.inference_and_evaluate(self.all_set)
        gc.collect()
        torch.cuda.empty_cache()
        return valid_results, test_results
