import gc
import logging
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.utils import subgraph
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids=None,
        attention_mask=None,
        edge_index=None,
    ):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
            "edge_index": edge_index,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = {
            "input_ids": self.data["input_ids"][index],
            "att_mask": self.data["att_mask"][index],
            "edge_index": subgraph(index, self.data["edge_index"], relabel_nodes=True),
        }
        return batch_data


class InnerTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        eps = 1e-15
        input_ids = inputs.pop("input_ids")
        att_mask = inputs.pop("att_mask")
        edge_index = inputs.pop("edge_index")

        # forward
        h = model(input_ids=input_ids.cuda(), attention_mask=att_mask.cuda())

        # compute postive loss
        src, dst = edge_index[0], edge_index[1]
        pos_out = model.link_predict(h[src], h[dst])
        pos_out = -torch.log(pos_out + eps).mean()

        # compute negative loss
        dst_neg = torch.randint(0, h.size(0), src.size(), dtype=torch.long).cuda()
        neg_out = model.link_predict(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + eps).mean()

        loss = pos_out + neg_loss

        if return_outputs:
            outputs = {"hidden_features": h}
        return (loss, outputs) if return_outputs else loss


class LinkLMTrainer(Trainer):
    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test", "all"]
        if self.rank == 0:
            __import__("ipdb").set_trace()
        else:
            torch.distributed.barrier()
        dataset = Dataset(self.data.input_ids, self.data.attention_mask, self.data.y)
        return dataset if mode == "all" else torch.utils.data.Subset(dataset, self.split_idx[mode])

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid"), self._get_dataset("all")

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
            metric_for_best_model="eval_accuracy",
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
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def _evalute(self):
        pass

    def inference(self):
        pass
