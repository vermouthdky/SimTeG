import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.transforms import SIGN
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ...model import get_model_class
from ...utils import is_dist
from ..trainer import Trainer

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels=None):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels.view(-1, 1)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        data = {
            "input_ids": self.input_ids[idx],
            "att_mask": self.attention_mask[idx],
            "labels": self.labels[idx] if self.labels is not None else None,
        }
        # pop up None values
        for key in list(data.keys()):
            if data[key] is None:
                data.pop(key)
        return data


class OneIterTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        NOTE: add KL divergence here
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if return_outputs:
            logits, hidden_features = model(**inputs, return_hidden=True)
        else:
            logits = model(**inputs, return_hidden=False)

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor)
        loss = loss_op(logits, labels.view(-1))
        return (loss, {"logits": logits, "hidden_features": hidden_features}) if return_outputs else loss


class LM_Trainer(Trainer):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        super().__init__(args, data, split_idx, evaluator, **kwargs)

    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test", "all"]
        dataset = Dataset(self.data.input_ids, self.data.attention_mask, self.data.y)
        return dataset if mode == "all" else torch.utils.data.Subset(dataset, self.split_idx[mode])

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid")

    def _prepare_model(self):
        return get_model_class(self.args.model_type, self.args.use_adapter)(self.args)

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.batch_size * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            # overwrite_output_dir=True,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            metric_for_best_model="eval_accuracy",
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.args.accum_interval,
            label_smoothing_factor=self.args.label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            num_train_epochs=self.args.epochs,
            local_rank=self.rank,
            dataloader_num_workers=1,
            ddp_find_unused_parameters=False,
        )

        def compute_metrics(eval_pred):
            metric = evaluate.load("accuracy")
            logits, labels = eval_pred
            if isinstance(logits, tuple):
                logits = logits[0]
            predictions = logits.argmax(-1)
            return metric.compute(predictions=predictions, references=labels)

        return OneIterTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
