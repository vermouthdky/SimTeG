import gc
import logging
import os
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments as HugTrainingArguments
from transformers.trainer_pt_utils import distributed_concat

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class TrainingArguments(HugTrainingArguments):
    def __init__(self, *args, **kwargs):
        self.num_gnn_layers = kwargs.pop("num_gnn_layers", 4)
        super().__init__(*args, **kwargs)


class InnerTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels").cuda()
        elif "y" in inputs:
            labels = inputs.pop("y").cuda()
        else:
            labels = None

        edge_index, x = inputs.pop("edge_index").cuda(), inputs.pop("x").cuda()
        logits = model(x, edge_index)[: inputs.batch_size]
        # if return_outputs:
        #     if int(os.getenv("RANK", -1)) == 0:
        #         __import__("ipdb").set_trace()
        #     else:
        #         dist.barrier()

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduce="mean")
        loss = loss_op(logits, labels.view(-1)[: inputs.batch_size])

        return (loss, {"logits": logits}) if return_outputs else loss

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        tensors = distributed_concat(tensors)
        return tensors

    @property
    def rank(self):
        return int(os.getenv("RANK", -1))

    @property
    def world_size(self):
        return int(os.getenv("WORLD_SIZE", 0))

    def get_train_dataloader(self) -> DataLoader:
        train_idx = self.train_dataset.train_mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.split(train_idx.size(0) // self.world_size)[self.rank]
        return NeighborLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=True,
            input_nodes=train_idx,
            num_neighbors=[25] * self.args.num_gnn_layers,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_eval_dataloader(self, eval_dataset: Dataset = None) -> DataLoader:
        # eval_idx = self.train_dataset.valid_mask.nonzero(as_tuple=False).view(-1)
        # eval_idx = eval_idx.split(eval_idx.size(0) // self.world_size)[self.rank]
        return NeighborLoader(
            self.train_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=True,
            num_neighbors=[-1],
            shuffle=False,
            drop_last=False,
        )

    def get_test_dataloader(self, test_dataset: Dataset = None) -> DataLoader:
        raise NotImplementedError("you should not use this method")


class GNNSamplingTrainer(Trainer):
    def _prepare_dataset(self):  # prepare train_mask et.al.
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[self.split_idx[split]] = True
            self.data[f"{split}_mask"] = mask
        return self.data, self.data, self.data

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.gnn_batch_size
        train_steps = len(self.split_idx["train"]) // total_batch_size + 1
        eval_steps = train_steps * self.args.gnn_eval_interval
        warmup_steps = self.args.gnn_warmup_ratio * train_steps
        logger.info(f"eval_steps: {eval_steps}, train_steps: {train_steps}, warmup_steps: {warmup_steps}")
        training_args = TrainingArguments(
            seed=self.args.random_seed,
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            learning_rate=self.args.gnn_lr,
            weight_decay=self.args.gnn_weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            gradient_accumulation_steps=1,
            label_smoothing_factor=self.args.gnn_label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.gnn_batch_size,
            per_device_eval_batch_size=self.args.gnn_eval_batch_size,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.args.gnn_lr_scheduler_type,
            disable_tqdm=False,
            num_train_epochs=self.args.gnn_epochs,
            local_rank=self.rank,
            dataloader_num_workers=1,
            ddp_find_unused_parameters=False,
            label_names="y",
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data,
            eval_dataset=None,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def inference(self, dataset, embs_path):
        logits_name = f"logits.pt"
        emb_handler = EmbeddingHandler(embs_path)
        if self.args.use_cache and emb_handler.has(logits_name):
            logits_embs = emb_handler.load(logits_name)
            if isinstance(logits_embs, np.ndarray):
                logits_embs = torch.from_numpy(logits_embs)
        else:
            eval_output = self.trainer.predict(dataset)
            logits_embs = eval_output.predictions
            logits_embs = torch.from_numpy(logits_embs)
            emb_handler.save(logits_embs, logits_name)
            logger.info(f"save the logits of {self.args.gnn_type} to {os.path.join(embs_path, logits_name)}")
        return (logits_embs, None)
