import gc
import logging
import os
import os.path as osp
import shutil
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import SIGN, ToSparseTensor
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class InnerTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        logits = model(**inputs)

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduce="mean")
        loss = loss_op(logits, labels.view(-1))
        return (loss, {"logits": logits}) if return_outputs else loss

    def _get_dataloader(self, mode) -> DataLoader:
        assert mode in ["train", "valid", "test"]
        world_size = int(os.getenv("WORLD_SIZE", 1))
        rank = int(os.getenv("RANK", -1))
        mask = self.train_dataset[f"{mode}_mask"]
        train_idx = mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
        kwargs = dict(
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=True,
        )
        return NeighborLoader(
            self.train_dataset, input_nodes=train_idx, num_neighbors=[25, 10], shuffle=True, drop_last=True, **kwargs
        )

    def get_train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        del eval_dataset  # do not use it in transductive setting
        return self._get_dataloader("valid")

    def get_test_dataloader(self, test_dataset: Dataset | None = None) -> DataLoader:
        del test_dataset
        return self._get_dataloader("test")


class GNNSamplingTrainer(Trainer):
    def ckpt_path(self, iter, stage="gnn", module="gnn"):
        return osp.join(self.args.ckpt_dir, f"iter_{iter}_{stage}_{module}.pt")

    def _prepare_model(self):
        model_class = get_model_class(self.args.model_type, self.args.use_adapter)
        model = model_class(self.args)
        return model

    def _compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.gnn_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
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
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data,
            eval_dataset=self.data,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def inference(self, dataset, embs_path):
        logits_name = f"iter_{self.iter}_logits.pt"
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

    def train_once(self, iter):
        dist.barrier()
        self.model = self._prepare_model()
        # TODO: maybe we should not load the gnn model here
        if iter > 0 and self.args.inherit and self.args.gnn_inherit:
            self.load_model(self.model.gnn_model, self.ckpt_path(iter, "lm", "gnn"))

        self.train_set, self.valid_set = self._prepare_dataset()
        self.trainer = self._prepare_trainer()
        if self.trial is not None:
            self.trainer._hp_search_setup(self.trial)

        if self.args.train_mode != "lm" or iter == 0:
            train_output = self.trainer.train()
            self.save_model(self.model.gnn_model, self.ckpt_path(iter, "gnn", "gnn"))
            global_step, train_dict = train_output.global_step, train_output.metrics
            train_dict["global_step"] = global_step
            self.trainer.save_metrics("train", train_dict)
            logger.critical("".join("{}:{} ".format(k, v) for k, v in train_dict.items()))
        else:
            self.save_model(self.model.gnn_model, self.ckpt_path(iter, "gnn", "gnn"))
        gc.collect()
        torch.cuda.empty_cache()

    def edge_index_to_adj_t(self):
        self.data = ToSparseTensor()(self.data)
        deg = self.data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self.data.adj_t = deg_inv_sqrt.view(-1, 1) * self.data.adj_t * deg_inv_sqrt.view(1, -1)

    def train(self, return_value="valid"):  # used when only train gnn_models
        # preprocessing
        self.edge_index_to_adj_t()
        xs = [self.data.x]
        disable_tqdm = self.args.disable_tqdm or (is_dist() and int(os.environ["RANK"]) > 0)
        logger.info("propogating features")
        for i in tqdm(range(1, self.args.gnn_num_layers + 1), disable=disable_tqdm):
            xs += [self.data.adj_t @ xs[-1]]
        xs = xs[1:]  # remove the hop-0 feature, which is saved in self.data.x, this is consistent with gbert
        self.data.x_emb = torch.cat(xs, dim=-1)
        logger.info("finished processing")

        self.model = self._prepare_model()
        self.train_set, self.valid_set = self._prepare_dataset()
        self.all_set = self._get_dataset("all")
        self.trainer = self._prepare_trainer()
        iter = self.iter = 0

        # ckpt_path = os.path.join(self.args.ckpt_dir, "iter_0")
        # if self.args.use_cache and os.path.exists(ckpt_path):
        #     logger.warning(f"\n*********iter {iter} has been trained, use cached ckpt instead!*********\n")
        # else:
        self.train_once(iter)
        logger.warning(f"\n*************** Start inference and testing ***************\n")
        # NOTE inference for SLE and propogation
        _, _, results = self.inference_and_evaluate()

        gc.collect()
        torch.cuda.empty_cache()
        return results["valid_acc"] if return_value == "valid" else results["test_acc"]
