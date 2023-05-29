import gc
import logging
import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class OneIterTrainer(HugTrainer):
    pass


class Trainer(ABC):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.iter = 0
        self.trial = kwargs.get("trial", None)

    @property
    def rank(self):
        return int(os.environ["RANK"]) if is_dist() else -1

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"]) if is_dist() else 1

    @property
    def disable_tqdm(self):
        return self.args.disable_tqdm or (is_dist() and self.rank > 0)

    @property
    def ckpt_path(self):
        return osp.join(self.args.ckpt_dir, "model.pt")

    def save_model(self, model: torch.nn.Module, ckpt_path):
        if self.rank <= 0:
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved the model to {}".format(ckpt_path))
        if is_dist():
            dist.barrier()

    def load_model(self, model: torch.nn.Module, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    def _prepare_model(self):
        model_class = get_model_class(self.args.model_type, self.args.task_type)
        model = model_class(self.args)
        n_params = sum(p.numel() for p in model.parameters())
        logger.warning(f"Model: {self.args.model_type}, Num of Params: {n_params}")
        return model

    @abstractmethod
    def _prepare_dataset(self):
        pass

    @abstractmethod
    def _prepare_trainer(self):
        pass

    def compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)

    def inference(self, dataset, embs_path):
        x_embs_name = f"x_embs.pt"
        logits_name = f"logits.pt"
        emb_handler = EmbeddingHandler(embs_path)
        if self.args.use_cache and emb_handler.has([x_embs_name, logits_name]):
            x_embs = emb_handler.load(x_embs_name)
            logits_embs = emb_handler.load(logits_name)
            if isinstance(x_embs, np.ndarray):
                x_embs, logits_embs = torch.from_numpy(x_embs), torch.from_numpy(logits_embs)
        else:
            eval_output = self.trainer.predict(dataset)
            logits_embs, x_embs = eval_output.predictions[0], eval_output.predictions[1]
            logits_embs, x_embs = torch.from_numpy(logits_embs), torch.from_numpy(x_embs)
            emb_handler.save(x_embs, x_embs_name)
            emb_handler.save(logits_embs, logits_name)
            logger.info(f"save the logits of {self.args.lm_type} to {osp.join(embs_path, logits_name)}")
            logger.info(f"save the hidden features of {self.args.lm_type} to {osp.join(embs_path, x_embs_name)}")
        return logits_embs, x_embs

    def _evaluate(self, logits, y):
        def accuracy(logits, y_true):
            y_pred = logits.argmax(dim=-1, keepdim=True)
            acc = y_pred.eq(y_true.view_as(y_pred)).sum() / y_true.shape[0]
            return acc.item()

        results = dict()
        for split in ["train", "valid", "test"]:
            split_idx = self.split_idx[split]
            acc = accuracy(logits[split_idx], y[split_idx])
            loss = F.cross_entropy(logits[split_idx], y[split_idx].view(-1)).item()
            results[f"{split}_acc"] = acc
            results[f"{split}_loss"] = loss
        return results

    def inference_and_evaluate(self, dataset):
        embs_path = os.path.join(self.trainer.args.output_dir, "cached_embs")
        logits_embs, x_embs = self.inference(dataset, embs_path)
        results = self._evaluate(logits_embs, self.data.y)
        logger.critical("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))
        gc.collect()
        torch.cuda.empty_cache()
        return logits_embs, x_embs, results  # x_embs is None in GNNTrainer

    def train_once(self):
        dist.barrier()
        if self.trial is not None:
            self.trainer._hp_search_setup(self.trial)
        train_output = self.trainer.train()
        # save outputs
        self.save_model(self.model, self.ckpt_path)
        global_step, train_dict = train_output.global_step, train_output.metrics
        train_dict["global_step"] = global_step
        self.trainer.save_metrics("train", train_dict)
        logger.critical("".join("{}:{} ".format(k, v) for k, v in train_dict.items()))
        gc.collect()
        torch.cuda.empty_cache()

    def prepare(self):
        self.model = self._prepare_model()
        self.train_set, self.valid_set, self.all_set = self._prepare_dataset()
        self.trainer = self._prepare_trainer()

    def train(self, return_value="valid"):
        self.prepare()
        assert self.args.mode in ["train", "test"]
        if self.args.mode == "train":
            self.train_once()

        logger.warning(f"\n*************** Start inference and testing ***************\n")
        _, _, results = self.inference_and_evaluate(self.all_set)
        gc.collect()
        torch.cuda.empty_cache()
        return results["test_acc"], results["valid_acc"]
