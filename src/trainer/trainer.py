import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
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

    def next_iter(self):
        self.iter += 1

    def save_model(self, model: torch.nn.Module, ckpt_path):
        if self.rank <= 0:
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved the model to {}".format(ckpt_path))
        if is_dist():
            dist.barrier()

    def load_model(self, model: torch.nn.Module, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    @abstractmethod
    def _get_dataset(self, mode):
        pass

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid")

    def update_data(self, data, split_idx):
        self.data = data
        self.split_idx = split_idx

    def _get_dataloader(self, dataset, batch_size, shuffle, enumerate=False):
        """
        return a dataloader with DistributedSampler:
        NOTE for inference, you have to use 'dist.gather()' to gather the results from all gpus
        """
        sampler = DistributedSampler(dataset, shuffle=shuffle) if is_dist() else None
        shuffle = shuffle if sampler is None else False
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle)

    @abstractmethod
    def _prepare_model(self):
        pass

    @abstractmethod
    def _prepare_trainer(self):
        pass

    @abstractmethod
    def inference(self, dataset, embs_path):
        pass

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

    def inference_and_evaluate(self):
        embs_path = os.path.join(self.trainer.args.output_dir, "cached_embs")
        logits_embs, x_embs = self.inference(self.all_set, embs_path)
        results = self._evaluate(logits_embs, self.data.y)
        logger.critical("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))
        gc.collect()
        torch.cuda.empty_cache()
        return logits_embs, x_embs, results  # x_embs is None in GNNTrainer

    @abstractmethod
    def train_once(self, iter):
        pass

    def train(self):
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

        valid_acc = results["valid_acc"]
        gc.collect()
        torch.cuda.empty_cache()
        return valid_acc
