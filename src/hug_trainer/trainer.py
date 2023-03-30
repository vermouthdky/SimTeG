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
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ..model import get_model_class
from ..utils import is_dist

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

    def save_model(self, model: torch.nn.Module, ckpt_name):
        if self.rank <= 0:
            ckpt_path = os.path.join(self.args.ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved the model to {}".format(ckpt_path))
        if is_dist():
            dist.barrier()

    def load_model(self, model: torch.nn.Module, ckpt_name):
        ckpt_path = os.path.join(self.args.ckpt_dir, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    @abstractmethod
    def _get_dataset(self, mode):
        pass

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid")

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

    def inference_and_evaluate(self):
        embs_path = os.path.join(self.args.output_dir, "cached_embs")
        if not os.path.exists(embs_path):
            os.makedirs(embs_path)

        def has_embs(saved_name: Union[str, list[str]]):
            if isinstance(saved_name, list):
                return all([has_embs(name) for name in saved_name])
            return os.path.exists(os.path.join(embs_path, saved_name))

        def save_embs(embs: torch.Tensor, saved_name: str):
            if self.rank == 0:
                torch.save(embs, os.path.join(embs_path, saved_name))
                logger.info(f"Saved {saved_name} to {embs_path}")
            dist.barrier()

        def load_embs(saved_name: str):
            return torch.load(os.path.join(embs_path, saved_name))

        def evalute(logits, y_true):
            y_pred = logits.argmax(dim=-1, keepdim=True)
            acc = y_pred.eq(y_true.view_as(y_pred)).sum() / y_true.shape[0]
            return acc.item()

        x_embs_name = f"iter_{self.iter}_x_embs.pt"
        logits_name = f"iter_{self.iter}_logits.pt"
        if self.args.use_cache and has_embs([x_embs_name, logits_name]):
            logger.info("Loading cached embs...")
            x_embs = load_embs(x_embs_name)
            logits_embs = load_embs(logits_name)
        else:
            eval_output = self.trainer.predict(self.all_set)
            logits_embs, x_embs = eval_output.predictions[0], eval_output.predictions[1]
            logits_embs, x_embs = torch.from_numpy(logits_embs), torch.from_numpy(x_embs)

        save_embs(x_embs, x_embs_name)
        save_embs(logits_embs, logits_name)

        results = dict()
        for split in ["train", "valid", "test"]:
            split_idx = self.split_idx[split]
            acc = evalute(logits_embs[split_idx], self.data.y[split_idx])
            loss = F.cross_entropy(logits_embs[split_idx], self.data.y[split_idx].view(-1)).item()
            results[f"{split}_acc"] = acc
            results[f"{split}_loss"] = loss
        logger.critical("".join("{}:{} ".format(k, v) for k, v in results.items()))
        del logits_embs
        gc.collect()
        torch.cuda.empty_cache()
        return x_embs, results

    def train_once(self, iter):
        dist.barrier()
        if self.trial is not None:
            self.trainer._hp_search_setup(self.trial)
        train_output = self.trainer.train()  # none if not using optuna
        self.save_model(self.model, f"iter_{iter}.pt")
        global_step, train_dict = train_output.global_step, train_output.metrics
        train_dict["global_step"] = global_step
        self.trainer.save_metrics("train", train_dict)
        # print train_dict
        logger.critical("".join("{}:{} ".format(k, v) for k, v in train_dict.items()))
        eval_dict = self.trainer.evaluate()
        self.trainer.save_metrics("eval", eval_dict)
        logger.critical("".join("{}:{} ".format(k, v) for k, v in eval_dict.items()))
        return eval_dict["eval_accuracy"]

    def train(self):
        self.model = self._prepare_model()
        self.train_set, self.valid_set = self._prepare_dataset()
        self.all_set = self._get_dataset("all")
        self.trainer = self._prepare_trainer()
        iter = self.iter = 0

        ckpt_path = os.path.join(self.args.ckpt_dir, "iter_0")
        if self.args.use_cache and os.path.exists(ckpt_path):
            logger.warning(f"\n*********iter {iter} has been trained, use cached ckpt instead!*********\n")
        else:
            valid_acc = self.train_once(iter)
            logger.critical("Iter {} Best valid acc: {:.4f}".format(iter, valid_acc))

        logger.warning(f"\n*************** Start iter {iter} testing ***************\n")
        # NOTE inference for SLE and propogation
        _, results = self.inference_and_evaluate()

        valid_acc = results["valid_acc"]
        gc.collect()
        torch.cuda.empty_cache()
        return valid_acc
