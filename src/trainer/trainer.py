import gc
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import get_scheduler

import optuna

from ..model import get_model_class
from ..utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


def _mask_to_idx(mask):
    return torch.range(0, mask.shape[0])[mask]


class Trainer(ABC):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.loss_op = self._get_loss_op()
        self.result_dict = {}
        # initialize optuna
        self.trial = kwargs.get("trial", None)

    @property
    def rank(self):
        return int(os.environ["RANK"]) if is_dist() else self.args.single_gpu

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"]) if is_dist() else 1

    @property
    def disable_tqdm(self):
        return self.args.disable_tqdm or (is_dist() and self.rank > 0)

    def _add_result(self, key: str, value: Dict[str, Any]):
        self.result_dict[key] = value

    def save_result(self, path: str):
        if self.rank in [0, -1]:
            w_path = os.path.join(path, "results.json")
            with open(w_path, "w") as f:
                json.dump(self.result_dict, f)
            logger.info("Save ckpt to: {}".format(w_path))
        if is_dist():
            dist.barrier()

    def _load_state_dict(self, model, state_dict, strict=True, is_dist=False):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name[:7] == "module." and not is_dist:  # remove the "module." prefix
                name = name[7:]
            if name not in own_state:
                if strict:
                    raise KeyError("unexpected key '{}' in model (own_state)".format(name))
                else:
                    logger.warning("Ignore unexpected key '{}' in state_dict".format(name))
                    # logger.warning(
                    #     "Please make sure the model related parameters in 'test.sh' is consistent with the 'train.sh'!"
                    # )
            else:
                try:
                    own_state[name].copy_(param)
                except:  # the param shape is different between DDP and single gpu
                    own_state[name].copy_(param.squeeze(0))

    def _init_model_and_metric(self, model):
        metric = Accuracy(task="multiclass", num_classes=self.args.num_labels)
        model.metric = metric
        model.to(self.rank)
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                # find_unused_parameters=True,
            )
        return model, metric

    @abstractmethod
    def _get_train_loader(self):
        pass

    @abstractmethod
    def _get_eval_loader(self, mode):
        pass

    def _get_dataloader(self, dataset: TensorDataset, batch_size: int, shuffle: bool):
        """
        return a dataloader with DistributedSampler:
        NOTE for inference, you have to use 'dist.gather()' to gather the results from all gpus
        """
        sampler = DistributedSampler(dataset, shuffle=shuffle) if is_dist() else None
        shuffle = shuffle if sampler is None else False
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle)

    def save_model(self, ckpt_name):
        if self.rank in [0, -1]:
            ckpt_path = os.path.join(self.args.ckpt_dir, ckpt_name)
            torch.save(self.model.state_dict(), ckpt_path)
            logger.info("Save ckpt to: {}".format(ckpt_path))
        if is_dist():
            dist.barrier()

    def _get_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _get_loss_op(self):
        return torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

    def _get_scheduler(self):
        num_training_steps_per_epoch = len(self.train_loader)
        num_training_steps = self.args.epochs * num_training_steps_per_epoch
        num_warmup_steps = int(self.args.scheduler_warmup_ratio * num_training_steps)
        return get_scheduler(
            name=self.args.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def training_step(self, *inputs, **kwargs):
        self.model.train()
        # self.optimizer.zero_grad()
        inputs, y = inputs[:-1], inputs[-1]
        logits = self.model(*inputs)
        loss = self.loss_op(logits, y.to(self.rank))
        loss.backward()
        step = kwargs.get("step", 1)
        accum_interval = kwargs.get("accum_interval", 1)
        batch_len = kwargs.get("batch_len", 1)
        if ((step + 1) % accum_interval == 0) or ((step + 1) == batch_len):
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.item()

    def _list_tensor_to_gpu(self, list: List):
        return [input.to(self.rank) if isinstance(input, torch.Tensor) else input for input in list]

    def train_once(self, iter=0):
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_scheduler()
        best_acc, best_count = 0.0, 0
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            t_start_epoch = time.time()
            for step, batch_input in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=self.disable_tqdm)
            ):
                batch_input = self._list_tensor_to_gpu(batch_input)
                kwargs = {"step": step, "accum_interval": self.args.accum_interval, "len_batch": len(self.train_loader)}
                batch_loss = self.training_step(*batch_input, **kwargs)
                loss += batch_loss
                dist.barrier()
                t_end_epoch = time.time()
                train_time_per_epoch = t_end_epoch - t_start_epoch
                loss /= len(self.train_loader)
            # evalutation and early stop
            if (epoch + 1) % self.args.eval_interval == 0:
                if self.args.save_ckpt_per_valid:
                    ckpt_name = "{}_iter_{}_epoch_{}.pt".format(self.args.model_type, iter, epoch + 1)
                    self.save_model(ckpt_name)
                if self.args.eval_train_set:
                    train_acc, train_loss, train_time = self.evaluate(mode="train")
                else:
                    train_acc, train_loss, train_time = -1, -1, -1
                valid_acc, valid_loss, valid_time = self.evaluate(mode="valid")
                result = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                    "train_time_per_epoch": train_time_per_epoch,
                    "inference_time_on_train_set": train_time,
                    "inference_time_on_valid_set": valid_time,
                }
                logger.warning("".join("{}:{:.4f} ".format(k, v) for k, v in result.items()))
                self._add_result(f"iter_{iter}_epoch_{epoch+1}", result)
                if self.args.optuna and self.trial is not None:
                    self.trial.report(valid_acc, epoch + 1)
                    # if not iterative pruning, prune it if necessary
                    # else we do pruning in the outer loop
                    if iter == -1 and self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                # record loss and accs
                # explictly early stop
                if valid_acc > best_acc:
                    ckpt_name = "{}_iter_{}_best.pt".format(self.args.model_type, iter)
                    self.save_model(ckpt_name)
                    best_acc = valid_acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        return best_acc
        return best_acc  # best valid acc

    def train(self):
        # NOTE model and metric is also initialized here
        model = get_model_class(self.args.model_type, use_adapter=self.args.use_adapter)(self.args)
        self.model, self.metric = self._init_model_and_metric(model)
        self.train_loader = self._get_train_loader()
        self.eval_loader = {
            "train": self._get_eval_loader("train"),
            "valid": self._get_eval_loader("valid"),
            "test": self._get_eval_loader("test"),
        }
        t_start = time.time()
        best_acc = self.train_once()
        if self.args.optuna and self.trial is not None:
            self.save_result(self.args.output_dir)
            return best_acc

        ckpt_path = os.path.join(
            self.args.ckpt_dir,
            "{}_iter_{}_best.pt".format(self.args.model_type, 0),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._load_state_dict(self.model, ckpt, is_dist=True)
        self.model.to(self.rank)
        logger.info("Start testing best model loaded from: {}".format(ckpt_path))
        result = {}
        for mode in ["train", "valid", "test"]:
            eval_acc, eval_loss, eval_time = self.evaluate(mode=mode)
            result["{}_acc".format(mode)] = eval_acc
            result["{}_loss".format(mode)] = eval_loss
            result["{}_time".format(mode)] = eval_time

        logger.critical("".join("{}:{} ".format(k, v) for k, v in result.items()))
        self._add_result("final_test", result)
        self.save_result(self.args.output_dir)
        logger.critical("Training finished, time: {}".format(time.time() - t_start))
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, mode="test"):
        assert mode in ["train", "test", "valid"]
        self.model.eval()
        eval_loader = self.eval_loader[mode]
        disable_tqdm = self.args.disable_tqdm or (is_dist() and self.rank > 0)
        pbar = tqdm(total=len(eval_loader), desc=f"evaluating {mode} set", disable=disable_tqdm)
        # NOTE torchmetrics support distributed inference
        self.metric.reset()
        loss = 0.0
        t_start_eval = time.time()
        for step, batch_input in enumerate(eval_loader):
            batch_input = self._list_tensor_to_gpu(batch_input)
            batch_input, y_true = batch_input[:-1], batch_input[-1]
            with torch.no_grad():
                logits = self.model(*batch_input)
            y_pred = logits.argmax(dim=-1).view_as(y_true)
            y_true = y_true.to(self.rank)
            acc = self.metric(y_pred, y_true)
            loss += self.loss_op(logits, y_true.squeeze(-1)).item()
            pbar.update(1)
        t_end_eval = time.time()
        acc = self.metric.compute().item()
        loss /= len(eval_loader)
        return acc, loss, t_end_eval - t_start_eval