import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import optuna
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import Accuracy
from tqdm import tqdm

from .utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


def _mask_to_idx(mask):
    return torch.range(0, mask.shape[0])[mask]


class Trainer(ABC):
    def __init__(self, args, model, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        # NOTE model and metric is also initialized here
        self.model, self.metric = self._init_model_and_metric(model)
        self.train_loader = self._get_train_loader()
        self.optimizer = self._get_optimizer()
        self.loss_op = self._get_loss_op()
        self.result_dict = self._init_result_dict()
        # initialize optuna
        self.trial = kwargs.get("trial", None)

    @property
    def rank(self):
        return int(os.environ["RANK"]) if is_dist() else self.args.single_gpu

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"]) if is_dist() else 1

    def _init_result_dict(self):
        return {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "test_loss": [],
            "train_acc": [],
            "valid_acc": [],
            "test_acc": [],
            "train_time_per_epoch": [],  # time of training per epoch
            "inference_time_on_train_set": [],  # time of inference on train_set
            "inference_time_on_valid_set": [],  # time of inference on valid_set
            "inference_time_on_test_set": [],  # time of inference on test_set
        }

    def _add_result(self, dict: Dict[str, Any]):
        for key, value in dict.items():
            assert key in self.result_dict.keys()
            self.result_dict[key].append(value)

    def _load_state_dict(self, model, state_dict, strict=True, is_dist=False):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name[:7] == "module." and not is_dist:  # remove the "module." prefix
                name = name[7:]
            if name not in own_state:
                if strict:
                    raise KeyError("unexpected key '{}' in state_dict".format(name))
                else:
                    logger.warning("Ignore unexpected key '{}' in state_dict".format(name))
                    logger.warning(
                        "Please make sure the model related parameters in 'test.sh' is consistent with the 'train.sh'!"
                    )
            else:
                try:
                    own_state[name].copy_(param)
                except:  # the param shape is different between DDP and single gpu
                    own_state[name].copy_(param.squeeze(0))

    def _init_model_and_metric(self, model):
        if self.args.cont:
            ckpt_name = os.path.join(self.args.ckpt_dir, self.args.ckpt_name)
            ckpt = torch.load(ckpt_name, map_location="cpu")
            self._load_state_dict(model, ckpt, is_dist=False)
            logging.info("load ckpt:{}".format(ckpt_name))
        metric = Accuracy(task="multiclass", num_classes=self.args.num_labels)
        model.metric = metric
        model.to(self.rank)
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        return model, metric

    @abstractmethod
    def _get_train_loader(self):
        pass

    @abstractmethod
    def _get_eval_loader(self, mode):
        pass

    def save_model(self, ckpt_name):
        if self.rank in [0, -1]:
            ckpt_path = os.path.join(self.args.ckpt_dir, ckpt_name)
            torch.save(self.model.state_dict(), ckpt_path)
            logger.info("Save ckpt to: {}".format(ckpt_path))
        if is_dist():
            dist.barrier()

    def _get_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def _get_loss_op(self):
        return torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

    def training_step(self, *inputs, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        inputs, y = inputs[:-1], inputs[-1]
        logits = self.model(*inputs)
        loss = self.loss_op(logits, y.to(self.rank))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _list_tensor_to_gpu(self, list: List):
        return [input.to(self.rank) if isinstance(input, torch.Tensor) else input for input in list]

    def train(self):
        t_start = time.time()
        best_acc, best_count = 0.0, 0
        disable_tqdm = self.args.disable_tqdm or (is_dist() and self.rank > 0)
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            t_start_epoch = time.time()
            for step, batch_input in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=disable_tqdm)
            ):
                batch_input = self._list_tensor_to_gpu(batch_input)
                batch_loss = self.training_step(*batch_input)
                loss += batch_loss
                dist.barrier()
            t_end_epoch = time.time()
            train_time_per_epoch = t_end_epoch - t_start_epoch
            loss /= len(self.train_loader)
            # evalutation and early stop
            if (epoch + 1) % self.args.eval_interval == 0:
                if self.args.save_ckpt_per_valid:
                    ckpt_name = "{}-epoch-{}.pt".format(self.args.model_type, epoch + 1)
                    self.save_model(ckpt_name)
                train_acc, train_loss, train_time = self.evaluate(mode="train")
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
                logger.info("".join("{}:{} ".format(k, v) for k, v in result.items()))
                self._add_result(result)
                if self.args.optuna and self.trial is not None:
                    self.trial.report(valid_acc, epoch + 1)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                # record loss and accs
                # early stop
                if valid_acc > best_acc:
                    ckpt_name = "{}-best.pt".format(self.args.model_type)
                    self.save_model(ckpt_name)
                    best_acc = valid_acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        break

        if self.args.optuna and self.trial is not None:
            return best_acc

        ckpt_path = os.path.join(
            self.args.ckpt_dir,
            "{}-best.pt".format(self.args.model_type),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._load_state_dict(self.model, ckpt, is_dist=True)
        self.model.to(self.rank)
        logger.info("Start testing best model loaded from: {}".format(ckpt_path))
        test_acc, test_loss, test_time = self.evaluate(mode="test")
        result = {"test_loss": test_loss, "test_acc": test_acc, "inference_time_on_test_set": test_time}
        logger.info("".join("{}:{} ".format(k, v) for k, v in result.items()))
        self._add_result(result)
        logger.info("Training finished, time: {}".format(time.time() - t_start))

    def evaluate(self, mode="test"):
        assert mode in ["train", "test", "valid"]
        self.model.eval()
        eval_loader = self._get_eval_loader(mode)
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
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = y_true.to(self.rank)
            acc = self.metric(y_pred, y_true)
            loss += self.loss_op(logits, y_true.squeeze(-1)).item()
            pbar.update(1)
        t_end_eval = time.time()
        acc = self.metric.compute()
        loss /= len(eval_loader)
        del eval_loader
        return acc, loss, t_end_eval - t_start_eval
