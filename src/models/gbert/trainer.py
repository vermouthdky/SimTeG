import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ...trainer import Trainer
from ...utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


class Gbert_Trainer(Trainer):
    def __init__(self, args, model, data, split_idx, evaluator):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        # NOTE model and metric is also initialized here
        self.model, self.metric = self._init_model_and_metric(model)
        self._init_pseudos()
        # initialize hidden_features for propogation
        self.propogated_x = None
        self.train_loader = self._get_train_loader()  # use pseudo labels
        self.optimizer = self._get_optimizer()
        self.loss_op = self._get_loss_op()

    def _init_pseudos(self):
        self.pseudo_train_idx = self.split_idx["train"].clone()
        # for label propagation
        self.y_embs = torch.zeros(self.data.y.size(0), self.args.num_labels)
        self.y_embs[self.pseudo_train_idx] = F.one_hot(
            self.data.y[self.pseudo_train_idx],
            num_classes=self.args.num_labels,
        ).float()
        # for self training
        self.pseudo_y = torch.zeros_like(self.data.y)
        self.pseudo_y[self.peusdo_train_idx] = self.data.y[self.peusdo_train_idx]

    def _get_train_loader(self):
        y_train = self.pseudo_y[self.pseudo_train_idx].squeeze(-1)
        input_ids, attention_mask = (
            self.data.input_ids[self.pseudo_train_idx],
            self.data.attention_mask[self.pseudo_train_idx],
        )
        propogated_x = self.propogated_x[self.pseudo_train_idx] if self.propogated_x is not None else None
        train_set = TensorDataset(propogated_x, input_ids, attention_mask, y_train)
        train_loader = DataLoader(
            train_set,
            sampler=DistributedSampler(train_set, shuffle=True) if is_dist() else None,
            batch_size=self.args.batch_size,
            shuffle=False if is_dist() else True,
            num_workers=48,
            pin_memory=True,
        )
        return train_loader

    def inference_and_evaluate(self, data):
        dataset = TensorDataset(self.propogated_x, data.input_ids, data.attention_mask, data.y)
        dataloader = DataLoader(
            dataset,
            sampler=DistributedSampler(dataset, shuffle=False) if is_dist() else None,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )
        hidden_features_list = []
        disable_tqdm = self.args.disable_tqdm or (is_dist() and self.rank > 0)
        pbar = tqdm(dataloader, desc="Inference and evaluating", disable=disable_tqdm)
        self.metric.reset()
        for step, batch_input in enumerate(dataloader):
            batch_input = self._list_tensor_to_gpu(batch_input)
            hidden_features = self.inference_step(*batch_input)
            hidden_features_list.append(hidden_features)

    def train_one_iteration(self):
        super().train()
        t_start = time.time()
        best_acc, best_count = 0.0, 0
        disable_tqdm = self.args.disable_tqdm or (is_dist() and self.rank > 0)
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            for step, batch_input in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=disable_tqdm)
            ):
                batch_input = self._list_tensor_to_gpu(batch_input)
                batch_loss = self.training_step(*batch_input)
                loss += batch_loss
                dist.barrier()
            loss /= len(self.train_loader)
            # evalutation and early stop
            if (epoch + 1) % self.args.eval_interval == 0:
                if self.args.save_ckpt_per_valid:
                    ckpt_name = "{}-epoch-{}.pt".format(self.args.model_type, epoch + 1)
                    self.save_model(ckpt_name)
                train_acc = self.evaluate(mode="train")
                valid_acc = self.evaluate(mode="valid")
                logger.info(
                    "epoch: {}, loss: {}, train_acc:{}, valid_acc: {}".format(epoch + 1, loss, train_acc, valid_acc)
                )
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

        test_t_start = time.time()
        ckpt_path = os.path.join(
            self.args.ckpt_dir,
            "{}-best.pt".format(self.args.model_type),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._load_state_dict(self.model, ckpt, is_dist=True)
        self.model.to(self.rank)
        logger.info("Start testing best model loaded from: {}".format(ckpt_path))
        test_acc = self.evaluate(mode="test")
        logger.info("test time: {}".format(time.time() - test_t_start))
        logger.info("final test_acc: {}".format(test_acc))
        logger.info("Training finished, time: {}".format(time.time() - t_start))
