import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.transforms import SIGN

logger = logging.getLogger(__name__)


def _mask_to_idx(mask):
    return torch.range(0, mask.shape[0])[mask]


class Trainer:
    def __init__(self, args, model, data, split_idx, evaluator):
        self.args = args
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.model = self._init_model(model)
        self._precompute(data)
        self.train_loader = self._get_train_loader(data, split_idx["train"])
        self.eval_loader = self._get_eval_loader(data)
        self.optimizer = self._get_optimizer()
        self.loss_op = self._get_loss_op()

    @property
    def rank(self):
        return int(os.environ["RANK"])

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    def _precompute(self, data):
        logger.info("Precomputing data: {}".format(data))
        t_start = time.time()
        data = SIGN(self.args.gnn_num_layers)(data)
        data.xs = [data.x] + [
            data[f"x{i}"] for i in range(1, self.args.gnn_num_layers + 1)
        ]
        del data.x
        for i in range(self.args.gnn_num_layers):
            del data[f"x{i}"]
        t_end = time.time()
        logger.info("Precomputing finished, time: {}".format(t_end - t_start))

    def _init_model(self, model):
        if self.args.cont:
            model.load_state_dict(
                torch.load(self.args.ckpt_name, map_location="cpu")
            )
            logging.info("load ckpt:{}".format(self.args.ckpt_name))
        model.cuda()
        if self.rank not in [-1]:
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        return model

    def _get_train_loader(self, data, train_idx):
        xs_train = torch.cat([x[train_idx] for x in data.xs], -1)
        y_train = data.y[train_idx]
        input_ids, attention_mask = (
            data.input_ids[train_idx],
            data.attention_mask[train_idx],
        )

        train_set = TensorDataset(xs_train, input_ids, attention_mask, y_train)
        train_sampler = DistributedSampler(train_set, shuffle=True)
        is_dist = self.world_size > 0
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler if is_dist else None,
            batch_size=self.args.batch_size,
            shuffle=False if is_dist else True,
            num_workers=24,
            pin_memory=True,
        )
        return train_loader

    def _get_eval_loader(self, data):
        xs = torch.cat(data.xs, -1)
        dataset = TensorDataset(
            xs,
            data.input_ids,
            data.attention_mask,
            data.y,
            data.train_mask,
            data.valid_mask,
            data.test_mask,
        )
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )

    def _get_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def _get_loss_op(self):
        return torch.nn.CrossEntropyLoss()

    def training_step(self, xs, input_ids, attention_mask, y):
        self.optimizer.zero_grad()
        logits = self.model(xs, input_ids, attention_mask)
        loss = self.loss_op(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        t_start = time.time()
        self.model.train()
        dim_feat = self.args.num_feats
        best_acc, best_count = 0.0, 0
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            for step, batch_input in enumerate(self.train_loader):
                xs = batch_input[0]
                batch_input[0] = [
                    x.to(self.rank) for x in torch.split(xs, dim_feat, -1)
                ]
                batch_loss = self.training_step(*batch_input)
                loss += batch_loss
                dist.barrier()
            if self.rank == 0 and epoch % self.args.eval_interval == 0:
                ckpt_path = os.path.join(
                    self.args.ckpt_dir,
                    "{}-epoch-{}.pt".format(self.args.model_type, epoch + 1),
                )
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info("Saved ckpt: {}".format(ckpt_path))
                # evaluate model on validation set and test set
                train_acc, valid_acc, test_acc = self.evaluate()
                logger.info(
                    "epoch: {}, train_acc: {}, valid_acc: {}, test_acc: {}".format(
                        epoch, train_acc, valid_acc, test_acc
                    )
                )
                if valid_acc > best_acc:
                    ckpt_path = os.path.join(
                        self.args.ckpt_dir,
                        "{}-best.pt".format(self.args.model_type),
                    )
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("Best model saved to: {}".format(ckpt_path))
                    best_acc = valid_acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        t_start = time.time()
                        ckpt_path = os.path.join(
                            self.args.ckpt_dir,
                            "{}-best.pt".format(self.args.model_type),
                        )
                        self.model.load_state_dict(
                            torch.load(ckpt_path, map_location=self.rank)
                        )
                        logger.info(
                            "Start testing best model loaded from: {}".format(
                                ckpt_path
                            )
                        )
                        train_acc, valid_acc, test_acc = self.evaluate()
                        logger.info(
                            "test time: {}".format(time.time() - t_start)
                        )
                        logger.info(
                            "final tran_acc: {}, valid_acc: {}, test_acc: {}".format(
                                train_acc, valid_acc, test_acc
                            )
                        )
            dist.barrier()

        logger.info("Training finished, time: {}".format(time.time() - t_start))

    def evaluate(self):
        self.model.eval()
        dim_feat = self.args.num_feats
        train_accs, valid_accs, test_accs = [], [], []
        for step, (
            xs,
            input_ids,
            att_mask,
            y,
            train_mask,
            val_mask,
            test_mask,
        ) in enumerate(self.eval_loader):
            xs = [x.to(self.rank) for x in torch.split(xs, dim_feat, -1)]
            with torch.no_grad():
                logits = self.model(
                    xs, input_ids.to(self.rank), att_mask.to(self.rank)
                )
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = y.unsqueeze(-1)
            train_acc = self.evaluator.eval(
                {"y_true": y_true[train_mask], "y_pred": y_pred[train_mask]}
            )
            valid_acc = self.evaluator.eval(
                {"y_true": y_true[val_mask], "y_pred": y_pred[val_mask]}
            )
            test_acc = self.evaluator.eval(
                {"y_true": y_true[test_mask], "y_pred": y_pred[test_mask]}
            )
            len = xs.shape[0]
            train_accs.append(train_acc * len)
            valid_accs.append(valid_acc * len)
            test_accs.append(test_acc * len)
        len_total = len(self.eval_loader.dataset)
        train_acc = sum(train_accs) / len_total
        valid_acc = sum(valid_accs) / len_total
        test_acc = sum(test_acc) / len_total
        return train_acc, valid_acc, test_acc
