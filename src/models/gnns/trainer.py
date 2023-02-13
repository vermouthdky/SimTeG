import logging
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.transforms import SIGN
from tqdm import tqdm

from ...trainer import Trainer
from ...utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


class GNN_Trainer(Trainer):
    def __init__(self, args, model, data, split_idx, evaluator):
        data = self._precompute(data, args.gnn_num_layers)
        super(GNN_Trainer, self).__init__(args, model, data, split_idx, evaluator)

    def _precompute(self, data, num_layers):
        logger.info("Precomputing data: {}".format(data))
        t_start = time.time()
        data = SIGN(num_layers)(data)
        data.xs = [data.x] + [data[f"x{i}"] for i in range(1, num_layers + 1)]
        del data.x
        for i in range(num_layers):
            del data[f"x{i}"]
        t_end = time.time()
        logger.info("Precomputing finished, time: {}".format(t_end - t_start))
        return data

    def _get_train_loader(self):
        train_idx = self.split_idx["train"]
        data = self.data
        y_train = data.y[train_idx].squeeze(-1)
        xs_train = torch.cat([x[train_idx] for x in data.xs], -1)
        train_set = TensorDataset(xs_train, y_train)
        train_loader = DataLoader(
            train_set,
            sampler=DistributedSampler(train_set, shuffle=True) if is_dist() else None,
            batch_size=self.args.batch_size,
            shuffle=False if is_dist() else True,
            num_workers=48,
            pin_memory=True,
        )
        return train_loader

    def _get_eval_loader(self, mode="test"):
        data = self.data
        assert mode in ["train", "valid", "test"]
        eval_mask = data[f"{mode}_mask"]
        xs_eval = torch.cat([x[eval_mask] for x in data.xs], -1)
        dataset = TensorDataset(xs_eval, data.y[eval_mask])
        return DataLoader(
            dataset,
            # TODO: check if it works
            sampler=DistributedSampler(dataset, shuffle=False) if is_dist() else None,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )

    def train(self):
        t_start = time.time()
        dim_feat = self.args.num_feats
        best_acc, best_count = 0.0, 0
        disable_tqdm = self.args.disable_tqdm or (is_dist() and self.rank > 0)
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            for step, batch_input in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=disable_tqdm)
            ):
                xs = batch_input[0]
                batch_input[0] = [x.to(self.rank) for x in torch.split(xs, dim_feat, -1)]
                batch_input = self._list_tensor_to_gpu(batch_input)
                batch_loss = self.training_step(*batch_input)
                loss += batch_loss
                dist.barrier()

            loss /= len(self.train_loader)

            # evalutation and early stop
            if (epoch + 1) % self.args.eval_interval == 0:
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

    def evaluate(self, mode="test"):
        assert mode in ["train", "test", "valid"]
        self.model.eval()
        dim_feat = self.args.num_feats
        eval_loader = self._get_eval_loader(mode)
        disable_tqdm = self.args.disable_tqdm or (is_dist() and self.rank > 0)
        pbar = tqdm(total=len(eval_loader), desc=f"evaluating {mode} set", disable=disable_tqdm)
        # NOTE torchmetrics support distributed inference
        self.metric.reset()
        for step, (xs, y_true) in enumerate(eval_loader):
            xs = [x.to(self.rank) for x in torch.split(xs, dim_feat, -1)]
            with torch.no_grad():
                logits = self.model(xs)
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = y_true.to(self.rank)
            acc = self.metric(y_pred, y_true)
            pbar.update(1)
        acc = self.metric.compute()
        return acc
