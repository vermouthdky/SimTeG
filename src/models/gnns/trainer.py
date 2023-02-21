import logging
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.transforms import SIGN
from tqdm import tqdm

import optuna

from ...trainer import Trainer
from ...utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


class GNN_Trainer(Trainer):
    def __init__(self, args, model, data, split_idx, evaluator, **kwargs):
        data = self._precompute(data, args.gnn_num_layers)
        super(GNN_Trainer, self).__init__(args, model, data, split_idx, evaluator, **kwargs)

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
        return self._get_dataloader(train_set, self.args.batch_size, shuffle=True)

    def _get_eval_loader(self, mode="test"):
        data = self.data
        assert mode in ["train", "valid", "test"]
        eval_mask = data[f"{mode}_mask"]
        xs_eval = torch.cat([x[eval_mask] for x in data.xs], -1)
        dataset = TensorDataset(xs_eval, data.y[eval_mask])
        return self._get_dataloader(dataset, self.args.eval_batch_size, shuffle=False)
