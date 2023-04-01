import gc
import logging
import os
import os.path as osp
import shutil
from typing import Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.transforms import SIGN

from .gnn_trainer import GNNTrainer
from .lm_trainer import LMTrainer
from .trainer import Trainer

logger = logging.getLogger(__name__)


class GBertTrainer:
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.trial = kwargs.pop("trial", None)
        self.lm_trainer = LMTrainer(args, data, split_idx, evaluator, **kwargs)
        self.gnn_trainer = GNNTrainer(args, data, split_idx, evaluator, **kwargs)

    def run_once(self, iter, trainer: Union[GNNTrainer, LMTrainer]):
        # "lm" only affects lm_trainer, this value does not take effect for gnn_trainer
        # see gnn_trainer.py for implementation
        ckpt_path = trainer.ckpt_path(iter, "lm")
        if self.args.use_cache and osp.exists(ckpt_path):
            logger.info(f"use cached ckpt instead")
        else:
            trainer.train_once(iter)
        trainer.all_set = trainer._get_dataset("all")
        logger.info("iter: {iter}: start inference and evaluation")
        hidden_features, results = trainer.inference_and_evaluate()
        return hidden_features, results

    def train(self):
        best_valid_acc = 0.0
        best_count = 0
        for iter in range(self.args.num_iterations):
            self.iter = iter
            logger.critical(f"\n*************** Start iter {iter} training ***************\n")
            logger.info(f"*************** Start LM training ***************")
            self.lm_trainer.update_data(self.data, self.split_idx)
            hidden_features, results = self.run_once(iter, self.lm_trainer)

            logger.info(f"propogating hidden features of {self.args.lm_type} on the graph")
            # preserve data.x for KL divergence loss
            num_layers = self.args.gnn_num_layers
            self.data.x = hidden_features
            self.data = SIGN(num_layers)(self.data)
            self.data.x_emb = torch.cat([self.data[f"x{i}"] for i in range(1, num_layers + 1)], dim=-1)
            for i in range(1, num_layers + 1):
                del self.data[f"x{i}"]
            gc.collect()
            torch.cuda.empty_cache()

            logger.warning("*************** Start GNN training ***************")
            self.gnn_trainer.update_data(self.data, self.split_idx)
            _, results = self.run_once(iter, self.gnn_trainer)
            valid_acc = results["valid_acc"]

            if valid_acc > best_valid_acc:
                best_path = os.path.join(self.args.output_dir, "best", "ckpt")
                if not os.path.exists(best_path):
                    os.makedirs(best_path)
                shutil.copyfile(self.lm_trainer.ckpt_path(iter, "lm"), os.path.join(best_path, "bert_model.pt"))
                shutil.copyfile(self.gnn_trainer.ckpt_path(iter, "gnn"), os.path.join(best_path, "lm_model.pt"))
                best_valid_acc = valid_acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= 2:
                    return best_valid_acc

        return best_valid_acc
