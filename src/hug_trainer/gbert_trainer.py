import gc
import logging
import os
import os.path as osp
import shutil
from typing import Union

import optuna
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.transforms import ToSparseTensor

from ..utils import mkdirs_if_not_exists
from .gnn_trainer import GNNTrainer
from .lm_trainer import LMTrainer
from .trainer import Trainer

logger = logging.getLogger(__name__)


class SLE:
    def __init__(self, args, data, train_idx):
        """
        ground_truth: train_idx, y_true
        pesudos: pesudo_train_mask, pesudo_y
        for propogation: y_embs
        """
        self.enabled = args.use_SLE
        self.threshold = args.SLE_threshold
        self.num_labels = args.num_labels
        self.num_layers = args.gnn_num_layers
        # save ground truth
        self.y_true = data.y.view(-1)
        self.train_mask = torch.zeros_like(self.y_true, dtype=torch.bool)
        self.train_mask[train_idx] = True
        self.adj_t = data.adj_t

        # set pesudo train mask
        self.pesudo_train_mask = self.train_mask.clone()
        # for self training
        self.pesudo_y = torch.zeros_like(self.y_true)
        self.pesudo_y[train_idx] = self.y_true[train_idx]
        logger.info("Initialize pseudo labels, rate: {:.4f}".format(self.pesudo_label_rate))

        # y (label) propogation
        self.y_emb = self.compute_y_emb()

    def compute_y_emb(self):
        y_emb = torch.zeros(self.y_true.size(0), self.num_labels)
        y_emb[self.pesudo_train_mask] = F.one_hot(
            self.pesudo_y[self.pesudo_train_mask], num_classes=self.num_labels
        ).float()
        y_emb[self.train_mask] = F.one_hot(self.y_true[self.train_mask], num_classes=self.num_labels).float()
        for _ in range(self.num_layers):
            y_emb = self.adj_t @ y_emb
        return y_emb

    @property
    def pesudo_train_idx(self):
        return self.pesudo_train_mask.nonzero().view(-1)

    @property
    def pesudo_label_rate(self):
        return self.pesudo_train_mask.sum() / self.pesudo_train_mask.shape[0]

    def update(self, logits):
        if not self.enabled:
            return
        # self training
        val, pred = torch.max(F.softmax(logits, dim=1), dim=1)
        SLE_mask = val > self.threshold
        SLE_pred = pred[SLE_mask]
        self.pesudo_train_mask = SLE_mask | self.pesudo_train_mask
        self.pesudo_y[SLE_mask] = SLE_pred
        self.pesudo_y[self.train_mask] = self.y_true[self.train_mask]
        logger.info("Update pseudo labels, rate: {:.4f}".format(self.pesudo_label_rate))
        self.y_emb = self.compute_y_emb()


class GBertTrainer:
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.trial = kwargs.pop("trial", None)
        self.lm_trainer = LMTrainer(args, data, split_idx, evaluator, **kwargs)
        self.gnn_trainer = GNNTrainer(args, data, split_idx, evaluator, **kwargs)
        self.edge_index_to_adj_t()
        self.sle = SLE(args, data, split_idx["train"])

    def edge_index_to_adj_t(self):
        self.data = ToSparseTensor()(self.data)
        deg = self.data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self.data.adj_t = deg_inv_sqrt.view(-1, 1) * self.data.adj_t * deg_inv_sqrt.view(1, -1)

    def run_once(self, iter, trainer: Union[GNNTrainer, LMTrainer]):
        # "lm" only affects lm_trainer, this value does not take effect for gnn_trainer
        # see gnn_trainer.py for implementation
        # ckpt_path = trainer.ckpt_path(iter, "lm")
        # if self.args.use_cache and osp.exists(ckpt_path):
        #     logger.info(f"use cached ckpt instead")
        # else:
        trainer.train_once(iter)
        trainer.all_set = trainer._get_dataset("all")
        logger.info(f"iter: {iter}: start inference and evaluation")
        return trainer.inference_and_evaluate()

    def train(self):
        best_valid_acc = 0.0
        best_count = 0
        for iter in range(self.args.num_iterations):
            self.iter = iter
            logger.critical(f"\n*************** Start iter {iter} training ***************\n")
            logger.info(f"*************** Start LM training ***************")
            self.lm_trainer.update_data(self.data, self.split_idx)
            logits, hidden_features, results = self.run_once(iter, self.lm_trainer)
            self.lm_trainer.next_iter()
            if self.trial is not None:
                if results["valid_acc"] < self.args.expected_valid_acc or self.trial.should_prune():
                    logger.critical(
                        f"valid acc {results['valid_acc']:.4f} is lower than expected {self.args.expected_valid_acc:.4f}"
                    )
                    raise optuna.exceptions.TrialPruned()

            logger.info(f"propogating hidden features of {self.args.lm_type} on the graph")
            # preserve data.x for KL divergence loss
            num_layers = self.args.gnn_num_layers
            self.data.x = hidden_features

            # hidden features propogation using self.data.adj_t
            xs = [self.data.x]
            for i in range(1, num_layers + 1):
                xs += [self.data.adj_t @ xs[-1]]
            xs = xs[1:]  # remove the hop-0 feature, which is saved in self.data.x
            self.data.x_emb = torch.cat(xs, dim=-1)

            if self.args.use_SLE and self.args.SLE_mode in ["both", "lm"]:
                self.sle.update(logits)
                self.data.sle = self.sle

            gc.collect()
            torch.cuda.empty_cache()

            logger.info("*************** Start GNN training ***************")
            self.gnn_trainer.update_data(self.data, self.split_idx)
            logits, _, results = self.run_once(iter, self.gnn_trainer)
            valid_acc = results["valid_acc"]
            self.gnn_trainer.next_iter()

            if self.args.use_SLE and self.args.SLE_mode in ["both", "gnn"]:
                self.sle.update(logits)
                self.data.sle = self.sle

            if valid_acc > best_valid_acc:
                best_path = os.path.join(self.args.output_dir, "best", "ckpt")
                mkdirs_if_not_exists(best_path)
                shutil.copyfile(self.lm_trainer.ckpt_path(iter, "lm"), os.path.join(best_path, "bert_model.pt"))
                shutil.copyfile(self.gnn_trainer.ckpt_path(iter, "gnn"), os.path.join(best_path, "lm_model.pt"))
                best_valid_acc = valid_acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= 2:
                    logger.warning(f"early stop at iter {iter} with best valid acc {best_valid_acc:.4f}")
                    return best_valid_acc

        return best_valid_acc
