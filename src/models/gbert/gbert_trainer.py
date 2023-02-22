import gc
import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ...trainer import Trainer
from ...utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


class SLE_Data:
    def __init__(self, train_idx, y_true, num_labels, use_SLE=True):
        """
        ground_truth: train_idx, y_true
        pesudos: pesudo_train_mask, pesudo_y
        for propogation: y_embs
        """
        # save ground truth
        self.y_true = y_true.view(-1)
        self.train_mask = torch.zeros_like(self.y_true, dtype=torch.bool)
        self.train_mask[train_idx] = True
        # set pesudo train mask
        self.pesudo_train_mask = self.train_mask.clone()
        # for self training
        self.pesudo_y = torch.zeros_like(self.y_true)
        self.pesudo_y[train_idx] = self.y_true[train_idx]
        # for x (feature) and y (label) propogation
        self.y_embs = None
        if use_SLE:
            self.y_embs = torch.zeros(self.y_true.size(0), num_labels)
            self.y_embs[self.train_mask] = F.one_hot(
                self.y_true[self.train_mask],
                num_classes=num_labels,
            ).float()
        logger.info("Initialized pseudo labels, rate: {:.4f}".format(self.pesudo_label_rate))

    @property
    def pesudo_label_rate(self):
        return self.pesudo_train_mask.sum() / self.pesudo_train_mask.shape[0]

    def update(self, SLE_mask, SLE_y, adj_t):
        # self training
        self.pesudo_train_mask = SLE_mask | self.pesudo_train_mask
        self.pesudo_y[SLE_mask] = SLE_y
        self.pesudo_y[self.train_mask] = self.y_true[self.train_mask]
        # label propogation
        self.y_embs = adj_t @ self.y_embs


class SupportNoneTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        tensors = [t for t in tensors if t is not None]
        super(SupportNoneTensorDataset, self).__init__(*tensors)


class GBert_Trainer(Trainer):
    def __init__(self, args, model, data, split_idx, evaluator, **kwargs):
        self.sle_data = SLE_Data(split_idx["train"], data.y, args.num_labels, args.use_SLE)
        super(GBert_Trainer, self).__init__(args, model, data, split_idx, evaluator, **kwargs)
        self.inference_dataloader = self._get_inference_dataloader()
        self.adj_t = self._init_adj_t(data.adj_t)
        self.x_embs = None

    def _init_adj_t(self, adj_t):
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def _get_inference_dataloader(self):
        inference_set = SupportNoneTensorDataset(
            self.data.input_ids, self.data.attention_mask, self.x_embs, self.sle_data.y_embs
        )
        return DataLoader(inference_set, batch_size=self.args.eval_batch_size, shuffle=False)

    def _get_train_loader(self):
        train_mask = self.sle_data.pesudo_train_mask
        y_train = self.sle_data.pesudo_y[train_mask].view(-1)
        input_ids = self.data.input_ids[train_mask]
        attention_mask = self.data.attention_mask[train_mask]
        y_embs = self.sle_data.y_embs[train_mask] if self.sle_data.y_embs is not None else None
        x_embs = self.x_embs[train_mask] if self.x_embs is not None else None

        train_set = SupportNoneTensorDataset(input_ids, attention_mask, x_embs, y_embs, y_train)
        return self._get_dataloader(train_set, self.args.batch_size, shuffle=True)

    def _get_eval_loader(self, mode="test"):
        assert mode in ["train", "valid", "test"]
        eval_mask = self.split_idx[mode]
        y = self.sle_data.pesudo_y[eval_mask].view(-1)
        input_ids = self.data.input_ids[eval_mask]
        attention_mask = self.data.attention_mask[eval_mask]
        y_embs = self.sle_data.y_embs[eval_mask] if self.sle_data.y_embs is not None else None
        x_embs = self.x_embs[eval_mask] if self.x_embs is not None else None

        dataset = SupportNoneTensorDataset(input_ids, attention_mask, x_embs, y_embs, y)
        return self._get_dataloader(dataset, self.args.eval_batch_size, shuffle=False)

    def inference_and_evaluate(self):
        def evalute(logits, y_true):
            y_pred = logits.argmax(dim=-1, keepdim=True)
            acc = y_pred.eq(y_true.view_as(y_pred)).sum().item() / y_true.shape[0]
            return acc

        dataloader = self.inference_dataloader
        hidden_features_list, logits_list = [], []
        pbar = tqdm(dataloader, desc="Inference and evaluating", disable=self.disable_tqdm)
        for step, batch_input in enumerate(dataloader):
            batch_input = self._list_tensor_to_gpu(batch_input)
            logits, hidden_features = self.inference_step(batch_input)
            hidden_features_list.append(hidden_features.cpu())
            logits_list.append(logits.cpu())
        hidden_features = torch.cat(hidden_features_list, dim=0)
        logits = torch.cat(logits_list, dim=0)
        results = {}
        for split in ["train", "valid", "test"]:
            split_idx = self.split_idx[split]
            acc = evalute(logits[split_idx], self.data.y[split_idx])
            loss = self.loss_op(logits[split_idx], self.data.y[split_idx]).item()
            results[f"{split}_acc"] = acc
            results[f"{split}_loss"] = loss
        return hidden_features, logits, results

    @torch.no_grad()
    def inference_step(self, *inputs):
        return self.model(*inputs, return_hidden=True)

    def train(self):
        for iter in range(self.args.num_iterations + 1):
            logger.warning(f"\n*************** Start iter {iter} training ***************\n")
            best_valid_acc = self.train_once(iter)
            # if iter == self.args.num_iterations:
            #     break
            logger.warning(f"\n*************** Start iter {iter} testing and Postprocessing ***************\n")
            # NOTE init model for the next iteration
            test_t_start = time.time()
            ckpt_path = os.path.join(self.args.ckpt_dir, "{}-iter{}-best.pt".format(self.args.model_type, iter))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self._load_state_dict(self.model, ckpt, is_dist=True)
            self.model.to(self.rank)
            logger.info("initialize iter {iter} model with ckpt loaded from: {}".format(iter, ckpt_path))
            # NOTE inference for SLE and propogation
            hidden_features, logits, results = self.inference_and_evaluate()
            self._add_result(f"iter_{iter}_final_test", results)
            self.save_result()
            logger.critical("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))

            # NOTE SLE
            if self.args.SLE:
                val, pred = torch.max(F.softmax(logits, dim=1), dim=1)
                SLE_mask = val > self.args.SLE_threshold
                SLE_pred = pred[SLE_mask]
                self.sle_data.update(SLE_mask, SLE_pred, self.adj_t)
                del SLE_mask, SLE_pred, val, pred

            self.x_embs = self.adj_t @ hidden_features
            del hidden_features, logits
            gc.collect()
            torch.cuda.empty_cache()
            self.train_loader = self._get_train_loader()
