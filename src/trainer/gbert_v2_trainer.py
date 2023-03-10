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
from torch_geometric.transforms import SIGN
from tqdm import tqdm

from ..model import get_model_class
from ..utils import dataset2foldername, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class SLE:
    def __init__(self, train_idx, y_true, num_labels, SLE_threshold, use_SLE=True):
        """
        ground_truth: train_idx, y_true
        pesudos: pesudo_train_mask, pesudo_y
        for propogation: y_embs
        """
        self.enabled = use_SLE
        self.threshold = SLE_threshold
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

    def update(self, logits, adj_t):
        if not self.enabled:
            return
        # self training
        val, pred = torch.max(F.softmax(logits, dim=1), dim=1)
        SLE_mask = val > self.threshold
        SLE_pred = pred[SLE_mask]
        self.pesudo_train_mask = SLE_mask | self.pesudo_train_mask
        self.pesudo_y[SLE_mask] = SLE_pred
        self.pesudo_y[self.train_mask] = self.y_true[self.train_mask]
        # label propogation
        self.y_embs = adj_t @ self.y_embs


class GBert_v2_Trainer(Trainer):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.iter = 0
        super(GBert_v2_Trainer, self).__init__(args, data, split_idx, evaluator, **kwargs)

    def _get_inference_dataloader(self):
        input_ids = self.data.input_ids
        attention_mask = self.data.attention_mask
        x_emb = self.data.x_emb if self.iter > 0 else None
        idx = torch.arange(input_ids.size(0), dtype=torch.long)
        tensors = [input_ids, attention_mask, x_emb, idx]
        tensors = [t for t in tensors if t is not None]
        dataset = TensorDataset(*tensors)
        return self._get_dataloader(dataset, self.args.eval_batch_size, shuffle=False)

    def _get_dataset(self, mask: torch.Tensor):
        y = self.data.y[mask].view(-1)
        input_ids = self.data.input_ids[mask]
        attention_mask = self.data.attention_mask[mask]
        x_emb = self.data.x_emb[mask] if self.iter > 0 else None
        tensors = [input_ids, attention_mask, x_emb, y]
        tensors = [t for t in tensors if t is not None]
        return TensorDataset(*tensors)

    # def _get_optimizer(self):
    #     if self.iter == 0:
    #         super()._get_optimizer()
    #     else:
    #         bert_params = list(self.model.bert_model.parameters())
    #         gnn_params = list(self.model.gnn_model.parameters())
    #         return torch.optim.AdamW(
    #             [
    #                 {"params": bert_params, "lr": self.args.lr, "weight_decay": self.args.weight_decay},
    #                 {"params": gnn_params, "lr": self.args.gnn_lr, "weight_decay": self.args.weight_decay},
    #             ],
    #         )

    def _get_train_loader(self):
        train_set = self._get_dataset(self.split_idx["train"])
        return self._get_dataloader(train_set, self.args.batch_size, shuffle=True)

    def _get_eval_loader(self, mode="test"):
        assert mode in ["train", "valid", "test"]
        eval_mask = self.split_idx[mode]
        dataset = self._get_dataset(eval_mask)
        return self._get_dataloader(dataset, self.args.eval_batch_size, shuffle=False)

    def inference_and_evaluate(self, iter):
        def evalute(logits, y_true):
            y_pred = logits.argmax(dim=-1, keepdim=True)
            acc = y_pred.eq(y_true.view_as(y_pred)).sum() / y_true.shape[0]
            return acc.item()

        def dist_gather(tensor):
            tensor = tensor.contiguous()
            tensor_list = [tensor.clone() for _ in range(self.world_size)]
            # dist.gather(tensor, tensor_list, dst=0)
            dist.all_gather(tensor_list, tensor)
            return tensor_list

        def save_embs(tensor, saved_name):
            if self.rank == 0:
                output_path = os.path.join(self.args.output_dir, "cached_embs")
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                torch.save(tensor, os.path.join(output_path, saved_name))
                logger.info(f"Saved {saved_name} to {output_path}")
            dist.barrier()

        def has_embs(saved_name):
            output_path = os.path.join(self.args.output_dir, "cached_embs")
            return os.path.exists(os.path.join(output_path, saved_name))

        def load_embs(saved_name):
            output_path = os.path.join(self.args.output_dir, "cached_embs")
            tensor = torch.load(os.path.join(output_path, saved_name))
            return tensor

        if self.args.use_cache and has_embs(f"iter_{iter}_x_embs.pt") and has_embs(f"iter_{iter}_logits.pt"):
            logger.info("Loading cached embs...")
            all_logits = load_embs(f"iter_{iter}_logits.pt")
            all_x_embs = load_embs(f"iter_{iter}_x_embs.pt")
        else:
            self.model.eval()
            dataloader = self.inference_dataloader
            all_x_embs = torch.zeros(self.data.num_nodes, self.args.hidden_size)
            all_logits = torch.zeros(self.data.num_nodes, self.args.num_labels)
            pbar = tqdm(dataloader, desc="Inference and evaluating", disable=self.disable_tqdm)
            for step, batch_input in enumerate(dataloader):
                batch_input = self._list_tensor_to_gpu(batch_input)
                idx, input = batch_input[-1], batch_input[:-1]
                logits, hidden_features = self.inference_step(*input)

                # gather all hidden features and logits
                hidden_features_list = dist_gather(hidden_features)
                logits_list = dist_gather(logits)
                idx_list = dist_gather(idx)
                for hf, lg, idx in zip(hidden_features_list, logits_list, idx_list):
                    all_x_embs[idx.cpu()] = hf.cpu()
                    all_logits[idx.cpu()] = lg.cpu()
                dist.barrier()
                gc.collect()
                pbar.update(1)

            save_embs(all_x_embs, f"iter_{iter}_x_embs.pt")
            save_embs(all_logits, f"iter_{iter}_logits.pt")

        results = {}
        for split in ["train", "valid", "test"]:
            split_idx = self.split_idx[split]
            acc = evalute(all_logits[split_idx], self.data.y[split_idx])
            loss = self.loss_op(all_logits[split_idx], self.data.y[split_idx].view(-1)).item()
            results[f"{split}_acc"] = acc
            results[f"{split}_loss"] = loss
        return all_x_embs, all_logits, results

    @torch.no_grad()
    def inference_step(self, *inputs):
        return self.model(*inputs, return_hidden=True)

    def train(self):
        for iter in range(self.args.num_iterations):
            logger.warning(f"\n*************** Start iter {iter} training ***************\n")
            # load the pretrained bert model
            model = get_model_class("GBert")(self.args, iter)
            self.model, self.metric = self._init_model_and_metric(model)
            self.train_loader = self._get_train_loader()
            self.inference_dataloader = self._get_inference_dataloader()
            self.eval_loader = {
                "train": self._get_eval_loader("train"),
                "valid": self._get_eval_loader("valid"),
                "test": self._get_eval_loader("test"),
            }
            gc.collect()
            torch.cuda.empty_cache()
            if self.iter == 0:
                ckpt_path = os.path.join(self.args.ckpt_dir, "{}_iter_{}_best.pt".format(self.args.model_type, 0))
                if self.args.use_cache and os.path.exists(ckpt_path):
                    logger.warning(f"\n*********iter {iter} has been trained, use cached ckpt instead!*********\n")
                else:
                    best_valid_acc = self.train_once(iter)
                    logger.info("Iter {} Best valid acc: {:.4f}".format(iter, best_valid_acc))
            else:
                ckpt_path = os.path.join(
                    self.args.ckpt_dir, "{}_iter_{}_best.pt".format(self.args.model_type, self.iter - 1)
                )
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self._load_state_dict(self.model, ckpt, strict=False, is_dist=True)
                self.model.to(self.rank)
                self.train_once(iter)

            logger.warning(f"\n*************** Start iter {iter} testing and Postprocessing ***************\n")
            # NOTE init model for the next iteration
            test_t_start = time.time()
            ckpt_path = os.path.join(self.args.ckpt_dir, "{}_iter_{}_best.pt".format(self.args.model_type, self.iter))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self._load_state_dict(self.model, ckpt, is_dist=True)
            self.model.to(self.rank)
            logger.info("initialize iter {} model with ckpt loaded from: {}".format(iter, ckpt_path))
            # NOTE inference for SLE and propogation
            hidden_features, logits, results = self.inference_and_evaluate(iter)
            logger.critical("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))
            self._add_result(f"iter_{iter}_final_test", results)
            self.save_result(self.args.output_dir)

            self.iter += 1
            self.data.x = hidden_features
            num_layers = self.args.gnn_num_layers
            self.data = SIGN(num_layers)(self.data)
            self.data.x_emb = torch.cat([self.data[f"x{i}"] for i in range(1, num_layers + 1)], dim=-1)
            del self.data.x
            for i in range(1, num_layers + 1):
                del self.data[f"x{i}"]
            del hidden_features, logits
            gc.collect()
            torch.cuda.empty_cache()
