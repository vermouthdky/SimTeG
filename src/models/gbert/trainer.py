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


class GBert_Trainer(Trainer):
    def __init__(self, args, model, data, split_idx, evaluator, **kwargs):
        self._init_pseudos(split_idx["train"], data.y, args.num_labels)
        super(GBert_Trainer, self).__init__(args, model, data, split_idx, evaluator, **kwargs)
        self.inference_dataloader = self._get_inference_dataloader()
        self.adj_t = self._get_adj_t(data.adj_t)

    def _get_adj_t(self, adj_t):
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def _propogate(self, x):
        return self.adj_t @ x

    def _init_pseudos(self, train_idx, y_true, num_labels):
        self.pseudo_train_mask = torch.zero_like(y_true, dtype=torch.bool)
        self.pseudo_train_mask[train_idx] = True
        # for label propagation
        self.y_embs = torch.zeros(y_true.size(0), num_labels)
        self.y_embs[self.pseudo_train_idx] = F.one_hot(
            y_true[self.pseudo_train_idx],
            num_classes=num_labels,
        ).float()
        # for self training
        self.pseudo_y = torch.zeros_like(y_true)
        self.pseudo_y[self.peusdo_train_idx] = self.data.y[self.peusdo_train_idx]
        # for propogation
        self.propogated_x = None
        pesudo_label_rate = self.pseudo_train_idx.sum().item() / self.pseudo_train_idx.size(0)
        logger.info("Initialized pseudo labels, rate: {:.4f}".format(}")

    def _get_inference_dataloader(self):
        inference_set = TensorDataset(self.propogated_x, self.data.input_ids, self.data.attention_mask)
        # return self._get_dataloader(inference_set, self.args.eval_batch_size, shuffle=False)
        return DataLoader(inference_set, batch_size=self.args.eval_batch_size, shuffle=False)

    def _get_train_loader(self):
        # initialize hidden_features for propogation
        y_train = self.pseudo_y[self.pseudo_train_idx].squeeze(-1)
        input_ids = self.data.input_ids[self.pseudo_train_idx]
        attention_mask = self.data.attention_mask[self.pseudo_train_idx]
        propogated_x = None
        if self.propogated_x is not None:
            propogated_x = self.propogated_x[self.pseudo_train_idx]

        train_set = TensorDataset(propogated_x, input_ids, attention_mask, y_train)
        return self._get_dataloader(train_set, self.args.batch_size, shuffle=True)

    def _get_eval_loader(self, mode="test"):
        assert mode in ["train", "valid", "test"]
        input_ids, attention_mask, y = self.data.input_ids, self.data.attention_mask, self.data.y
        eval_mask = self.split_idx[mode]
        dataset = TensorDataset(input_ids[eval_mask], attention_mask[eval_mask], y[eval_mask])
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
        return hidden_features, results

    @torch.no_grad()
    def inference_step(self, *inputs):
        return self.model(*inputs)

    def train(self):
        for iter in range(self.args.num_iterations):
            logger.info(f"\n*************** Start iter {iter} training ***************\n")
            self._train_one_iteration(iter)
            if iter == self.args.num_iterations - 1:
                break

            logger.info(f"\n*************** Start iter {iter} Postprocessing ***************\n")
            # NOTE init model for the next iteration
            test_t_start = time.time()
            ckpt_path = os.path.join(self.args.ckpt_dir, "{}-iter{}-best.pt".format(self.args.model_type, iter))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self._load_state_dict(self.model, ckpt, is_dist=True)
            self.model.to(self.rank)
            logger.info("initialize iter {iter} model with ckpt loaded from: {}".format(iter, ckpt_path))
            # NOTE inference to compute self.propogated_x
            hidden_features, results = self.inference_and_evaluate()
            self._add_result(f"iter_{iter}_final_test", results)
            logger.info("".join("{}:{:.4f} ".format(k, v) for k, v in results.items()))
            # NOTE update self.propogated_x and self.y_embs, further update self.train_loader
            self.propogated_x = self._propogate(hidden_features)
            self.y_embs = self._propogate(self.y_embs)
            return

    def _train_one_iteration(self, iter: int):
        t_start = time.time()
        best_acc, best_count = 0.0, 0
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            t_start_time = time.time()
            for step, batch_input in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=self.disable_tqdm)
            ):
                batch_input = self._list_tensor_to_gpu(batch_input)
                batch_loss = self.training_step(*batch_input)
                loss += batch_loss
                dist.barrier()
            dist.barrier()
            t_end_epoch = time.time()
            train_time_per_epoch = t_end_epoch - t_start_time
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
                logger.info("".join("{}:{:.4f} ".format(k, v) for k, v in result.items()))
                self._add_result(f"iter_{iter}_epoch_{epoch+1}", result)
                # early stop
                if valid_acc > best_acc:
                    ckpt_name = "{}-iter{}-best.pt".format(self.args.model_type, iter)
                    self.save_model(ckpt_name)
                    best_acc = valid_acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        break
