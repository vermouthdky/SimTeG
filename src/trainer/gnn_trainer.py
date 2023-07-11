import gc
import json
import logging
import os
import os.path as osp
import shutil

import evaluate
import numpy as np
import optuna
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import SIGN
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_emb, x0, y_emb, labels):
        super().__init__()
        self.data = {
            "x_emb": x_emb,
            "x0": x0,
            "y_emb": y_emb,
            "labels": labels.view(-1, 1),
        }

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class InnerTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        logits = model(**inputs)

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduce="mean")
        loss = loss_op(logits, labels.view(-1))
        return (loss, {"logits": logits}) if return_outputs else loss


class GNNDecouplingTrainer(Trainer):
    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test", "all"]
        use_pesudo = self.args.use_SLE and mode == "train"
        dataset = Dataset(
            x_emb=self.data.x_emb,
            x0=self.data.x,
            y_emb=self.data.sle.y_emb if self.args.use_SLE else None,
            labels=self.data.y if not use_pesudo else self.data.sle.pesudo_y,
        )
        if use_pesudo:
            return torch.utils.data.Subset(dataset, self.data.sle.pesudo_train_idx)
        return dataset if mode == "all" else torch.utils.data.Subset(dataset, self.split_idx[mode])

    def _prepare_dataset(self):
        return self._get_dataset("train"), self._get_dataset("valid"), self._get_dataset("all")

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.gnn_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        eval_steps = train_steps * self.args.gnn_eval_interval
        warmup_steps = self.args.gnn_warmup_ratio * train_steps
        logger.info(f"eval_steps: {eval_steps}, train_steps: {train_steps}, warmup_steps: {warmup_steps}")
        training_args = TrainingArguments(
            seed=self.args.random_seed,
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            learning_rate=self.args.gnn_lr,
            weight_decay=self.args.gnn_weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            gradient_accumulation_steps=1,
            label_smoothing_factor=self.args.gnn_label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.gnn_batch_size,
            per_device_eval_batch_size=self.args.gnn_eval_batch_size,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.args.gnn_lr_scheduler_type,
            disable_tqdm=False,
            num_train_epochs=self.args.gnn_epochs,
            local_rank=self.rank,
            dataloader_num_workers=1,
            ddp_find_unused_parameters=False,
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def inference(self, dataset, embs_path):
        logits_name = f"logits.pt"
        emb_handler = EmbeddingHandler(embs_path)
        if self.args.use_cache and emb_handler.has(logits_name):
            logits_embs = emb_handler.load(logits_name)
            if isinstance(logits_embs, np.ndarray):
                logits_embs = torch.from_numpy(logits_embs)
        else:
            eval_output = self.trainer.predict(dataset)
            logits_embs = eval_output.predictions
            logits_embs = torch.from_numpy(logits_embs)
            emb_handler.save(logits_embs, logits_name)
            logger.info(f"save the logits of {self.args.gnn_type} to {os.path.join(embs_path, logits_name)}")
        return (logits_embs, None)

    # def _edge_index_to_adj_t(self):
    #     self.data = ToSparseTensor()(self.data)
    #     if self.rank == 0:
    #         __import__("ipdb").set_trace()
    #     else:
    #         dist.barrier()
    #     deg = self.data.adj_t.sum(dim=1).to(torch.float)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    #     self.data.adj_t = deg_inv_sqrt.view(-1, 1) * self.data.adj_t * deg_inv_sqrt.view(1, -1)

    def train(self, return_value="valid"):  # used when only train gnn_models
        # preprocessing
        # self._edge_index_to_adj_t()
        # xs = [self.data.x]
        # disable_tqdm = self.args.disable_tqdm or (is_dist() and int(os.environ["RANK"]) > 0)
        # logger.info("propogating features")
        # for i in tqdm(range(1, self.args.gnn_num_layers + 1), disable=disable_tqdm):
        #     xs += [self.data.adj_t @ xs[-1]]
        # xs = xs[1:]  # remove the hop-0 feature, which is saved in self.data.x, this is consistent with gbert
        # self.data.x_emb = torch.cat(xs, dim=-1)

        k = self.args.gnn_num_layers
        self.data = SIGN(k)(self.data)
        x_emb = []
        for i in range(1, k + 1):
            x_emb.append(self.data[f"x{i}"])
            del self.data[f"x{i}"]
        self.data.x_emb = torch.cat(x_emb, dim=-1)
        return super().train(return_value=return_value)


class GNNSamplingTrainer:  # single gpu
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.trial = kwargs.pop("trial", None)

    @property
    def device(self):
        return torch.device(self.args.single_gpu if torch.cuda.is_available() else "cpu")

    def _prepare_model(self):
        assert self.args.model_type in ["GraphSAGE", "GCN"]
        model_class = get_model_class(self.args.model_type, self.args.task_type)
        return model_class(self.args)

    def _prepare_dataloader(self):
        # return train_loader and subgraph_loader
        num_neighbors = [15, 10, 5, 5]
        assert self.args.gnn_num_layers <= 4
        train_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx["train"],
            num_neighbors=num_neighbors[: self.args.gnn_num_layers],
            batch_size=self.args.gnn_batch_size,
            shuffle=True,
            num_workers=12,
            persistent_workers=True,
        )
        subgraph_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=self.args.gnn_eval_batch_size,
            num_workers=12,
            persistent_workers=True,
        )
        return train_loader, subgraph_loader

    def _prepare_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

    def training_step(self, epoch):
        self.model.train()

        total_loss = total_correct = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            out = self.model(batch.x.to(self.device), batch.edge_index.to(self.device))[: batch.batch_size]
            y = batch.y[: batch.batch_size].squeeze().to(self.device)
            loss = F.cross_entropy(out, y, label_smoothing=self.args.gnn_label_smoothing)
            loss.backward()
            self.optimizer.step()

            loss_without_smoothing = F.cross_entropy(out, y)

            total_loss += float(loss_without_smoothing)
            total_correct += int(out.argmax(dim=-1).eq(y).sum())

        loss = total_loss / len(self.train_loader)
        approx_acc = total_correct / self.split_idx["train"].size(0)

        return loss, approx_acc

    @torch.no_grad()
    def eval(self, save_out=False):
        self.model.eval()
        out = self.model.inference(self.data.x, self.device, self.subgraph_loader)
        y_true = self.data.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)
        if save_out:
            embs_dir = osp.join(self.args.output_dir, "cached_embs")
            os.makedirs(embs_dir, exist_ok=True)
            torch.save(out, osp.join(embs_dir, f"logits_seed{self.args.random_seed}.pt"))
            logger.warning(f"saved logits to {embs_dir}/logits{self.args.random_seed}.pt")

        train_acc = self.evaluator.eval(
            {
                "y_true": y_true[self.split_idx["train"]],
                "y_pred": y_pred[self.split_idx["train"]],
            }
        )["acc"]
        val_acc = self.evaluator.eval(
            {
                "y_true": y_true[self.split_idx["valid"]],
                "y_pred": y_pred[self.split_idx["valid"]],
            }
        )["acc"]
        test_acc = self.evaluator.eval(
            {
                "y_true": y_true[self.split_idx["test"]],
                "y_pred": y_pred[self.split_idx["test"]],
            }
        )["acc"]

        return train_acc, val_acc, test_acc

    def train(self, return_value="valid"):
        self.model = self._prepare_model()
        self.model.to(self.device)
        self.model.reset_parameters()
        self.train_loader, self.subgraph_loader = self._prepare_dataloader()
        self.optimizer = self._prepare_optimizer()
        best_val_acc = final_test_acc = 0
        accumulate_patience = 0
        for epoch in range(1, self.args.gnn_epochs + 1):
            loss, acc = self.training_step(epoch)
            logger.info(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")
            if epoch >= self.args.gnn_eval_warmup and epoch % self.args.gnn_eval_interval == 0:
                train_acc, val_acc, test_acc = self.eval()
                logger.info(f"Epoch: {epoch:02d} Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
                if val_acc > best_val_acc:
                    accumulate_patience = 0
                    best_val_acc = val_acc
                    final_test_acc = test_acc
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.args.ckpt_dir, f"model_seed{self.args.random_seed}.pt"),
                    )
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 10:
                        break
                if self.trial is not None:
                    self.trial.report(val_acc, epoch)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
        self.model.load_state_dict(
            torch.load(os.path.join(self.args.ckpt_dir, f"model_seed{self.args.random_seed}.pt"))
        )
        train_acc, val_acc, test_acc = self.eval(save_out=True)
        logger.info(f"best_train_acc: {train_acc:.4f}, best_valid_acc: {val_acc:.4f}, best_test_acc: {test_acc:.4f}")
        return test_acc, val_acc


class MLPTrainer:
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.trial = kwargs.pop("trial", None)

    @property
    def device(self):
        return torch.device(self.args.single_gpu if torch.cuda.is_available() else "cpu")

    def _prepare_model(self):
        assert self.args.model_type in ["MLP"]
        model_class = get_model_class(self.args.model_type, self.args.task_type)
        return model_class(self.args)

    def _prepare_dataloader(self):
        train_set = TensorDataset(self.data.x[self.split_idx["train"]], self.data.y[self.split_idx["train"]])
        train_loader = DataLoader(train_set, batch_size=self.args.gnn_batch_size, shuffle=True)
        all_loader = DataLoader(self.data.x, batch_size=self.args.gnn_eval_batch_size, shuffle=False)
        return train_loader, all_loader

    def _prepare_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

    def training_step(self, epoch):
        self.model.train()
        total_loss = total_correct = 0
        for x, y in self.train_loader:
            self.optimizer.zero_grad()
            out = self.model(x.to(self.device))
            y = y.squeeze().to(self.device)
            loss = F.cross_entropy(out, y, label_smoothing=self.args.gnn_label_smoothing)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y).sum())

        loss = total_loss / len(self.train_loader)
        approx_acc = total_correct / self.split_idx["train"].size(0)

        return loss, approx_acc

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        out = self.model.inference(self.device, self.all_loader)
        y_true = self.data.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = self.evaluator.eval(
            {
                "y_true": y_true[self.split_idx["train"]],
                "y_pred": y_pred[self.split_idx["train"]],
            }
        )["acc"]
        val_acc = self.evaluator.eval(
            {
                "y_true": y_true[self.split_idx["valid"]],
                "y_pred": y_pred[self.split_idx["valid"]],
            }
        )["acc"]
        test_acc = self.evaluator.eval(
            {
                "y_true": y_true[self.split_idx["test"]],
                "y_pred": y_pred[self.split_idx["test"]],
            }
        )["acc"]

        return train_acc, val_acc, test_acc

    def train(self, return_value="valid"):
        self.model = self._prepare_model()
        self.model.to(self.device)
        self.train_loader, self.all_loader = self._prepare_dataloader()
        self.optimizer = self._prepare_optimizer()
        best_val_acc = final_test_acc = 0
        accumulate_patience = 0
        for epoch in range(1, self.args.gnn_epochs + 1):
            loss, acc = self.training_step(epoch)
            logger.info(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")
            if epoch > 0:
                train_acc, val_acc, test_acc = self.eval()
                logger.info(f"Epoch: {epoch:02d} Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
                if val_acc > best_val_acc:
                    accumulate_patience = 0
                    best_val_acc = val_acc
                    final_test_acc = test_acc
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 5:
                        break
                if self.trial is not None:
                    self.trial.report(val_acc, epoch)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
        logger.info(f"best_val_acc: {best_val_acc:.4f}, final_test_acc: {final_test_acc:.4f}")
        torch.save(self.model.state_dict(), os.path.join(self.args.ckpt_dir, "model.pt"))
        return final_test_acc, best_val_acc
