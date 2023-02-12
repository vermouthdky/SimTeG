import logging
import os
import time

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.transforms import SIGN
from tqdm import tqdm

from ...trainer import Trainer
from ...utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


class LM_Trainer(Trainer):
    def __init__(self, args, model, data, split_idx, evaluator):
        self.args = args
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.model = self._init_model(model)
        self.train_loader = self._get_train_loader(data, split_idx["train"])
        self.train_eval_loader = self._get_eval_loader(data, "train")
        self.test_eval_loader = self._get_eval_loader(data, "test")
        self.valid_eval_loader = self._get_eval_loader(data, "valid")
        self.optimizer = self._get_optimizer()
        self.loss_op = self._get_loss_op()
        self.data = data

    def _get_train_loader(self, data, train_idx):
        y_train = data.y[train_idx].squeeze(-1)
        input_ids, attention_mask = data.input_ids[train_idx], data.attention_mask[train_idx]
        train_set = TensorDataset(input_ids, attention_mask, y_train)
        train_loader = DataLoader(
            train_set,
            sampler=DistributedSampler(train_set, shuffle=True) if is_dist() else None,
            batch_size=self.args.batch_size,
            shuffle=False if is_dist() else True,
            num_workers=48,
            pin_memory=True,
        )
        return train_loader

    def _get_eval_loader(self, data, mode="test"):
        assert mode in ["train", "valid", "test"]
        eval_mask = data[f"{mode}_mask"]
        dataset = TensorDataset(data.input_ids[eval_mask], data.attention_mask[eval_mask], data.y[eval_mask])
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )

    # def _get_parallel_eval_loader(self, mode="test"):
    #     data = self.data
    #     assert mode in ["train", "valid", "test"]
    #     eval_mask = data[f"{mode}_mask"]
    #     dataset = TensorDataset(
    #         data.input_ids[eval_mask], data.attention_mask[eval_mask], data.y[eval_mask].squeeze(-1)
    #     )
    #     dataloader = DataLoader(
    #         dataset,
    #         sampler=DistributedSampler(dataset, shuffle=False) if is_dist() else None,
    #         batch_size=self.args.eval_batch_size,
    #         shuffle=False,
    #         num_workers=48,
    #         pin_memory=True,
    #     )
    #     return dataloader

    def training_step(self, input_ids, attention_mask, y):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss = self.loss_op(logits, y.to(self.rank))
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
                batch_input = self._list_tensor_to_gpu(batch_input)
                batch_loss = self.training_step(*batch_input)
                loss += batch_loss
                dist.barrier()
            loss /= len(self.train_loader)
            if self.rank == 0 and (epoch + 1) % self.args.eval_interval == 0:
                ckpt_path = os.path.join(
                    self.args.ckpt_dir,
                    "{}-epoch-{}.pt".format(self.args.model_type, epoch + 1),
                )
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info("Saved ckpt: {}".format(ckpt_path))
                # evaluate model on train and validation set
                train_acc = self.evaluate(mode="train")
                valid_acc = self.evaluate(mode="valid")
                logger.info(
                    "epoch: {}, loss: {}, train_acc:{}, valid_acc: {}".format(epoch + 1, loss, train_acc, valid_acc)
                )
                # early stop
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
                        break

        test_t_start = time.time()
        ckpt_path = os.path.join(
            self.args.ckpt_dir,
            "{}-best.pt".format(self.args.model_type),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._load_state_dict(self.model, ckpt)
        self.model.to(self.rank)
        logger.info("Start testing best model loaded from: {}".format(ckpt_path))
        # test_acc = self.evaluate(mode="test")
        test_acc = self.evaluate(mode="test")
        logger.info("test time: {}".format(time.time() - test_t_start))
        logger.info("final test_acc: {}".format(test_acc))
        logger.info("Training finished, time: {}".format(time.time() - t_start))

    def evaluate(self, mode="test"):
        assert mode in ["train", "test", "valid"]
        self.model.eval()
        dim_feat = self.args.num_feats
        eval_loader_list = {
            "train": self.train_eval_loader,
            "valid": self.valid_eval_loader,
            "test": self.test_eval_loader,
        }
        eval_loader = eval_loader_list[mode]
        pbar = tqdm(total=len(eval_loader), desc=f"evaluating {mode} set", disable=self.args.disable_tqdm)
        num_correct, num_total = 0, 0
        for step, (input_ids, att_mask, y_true) in enumerate(eval_loader):
            with torch.no_grad():
                logits = self.model(input_ids.to(self.rank), att_mask.to(self.rank))
            y_pred = logits.argmax(dim=-1, keepdim=True).to("cpu")
            num_correct += (y_pred == y_true).sum()
            num_total += y_true.shape[0]
            pbar.update(1)
        acc = float(num_correct / num_total)
        return acc

    # def parallel_evaluate(self, mode="test"):
    #     assert mode in ["train", "test", "valid"]
    #     self.model.eval()
    #     eval_loader = self._get_parallel_eval_loader(mode)
    #     pbar = tqdm(total=len(eval_loader), desc=f"evaluating {mode} set", disable=self.args.disable_tqdm)
    #     num_correct, num_total = 0, 0
    #     for step, (input_ids, att_mask, y_true) in enumerate(eval_loader):
    #         with torch.no_grad():
    #             logits = self.model(input_ids.to(self.rank), att_mask.to(self.rank))
    #         y_pred = logits.argmax(dim=-1, keepdim=True).to("cpu")
    #         num_correct += (y_pred == y_true).sum()
    #         num_total += y_true.shape[0]
    #         pbar.update(1)
    #     acc = float(num_correct / num_total)
    #     return acc

    def save_bert_x(self, data):
        """
        save bert features to disk, used after training
        """
        dataset = TensorDataset(data.input_ids, data.attention_mask)
        dataloader = DataLoader(
            dataset, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=24, pin_memory=True
        )
        bert_x_list = []
        for i, batch in enumerate(tqdm(dataloader, desc="saving bert featurs", disable=self.args.disable_tqdm)):
            input_ids, att_mask = batch
            with torch.no_grad():
                _, bert_x = self.model(input_ids.to(self.rank), att_mask.to(self.rank), return_bert_out=True)
            bert_x_list.append(bert_x.to("cpu"))
        bert_x = torch.concat(bert_x_list, dim=0)
        saved_dir = os.path.join(self.args.data_folder, dataset2foldername(self.args.dataset), "processed", "bert_x.pt")
        torch.save(bert_x, saved_dir)
        logger.info("save bert features {} to: {}".format(bert_x.shape, saved_dir))
