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
from tqdm import tqdm

from .utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


def _mask_to_idx(mask):
    return torch.range(0, mask.shape[0])[mask]


class Trainer:
    def __init__(self, args, model, data, split_idx, evaluator):
        self.args = args
        self.split_idx = split_idx
        self.evaluator = evaluator
        self.model = self._init_model(model)
        data = self._precompute(data)
        self.train_loader = self._get_train_loader(data, split_idx["train"])
        self.train_eval_loader = self._get_eval_loader(data, "train")
        self.test_eval_loader = self._get_eval_loader(data, "test")
        self.valid_eval_loader = self._get_eval_loader(data, "valid")
        self.optimizer = self._get_optimizer()
        self.loss_op = self._get_loss_op()

    @property
    def rank(self):
        return int(os.environ["RANK"]) if is_dist() else self.args.single_gpu

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"]) if is_dist() else 1

    def _precompute(self, data):
        logger.info("Precomputing data: {}".format(data))
        t_start = time.time()
        data = SIGN(self.args.gnn_num_layers)(data)
        data.xs = [data.x] + [data[f"x{i}"] for i in range(1, self.args.gnn_num_layers + 1)]
        del data.x
        for i in range(self.args.gnn_num_layers):
            del data[f"x{i}"]
        t_end = time.time()
        logger.info("Precomputing finished, time: {}".format(t_end - t_start))
        return data

    def _load_state_dict(self, model, state_dict, strict=True, is_dist=False):
        own_state = model.state_dict()
        if self.rank == 0:
            __import__("ipdb").set_trace()
        else:
            dist.barrier()
        for name, param in state_dict.items():
            if name[:7] == "module." and not is_dist:  # remove the "module." prefix
                name = name[7:]
            if name not in own_state:
                if strict:
                    raise KeyError("unexpected key '{}' in state_dict".format(name))
                else:
                    logger.warning("Ignore unexpected key '{}' in state_dict".format(name))
                    logger.warning(
                        "Please make sure the model related parameters in 'test.sh' is consistent with the 'train.sh'!"
                    )
            else:
                try:
                    own_state[name].copy_(param)
                except:  # the param shape is different between DDP and single gpu
                    own_state[name].copy_(param.squeeze(0))

    def _init_model(self, model):
        if self.args.cont:
            ckpt_name = os.path.join(self.args.ckpt_dir, self.args.ckpt_name)
            ckpt = torch.load(ckpt_name, map_location="cpu")
            self._load_state_dict(model, ckpt, is_dist=False)
            logging.info("load ckpt:{}".format(ckpt_name))
        model.to(self.rank)
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        return model

    def _get_train_loader(self, data, train_idx):
        xs_train = torch.cat([x[train_idx] for x in data.xs], -1)
        y_train = data.y[train_idx].squeeze(-1)
        input_ids, attention_mask = (
            data.input_ids[train_idx],
            data.attention_mask[train_idx],
        )
        train_set = TensorDataset(xs_train, input_ids, attention_mask, y_train)
        train_loader = DataLoader(
            train_set,
            sampler=DistributedSampler(train_set, shuffle=True) if is_dist() else None,
            batch_size=self.args.batch_size,
            shuffle=False if is_dist() else True,
            num_workers=24,
            pin_memory=True,
        )
        return train_loader

    def _get_eval_loader(self, data, mode="test"):
        assert mode in ["train", "valid", "test"]
        eval_mask = data[f"{mode}_mask"]
        xs_eval = torch.cat([x[eval_mask] for x in data.xs], -1)
        dataset = TensorDataset(xs_eval, data.input_ids[eval_mask], data.attention_mask[eval_mask], data.y[eval_mask])
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
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(xs, input_ids, attention_mask)
        loss = self.loss_op(logits, y.to(self.rank))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _list_tensor_to_gpu(self, list: List):
        return [input.to(self.rank) if isinstance(input, torch.Tensor) else input for input in list]

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
            if self.rank == 0 and (epoch + 1) % self.args.eval_interval == 0:
                # ckpt_path = os.path.join(
                #     self.args.ckpt_dir,
                #     "{}-epoch-{}.pt".format(self.args.model_type, epoch + 1),
                # )
                # torch.save(self.model.state_dict(), ckpt_path)
                # logger.info("Saved ckpt: {}".format(ckpt_path))
                # evaluate model on train and validation set
                train_acc = self.evaluate(mode="train")
                logger.info("epoch: {}, loss: {}, train_acc: {}".format(epoch + 1, loss, train_acc))
                valid_acc = self.evaluate(mode="valid")
                logger.info("epoch: {}, loss: {}, valid_acc: {}".format(epoch + 1, loss, valid_acc))
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
        eval_loader_list = {
            "train": self.train_eval_loader,
            "valid": self.valid_eval_loader,
            "test": self.test_eval_loader,
        }
        eval_loader = eval_loader_list[mode]
        pbar = tqdm(total=len(eval_loader), desc=f"evaluating {mode} set", disable=self.args.disable_tqdm)
        num_correct, num_total = 0, 0
        for step, (xs, input_ids, att_mask, y_true) in enumerate(eval_loader):
            xs = [x.to(self.rank) for x in torch.split(xs, dim_feat, -1)]
            with torch.no_grad():
                logits = self.model(xs, input_ids.to(self.rank), att_mask.to(self.rank))
            y_pred = logits.argmax(dim=-1, keepdim=True).to("cpu")
            num_correct += (y_pred == y_true).sum()
            num_total += y_true.shape[0]
            pbar.update(1)
        acc = float(num_correct / num_total)
        return acc

    def parallel_evaluate(self, mode="test"):
        pass

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
                _, bert_x = self.model(None, input_ids.to(self.rank), att_mask.to(self.rank), return_bert_out=True)
            bert_x_list.append(bert_x.to("cpu"))
        bert_x = torch.concat(bert_x_list, dim=0)
        saved_dir = os.path.join(self.args.data_folder, dataset2foldername(self.args.dataset), "processed", "bert_x.pt")
        torch.save(bert_x, saved_dir)
        logger.info("save bert features {} to: {}".format(bert_x.shape, saved_dir))
