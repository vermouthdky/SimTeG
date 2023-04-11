import gc
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

from ..utils import dataset2foldername, is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class LM_Trainer(Trainer):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        super(LM_Trainer, self).__init__(args, data, split_idx, evaluator, **kwargs)

    def _get_train_loader(self):
        data = self.data
        train_idx = self.split_idx["train"]
        y_train = data.y[train_idx].squeeze(-1)
        input_ids, attention_mask = data.input_ids[train_idx], data.attention_mask[train_idx]
        train_set = TensorDataset(input_ids, attention_mask, y_train)
        return self._get_dataloader(train_set, self.args.batch_size, shuffle=True)

    def _get_eval_loader(self, mode="test"):
        data = self.data
        assert mode in ["train", "valid", "test"]
        eval_mask = self.split_idx[mode]
        dataset = TensorDataset(data.input_ids[eval_mask], data.attention_mask[eval_mask], data.y[eval_mask])
        return self._get_dataloader(dataset, self.args.eval_batch_size, shuffle=False)

    def train_once(self, iter=0):
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_scheduler()
        best_acc, best_count = 0.0, 0
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            loss = 0.0
            t_start_epoch = time.time()
            for step, batch_input in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=self.disable_tqdm)
            ):
                batch_input = self._list_tensor_to_gpu(batch_input)
                kwargs = {"step": step, "accum_interval": self.args.accum_interval, "len_batch": len(self.train_loader)}
                batch_loss = self.training_step(*batch_input, **kwargs)
                loss += batch_loss
                dist.barrier()
                t_end_epoch = time.time()
                train_time_per_epoch = t_end_epoch - t_start_epoch
                loss /= len(self.train_loader)
            # evalutation and early stop
            if (epoch + 1) % self.args.eval_interval == 0:
                if self.args.save_ckpt_per_valid:
                    ckpt_name = "{}_iter_{}_epoch_{}.pt".format(self.args.model_type, iter, epoch + 1)
                    self.save_model(ckpt_name)
                if self.args.eval_train_set:
                    train_acc, train_loss, train_time = self.evaluate(mode="train")
                else:
                    train_acc, train_loss, train_time = -1, -1, -1
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
                logger.warning("".join("{}:{:.4f} ".format(k, v) for k, v in result.items()))
                self._add_result(f"iter_{iter}_epoch_{epoch+1}", result)
                if self.args.optuna and self.trial is not None:
                    self.trial.report(valid_acc, epoch + 1)
                    # if not iterative pruning, prune it if necessary
                    # else we do pruning in the outer loop
                    if iter == -1 and self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                # record loss and accs
                # explictly early stop
                if valid_acc > best_acc:
                    ckpt_name = "{}_iter_{}_best.pt".format(self.args.model_type, iter)
                    self.save_model(ckpt_name)
                    best_acc = valid_acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        return best_acc
        return best_acc  # best valid acc

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
                _, bert_x = self.model(input_ids.to(self.rank), att_mask.to(self.rank), return_hidden=True)
            bert_x_list.append(bert_x.to("cpu"))
        bert_x = torch.concat(bert_x_list, dim=0)
        saved_dir = os.path.join(self.args.data_folder, dataset2foldername(self.args.dataset), "processed", "bert_x.pt")
        torch.save(bert_x, saved_dir)
        del dataloader, bert_x, bert_x_list
        gc.collect()
        logger.info("save bert features {} to: {}".format(bert_x.shape, saved_dir))

