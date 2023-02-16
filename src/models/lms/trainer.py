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
    def __init__(self, args, model, data, split_idx, evaluator, **kwargs):
        super(LM_Trainer, self).__init__(args, model, data, split_idx, evaluator, **kwargs)

    def _get_train_loader(self):
        data = self.data
        train_idx = self.split_idx["train"]
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

    def _get_eval_loader(self, mode="test"):
        data = self.data
        assert mode in ["train", "valid", "test"]
        eval_mask = data[f"{mode}_mask"]
        dataset = TensorDataset(data.input_ids[eval_mask], data.attention_mask[eval_mask], data.y[eval_mask])
        return DataLoader(
            dataset,
            sampler=DistributedSampler(dataset, shuffle=False) if is_dist() else None,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
        )

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
