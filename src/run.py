import gc
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import subgraph

from .dataset import load_dataset
from .trainer import get_trainer_class
from .utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    gc.collect()


def load_data(args):
    tokenize = args.model_type not in ["GAMLP", "SAGN", "SIGN"]
    rank = os.getenv("RANK", -1)
    dataset = load_dataset(
        args.dataset,
        root=args.data_folder,
        tokenizer=args.pretrained_repo,
        tokenize=tokenize,
    )
    split_idx = dataset.get_idx_split()
    data = dataset.data
    # explictly convert to sparse tensor
    if args.dataset == "ogbn-arxiv":
        transform = T.ToUndirected()
        data = transform(data)
    # if use bert_x, change it
    if args.use_bert_x:
        # saved_dir = os.path.join(args.data_folder, dataset2foldername(args.dataset), "processed", "bert_x.pt")
        bert_x = torch.load(args.bert_x_dir)
        assert bert_x.size(0) == data.x.size(0)
        logger.warning(f"using bert x loaded from {args.bert_x_dir}")
        data.x = bert_x
    evaluator = Evaluator(name=args.dataset)
    gc.collect()

    if args.debug:
        all_idx = torch.arange(0, 3000)
        data = data.subgraph(all_idx)
        split_idx["train"] = all_idx[:1000]
        split_idx["valid"] = all_idx[1000:2000]
        split_idx["test"] = all_idx[2000:3000]

    return data, split_idx, evaluator, dataset.processed_dir


def train(args, return_value="valid"):
    # setup running envs
    rank = os.getenv("RANK", -1)
    if rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    if rank == 0:
        torch.distributed.barrier()
    # trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(args, data, split_idx, evaluator)
    acc = trainer.train(return_value=return_value)
    del trainer, data, split_idx, evaluator
    return acc
