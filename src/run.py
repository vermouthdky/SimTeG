import gc
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from torch_geometric.utils import subgraph

from .args import GNN_LIST
from .dataset import load_data_bundle
from .trainer import get_trainer_class
from .utils import dataset2foldername, dist_barrier_context, is_dist

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    gc.collect()


def load_data(args):
    assert args.dataset in ["ogbn-arxiv", "ogbl-citation2", "ogbn-products"]
    tokenize = args.model_type not in GNN_LIST
    data, split_idx, evaluator = load_data_bundle(
        args.dataset, root=args.data_folder, tokenizer=args.pretrained_repo, tokenize=tokenize
    )
    # process data
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

    if args.debug:
        all_idx = torch.arange(0, 3000)
        data = data.subgraph(all_idx)
        if args.dataset in ["ogbl-citation2"]:
            split_idx["train"] = {"source_node": all_idx[:1000], "target_node": all_idx[1000:2000]}
            split_idx["valid"] = {"source_node": all_idx[1000:2000], "target_node": all_idx[2000:3000]}
            split_idx["train"] = {"source_node": all_idx[2000:3000], "target_node": all_idx[:1000]}
        else:
            split_idx["train"] = all_idx[:1000]
            split_idx["valid"] = all_idx[1000:2000]
            split_idx["test"] = all_idx[2000:3000]

    gc.collect()
    return data, split_idx, evaluator


def train(args, return_value="valid"):
    # setup dataset: [ogbn-arxiv]

    with dist_barrier_context():
        data, split_idx, evaluator = load_data(args)
    # trainer
    Trainer = get_trainer_class(args)
    trainer = Trainer(args, data, split_idx, evaluator)
    acc = trainer.train(return_value=return_value)
    del trainer, data, split_idx, evaluator
    return acc
