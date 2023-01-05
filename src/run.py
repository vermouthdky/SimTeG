import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from ogb.nodeproppred import Evaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import NeighborSampler
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

from src.datasets import load_dataset
from src.models.modeling_gbert import GBert
from src.trainer import Trainer

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_data(args):
    dataset = load_dataset(
        args.dataset, root=args.data_folder, transform=ToUndirected(), tokenizer=args.pretrained_model
    )
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    evaluator = Evaluator(name=args.dataset)
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f"{split}_mask"] = mask

    return data, split_idx, evaluator, dataset.processed_dir


def train(args):
    # setup running envs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    set_single_env(rank, world_size)
    if rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    model = GBert(args)
    if rank == 0:
        torch.distributed.barrier()
    # trainer
    trainer = Trainer(args, model, data, split_idx, evaluator)
    trainer.train()
    cleanup()


# TODO: add test function
def test(args):
    model = load_model(args)
    data, split_idx, evaluator, processed_dir = load_data(args)
    model.eval()
    subgraph_loader = NeighborSampler(
        data.edge_index, sizes=[15, 10], node_idx=None, batch_size=5, shuffle=False, num_workers=24
    )
    module = model if int(os.environ["RANK"]) == -1 else model.module
    logits = module.inference(data, subgraph_loader, rank)
    y_true = data.y.unsqueeze(-1)
    y_pred = logits.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({"y_true": y_true[data.train_mask], "y_pred": y_pred[data.train_mask]})["acc"]
    valid_acc = evaluator.eval({"y_true": y_true[data.valid_mask], "y_pred": y_pred[data.valid_mask]})["acc"]
    test_acc = evaluator.eval({"y_true": y_true[data.test_mask], "y_pred": y_pred[data.test_mask]})["acc"]
    return train_acc, valid_acc, test_acc
