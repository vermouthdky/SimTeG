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
from src.models.modeling_gbert import GBert, Roberta
from src.models.modeling_gnn import SAGN
from src.trainer import Trainer
from src.utils import is_dist

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_model(args):
    model_class = {"GBert": GBert, "SAGN": SAGN, "Roberta": Roberta}
    assert args.model_type in model_class.keys()
    return model_class[args.model_type](args)


def load_data(args):
    dataset = load_dataset(
        args.dataset,
        root=args.data_folder,
        transform=ToUndirected(),
        tokenizer=args.pretrained_model,
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
    model = load_model(args)
    if rank == 0:
        torch.distributed.barrier()
    # trainer
    trainer = Trainer(args, model, data, split_idx, evaluator)
    trainer.train()
    cleanup()


def test(args):
    if is_dist():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        set_single_env(rank, world_size)
        if rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    model = GBert(args)
    if is_dist() and rank == 0:
        torch.distributed.barrier()
    # trainer
    trainer = Trainer(args, model, data, split_idx, evaluator)
    # train_acc = trainer.evaluate(mode="train")
    train_acc = 0.0  # BUG: for debug
    valid_acc = trainer.evaluate(mode="valid")
    logger.info("valid_acc: {:.4f}".format(valid_acc))
    test_acc = trainer.evaluate(mode="test")
    logger.info(f"train acc: {train_acc:.4f}, valid: {valid_acc:.4f}, test: {test_acc:.4f}")
    # cleanup()
