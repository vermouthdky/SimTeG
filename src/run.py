import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def set_single_env(rank, world_size):
    # initialize the process group
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_model(model_type):
    pass


def train(rank, args):
    set_single_env(rank, args.world_size)
    model = load_model(args.model_type)
    logging.info("load model: {}".format(args.model_type))
    model = model.cuda()

    if args.word_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    if args.cont:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        model.load_state_dict(torch.load(args.ckpt_name, map_location=map_location))
        logging.info("load ckpt:{}".format(args.ckpt_name))

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # data_collator = Dat


def test(args):
    pass


def train_and_test(args):
    pass
