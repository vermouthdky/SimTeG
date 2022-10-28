import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from huggingface_hub import hf_hub_download
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import ClusterData, ClusterLoader

from src.datasets import load_dataset
from src.models.TGRoberta import TGRobertaConfig, TGRobertaForMaksedLM, TGRobertaForNodeClassification


def set_single_env(rank, world_size):
    # initialize the process group
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_model(args):
    if args.model_type == "TGRoberta":
        # config_name = args.config_name if args.config_name else args.model_name_or_path
        # config = TGRobertaConfig.from_pretrained(config_name)
        config = TGRobertaConfig()
        model = TGRobertaForNodeClassification(config)
        model_cached_dir = os.path.join(args.pretrained_dir, args.pretrained_model)
        if not os.path.exists(model_cached_dir):
            hf_hub_download(repo_id=f"{args.pretrained_model}", filename="pytorch_model.bin", cache_dir=f"{model_cached_dir}")
        model.load_state_dict(torch.load(model_cached_dir, map_location="cpu")["model_state_dict"], strict=False)
    else:
        raise NotImplementedError
    return model


def train(rank, args):
    set_single_env(rank, args.world_size)
    model = load_model(args)
    logging.info("load model: {}".format(args.model_type))
    model = model.cuda()

    if args.word_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    if args.cont:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        model.load_state_dict(torch.load(args.ckpt_name, map_location=map_location))
        logging.info("load ckpt:{}".format(args.ckpt_name))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = load_dataset(args.dataset, root=args.data_folder, tokenizer="roberta-base")
    data = dataset[0]
    # TODO finish the pretraining and hopefully run the whole pipline successfully
    cluster_data = ClusterData(data, num_parts=args.num_parts, recursive=False, save_dir=dataset.processd_dir)
    dist_sampler = DistributedSampler(dataset=dataset, shuffle=True)
    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=False, sampler=dist_sampler, num_workers=12)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        __import__("ipdb").set_trace()
        for step, batch in enumerate(train_loader):
            pass


# def pretrain(rank, args):
#     set_single_env(rank, args.word_size)
#     model = load_model(args.model_type)
#     logging.info("load model {} for pretraining".format(args.model_type))
#     model = mode.cuda()

#     if args.word_size > 1:
#         model = DDP(model, device_ids=[rank], output_device=rank)

#     if args.cont:
#         map_locatoi
