import json
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.cuda import is_available

from src.options import parse_args
from src.runner import train
from src.utils import set_logging


def set_env(random_seed, world_size):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "32020"
    if world_size > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        gpus = ",".join([str(_) for _ in range(world_size)])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    args = parse_args()
    set_logging()
    set_env(args.random_seed, args.world_size)
    if args.mode == "train":
        print("--------------------train-------------------")
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value("b", False)
            mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
        else:
            end = None
            train(0, args)


# d8d71a8396ff11200038a65989fa142c56290704

if __name__ == "__main__":
    main()
