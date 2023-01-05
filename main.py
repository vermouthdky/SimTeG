import json
import logging
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from src.options import parse_args
from run import train
from src.utils import set_logging


def set_env(random_seed, world_size):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    set_env(args.random_seed, int(os.environ["WORLD_SIZE"]))
    train(args)


if __name__ == "__main__":
    main()
