import gc
import json
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.args import load_args, parse_args, save_args
from src.run import train
from src.utils import is_dist, set_logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=ExperimentalWarning)

# to this destination


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    gc.collect()


def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_dist():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    if is_dist():
        gpus = ",".join([str(_) for _ in range(int(os.environ["WORLD_SIZE"]))])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):
    set_logging()
    test_acc_list = []
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    set_single_env(rank, world_size)
    for random_seed in range(args.n_exps):
        random_seed += args.start_seed
        set_seed(random_seed)
        logger.critical(f"{random_seed}-th run with seed {random_seed}")
        args.random_seed = random_seed
        logger.info(args)
        test_acc = train(args, return_value="test")
        test_acc_list.append(test_acc)
    cleanup()
    logger.critical(f"final test_acc {np.mean(test_acc_list)} ± {np.std(test_acc_list)}")


if __name__ == "__main__":
    args = parse_args()
    save_args(args, args.output_dir)
    main(args)
