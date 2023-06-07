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

from src.args import LINK_PRED_DATASETS, load_args, parse_args, save_args
from src.run import train
from src.utils import dict_append, is_dist, set_logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=ExperimentalWarning)

# to this destination


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    if is_dist():
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


def print_metrics(metrics: dict, type):
    results = {key: (np.mean(value), np.std(value)) for key, value in metrics.items()}
    logger.critical(
        f"{type} Metrics:\n" + "\n".join("{}: {} ± {} ".format(k, _mean, _std) for k, (_mean, _std) in results.items())
    )


def main(args):
    set_logging()
    if is_dist():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        set_single_env(rank, world_size)

    save_args(args, args.output_dir)

    if args.dataset in LINK_PRED_DATASETS:
        val_metrics_list = {"mrr": [], "hits@1": [], "hits@5": [], "hits@10": []}
        test_metrics_list = val_metrics_list.copy()
        for i, random_seed in enumerate(range(args.n_exps)):
            random_seed += args.start_seed
            set_seed(random_seed)
            logger.critical(f"{i}-th run with seed {random_seed}")
            args.random_seed = random_seed
            logger.info(args)

            test_metrics, val_metrics = train(args, return_value="test")

            val_metrics_list = dict_append(val_metrics_list, val_metrics)
            test_metrics_list = dict_append(test_metrics_list, test_metrics)

            print_metrics(val_metrics_list, "Current Val")
            print_metrics(test_metrics_list, "Current Test")
        cleanup()
        print_metrics(val_metrics_list, "Final Val")
        print_metrics(val_metrics_list, "Final Test")

    else:
        test_acc_list = []
        val_acc_list = []
        for i, random_seed in enumerate(range(args.n_exps)):
            random_seed += args.start_seed
            set_seed(random_seed)
            logger.critical(f"{i}-th run with seed {random_seed}")
            args.random_seed = random_seed
            logger.info(args)
            test_acc, val_acc = train(args, return_value="test")
            test_acc_list.append(test_acc)
            val_acc_list.append(val_acc)
            logger.warning(f"current val_acc {np.mean(val_acc_list)} ± {np.std(val_acc_list)}")
            logger.warning(f"current test_acc {np.mean(test_acc_list)} ± {np.std(test_acc_list)}")
        cleanup()
        logger.critical(f"final val_acc {np.mean(val_acc_list)} ± {np.std(val_acc_list)}")
        logger.critical(f"final test_acc {np.mean(test_acc_list)} ± {np.std(test_acc_list)}")


if __name__ == "__main__":
    args = parse_args()
    save_args(args, args.output_dir)
    main(args)
