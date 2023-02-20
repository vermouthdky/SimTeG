import json
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch.multiprocessing as mp

from src.args import load_args, parse_args, save_args
from src.run import save_bert_x, test, train
from src.utils import is_dist, set_logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=ExperimentalWarning)


def set_env(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_dist():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if is_dist():
            gpus = ",".join([str(_) for _ in range(int(os.environ["WORLD_SIZE"]))])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):
    set_logging()
    logger.info(args)
    set_env(args.random_seed)
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "save_bert_x":
        save_bert_x(args)
    else:
        raise NotImplementedError("running mode should be either 'train' or 'test'")


if __name__ == "__main__":
    args = parse_args()
    save_args(args, args.output_dir)
    main(args)
