import logging
import os
import sys
from datetime import datetime
from typing import List, Union

import colorlog
import torch
import torch.distributed as dist


class EmbeddingHandler:
    def __init__(self, emb_path: str):
        self.emb_path = emb_path
        if not os.path.exists(self.emb_path):
            os.makedirs(self.emb_path)

    def save(self, emb: torch.Tensor, saved_name: str):
        saved_name = os.path.join(self.emb_path, saved_name)
        rank = int(os.environ["RANK"]) if is_dist() else -1
        if rank <= 0:
            torch.save(emb, saved_name)
        dist.barrier()

    def load(self, saved_name: str):
        if not self.has(saved_name):
            return None
        return torch.load(os.path.join(self.emb_path, saved_name))

    def has(self, saved_name: Union[str, List[str]]):
        if isinstance(saved_name, List):
            return all([os.path.exists(os.path.join(self.emb_path, name)) for name in saved_name])
        return os.path.exists(os.path.join(self.emb_path, saved_name))


def dataset2foldername(dataset):
    assert dataset in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]
    name_dict = {"ogbn-arxiv": "ogbn_arxiv", "ogbn-products": "ogbn_products", "ogbn-papers100M": "ogbn_papers100M"}
    return name_dict[dataset]


def is_dist():
    return False if os.getenv("WORLD_SIZE") is None else True


class RankFilter(logging.Filter):
    def filter(self, rec):
        return is_dist() == False or int(os.environ["RANK"]) == 0


def set_logging():
    root = logging.getLogger()
    # NOTE: clear the std::out handler first to avoid duplicated output
    if root.hasHandlers():
        root.handlers.clear()

    root.setLevel(logging.INFO)
    log_format = "%(message)s"
    color_format = "%(log_color)s" + log_format

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colorlog.ColoredFormatter(color_format))
    console_handler.addFilter(RankFilter())
    root.addHandler(console_handler)


if __name__ == "__main__":
    set_logging()
    logger = logging.getLogger(__name__)
    logger.info("test")
