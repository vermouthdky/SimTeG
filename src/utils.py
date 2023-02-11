import logging
import os
import sys
from datetime import datetime

from torch.distributed import barrier


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
    formatter = logging.Formatter("[%(name)s %(asctime)s] %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RankFilter())
    root.addHandler(console_handler)
    # if not os.path.exists("./logs"):
    #     os.mkdir("./logs")
    # date_time = datetime.now().strftime("%Y-%b-%d_%H-%M")
    # file_handler = logging.FileHandler("./logs/{}.log".format(date_time))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # file_handler.addFilter(RankFilter())  # setup filter
    # root.addHandler(file_handler)


if __name__ == "__main__":
    set_logging()
    logger = logging.getLogger(__name__)
    logger.info("test")
