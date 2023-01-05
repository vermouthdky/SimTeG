import logging
import os
import sys
from datetime import datetime

from torch.distributed import barrier


def set_trace():
    if int(os.environ["RANK"]) == 0:
        __import__("ipdb").set_trace()
    else:
        barrier()


class RankFilter(logging.Filter):
    def filter(self, rec):
        return int(os.environ["RANK"]) == 0


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

    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    date_time = datetime.now().strftime("%Y-%b-%d_%H-%M")
    file_handler = logging.FileHandler("./logs/{}.log".format(date_time))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RankFilter())  # setup filter
    root.addHandler(file_handler)


if __name__ == "__main__":
    set_logging()
    logger = logging.getLogger(__name__)
    logger.info("test")
