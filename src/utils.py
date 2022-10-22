import logging
import sys
import os
from datetime import datetime


def set_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    date_time = datetime.now().strftime("%Y-%b-%d_%H-%M")
    file_handler = logging.FileHandler("./logs/{}.log".format(date_time))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
