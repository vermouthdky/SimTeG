import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--cont", type=bool, default=False)

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="./data/")
    parser.add_argument("--model_type", type=str, default="G-BERT")
    parser.add_argument("--model_dir", type=str, default="./ckpt")  # path to save
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained")
    parser.add_argument("--ckpt_name", type=str, default="./ckpt/G-BERT/epoch-1.pt")

    # essential hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
