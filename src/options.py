import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--cont", type=bool, default=False)

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="/raid/common")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--model_type", type=str, default="TGRoberta")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")  # path to save
    # parser.add_argument("--config_name", type=str, default="#TODO")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained")
    parser.add_argument("--pretrained_model", type=str, default="roberta")

    # essential hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_parts", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
