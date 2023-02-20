import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "save_bert_x"])
    parser.add_argument("--single_gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--cont", type=bool, default=False)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="main")

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--model_type", type=str, default="GBert")
    parser.add_argument("--output_dir", type=str)  # output dir
    parser.add_argument("--ckpt_dir", type=str)  # ckpt path to save
    parser.add_argument("--ckpt_name", type=str, default="TGRoberta-best.pt")  # ckpt name to be loaded
    parser.add_argument("--save_ckpt_per_valid", type=bool, default=False)
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="has to be consistent with repo_id in huggingface",
    )
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--scheduler_type", type=str, default="linear")

    # dataset args
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--num_feats", type=int)

    # flag
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--use_bert_x", action="store_true", default=False)
    parser.add_argument("--use_adapter", action="store_true", default=False)
    parser.add_argument("--optuna", type=bool, default=False, help="use optuna to tune hyperparameters")

    # training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accum_interval", type=int, default=5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="consistent with the one in bert")
    parser.add_argument("--header_dropout_prob", type=float, default=0.2)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adapter_hidden_size", type=int, default=64)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--init_alpha", type=float, default=0.8)
    parser.add_argument("--scheduler_warmup", type=float, default=0.6)

    # gnn hyperparameters
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_type", type=str, default="SIGN")
    parser.add_argument("--gnn_dropout", type=float, default=0.2)
    parser.add_argument("--gnn_dim_hidden", type=int, default=256)

    args = parser.parse_args()
    args = _post_init(args)
    return args


def save_args(args, dir):
    FILE_NAME = "args.json"
    with open(os.path.join(dir, FILE_NAME), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logger.info("args saved to {}".format(os.path.join(dir, FILE_NAME)))


def load_args(dir):
    with open(os.path.join(dir, "args.txt"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    return args


def _post_init(args):
    args = _set_dataset_specific_args(args)
    return args


def _set_dataset_specific_args(args):
    if args.dataset == "ogbn-arxiv":
        args.num_labels = 40
        args.num_feats = 768 if args.use_bert_x else 128

    elif args.dataset == "ogbn-products":
        args.num_labels = 47
        args.num_feats = 100

    return args


if __name__ == "__main__":
    args = parse_args()
