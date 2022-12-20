import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--single_cuda", type=int, default=0)
    # parser.add_argument("--world_size", type=int, default=6)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--cont", type=bool, default=False)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--local_rank", type=int)

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--model_type", type=str, default="TGRoberta")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")  # path to save
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained")
    parser.add_argument(
        "--pretrained_model", type=str, default="roberta-base", help="has to be consistent with repo_id in huggingface"
    )
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--debug", action="store_true", default=False, help="load partial datset ogbn-arxiv")

    # dataset args
    parser.add_argument("--num_labels", type=int)

    # essential hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--accum_interval", type=int, default=5)

    # gnn hyperparameters
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE", choices=["GIN", "GAT", "GraphSAGE"])
    parser.add_argument("--gnn_dropout", type=float, default=0.2)

    args = parser.parse_args()
    args = _set_dataset_specific_args(args)
    logging.info(args)
    return args


def _set_dataset_specific_args(args):
    if args.dataset == "ogbn-arxiv":
        args.num_labels = 128

    elif args.dataset == "ogbn-products":
        args.num_classes = 47

    return args


if __name__ == "__main__":
    args = parse_args()
