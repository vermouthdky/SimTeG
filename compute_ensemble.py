import argparse

import numpy as np
import torch
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.transforms import ToSparseTensor
from tqdm import tqdm

from src.dataset import load_data_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--list_logits", type=str, help="for ensembling")
parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="for ensembling")
parser.add_argument("--c_and_s", action="store_true", help="correct and smoothing")
parser.add_argument("--weights", nargs="+", type=float)
parser.add_argument("--start_seed", type=int, default=0)
args = parser.parse_args()


def ensembling(list_logits, c_and_s=False):
    data, split_idx, evaluator = load_data_bundle(
        "ogbn-arxiv", root="../data", tokenizer=None, tokenize=False
    )
    list_logits = [torch.load(logits).cpu() for logits in list_logits]
    weights = np.asarray(args.weights) / sum(args.weights)
    list_logits = [
        logits.softmax(dim=-1) * weight for logits, weight in zip(list_logits, weights)
    ]
    y_pred = sum(list_logits) / len(list_logits)

    if c_and_s:
        y_pred = correct_and_smooth(data, split_idx, y_pred)

    y_pred = y_pred.argmax(dim=-1, keepdim=True)
    y_true = data.y
    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, val_acc, test_acc


def compute():
    args = parser.parse_args()
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for seed in range(args.start_seed, args.start_seed + 10):
        list_logits = args.list_logits.split(" ")
        list_logits = [logits + f"/logits_seed{seed}.pt" for logits in list_logits]
        train_acc, val_acc, test_acc = ensembling(list_logits, c_and_s=args.c_and_s)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        # print("cur train_acc: {:.2f} ± {:.2f}".format(100 * np.mean(train_acc_list), 100 * np.std(train_acc_list)))
        # print("cur val_acc: {:.2f} ± {:.2f}".format(100 * np.mean(val_acc_list), 100 * np.std(val_acc_list)))
        # print("cur test_acc: {:.2f} ± {:.2f}".format(100 * np.mean(test_acc_list), 100 * np.std(test_acc_list)))
    print(
        "train_acc: {:.2f} ± {:.2f}".format(
            100 * np.mean(train_acc_list), 100 * np.std(train_acc_list)
        )
    )
    print(
        "val_acc: {:.2f} ± {:.2f}".format(
            100 * np.mean(val_acc_list), 100 * np.std(val_acc_list)
        )
    )
    print(
        "test_acc: {:.2f} ± {:.2f}".format(
            100 * np.mean(test_acc_list), 100 * np.std(test_acc_list)
        )
    )


def correct_and_smooth(data, split_idx, y_soft):
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    if not hasattr(data, "adj_t"):
        data = ToSparseTensor()(data)
    adj_t = data.adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

    y, train_idx = data.y.to(device), split_idx["train"].to(device)
    y_train = y[train_idx]
    y_soft = y_soft.to(device)

    post = CorrectAndSmooth(
        num_correction_layers=50,
        correction_alpha=0.5,
        num_smoothing_layers=50,
        smoothing_alpha=0.5,
        autoscale=False,
        scale=20.0,
    )

    print("Correct and smooth...")
    y_soft = post.correct(y_soft, y_train, train_idx, DAD)
    y_soft = post.smooth(y_soft, y_train, train_idx, DA)
    print("Done!")

    return y_soft


if __name__ == "__main__":
    compute()
