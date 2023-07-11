import argparse
import gc
import os

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import spmm
from tqdm import tqdm


def load_data(args):
    dataset = PygNodePropPredDataset(name="ogbn-products", root=args.data_dir)
    data = dataset.data
    split_idx = dataset.get_idx_split()
    return data, split_idx


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-products")
parser.add_argument("--data_dir", type=str, default="../data")
parser.add_argument("--lm_model_type", type=str, default=None)
parser.add_argument("--num_hops", type=int, default=5)
# NOTE: set root = output_dir
parser.add_argument("--output_dir", type=str)
parser.add_argument("--giant_path", type=str, default=None)
parser.add_argument("--bert_x_dir", type=str, default=None)

args = parser.parse_args()
print(args)

graph, splitted_idx = load_data(args)
train_nid = splitted_idx["train"]
val_nid = splitted_idx["valid"]
test_nid = splitted_idx["test"]

dirs = f"{args.output_dir}/feat"
if not os.path.exists(dirs):
    os.makedirs(dirs)

if args.giant_path != None:
    graph.x = torch.tensor(np.load(args.giant_path)).float()
    print("Pretrained node feature loaded! Path: {}".format(args.giant_path))
elif args.bert_x_dir != None:
    graph.x = torch.load(args.bert_x_dir).float()
    print("Pretrained node feature loaded! Path: {}".format(args.bert_x_dir))
    print(f"graph.x.shape: {graph.x.shape}")


# edge_index to adj_t
graph = ToSparseTensor()(graph)
deg = graph.adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
adj_t = deg_inv_sqrt.view(-1, 1) * graph.adj_t * deg_inv_sqrt.view(1, -1)
feats = [graph.x]
print("Compute neighbor-averaged feats")
for hop in tqdm(range(1, args.num_hops + 1)):
    feat = adj_t @ feats[-1]
    feats.append(feat)

for i, x in enumerate(feats):
    feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
    if args.giant_path is not None:
        save_dir = f"{dirs}/feat_{i}_giant.pt"
        print(f"saved feat_{i}_giant.pt to {save_dir}")
        torch.save(feats[i], save_dir)
    elif args.bert_x_dir is not None:
        save_dir = f"{dirs}/feat_{i}_{args.lm_model_type}.pt"
        print(f"saved feat_{i}_{args.lm_model_type}.pt to {save_dir}")
        torch.save(feats[i], save_dir)
    else:
        save_dir = f"{dirs}/feat_{i}.pt"
        print(f"saved feat_{i}.pt to {save_dir}")
        torch.save(feats[i], save_dir)
gc.collect()
