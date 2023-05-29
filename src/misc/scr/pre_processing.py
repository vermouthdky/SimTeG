import argparse
import os

import numpy as np
import torch
from cogdl.data import Graph
from cogdl.utils import spmm_cpu
from ogb.nodeproppred import NodePropPredDataset
from torch_sparse import spmm
from tqdm import tqdm


def build_cogdl_graph(name, root):
    dataset = NodePropPredDataset(name=name, root=root)
    graph, y = dataset[0]
    x = torch.tensor(graph["node_feat"]).float().contiguous() if graph["node_feat"] is not None else None
    y = torch.tensor(y.squeeze())
    row, col = graph["edge_index"][0], graph["edge_index"][1]
    row = torch.from_numpy(row)
    col = torch.from_numpy(col)
    edge_index = torch.stack([row, col], dim=0)
    graph = Graph(x=x, edge_index=edge_index, y=y)
    graph.splitted_idx = dataset.get_idx_split()

    return graph


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-products")
parser.add_argument("--data_dir", type=str, default="../cogdl_data")
parser.add_argument("--lm_model_type", type=str, default=None)
parser.add_argument("--num_hops", type=int, default=5)
# NOTE: set root = output_dir
parser.add_argument("--output_dir", type=str)
parser.add_argument("--giant_path", type=str, default=None)
parser.add_argument("--bert_x_dir", type=str, default=None)

args = parser.parse_args()
print(args)

graph = build_cogdl_graph(name=args.dataset, root=args.data_dir)
splitted_idx = graph.splitted_idx
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

__import__("ipdb").set_trace()

graph.row_norm()
feats = [graph.x]
print("Compute neighbor-averaged feats")
for hop in tqdm(range(1, args.num_hops + 1)):
    # feats.append(spmm_cpu(graph, feats[-1]))
    feat = spmm(graph.edge_index, graph.edge_weight, graph.num_nodes, graph.num_nodes, feats[-1])
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
