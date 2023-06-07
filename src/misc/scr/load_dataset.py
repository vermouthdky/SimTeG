import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.transforms import ToSparseTensor
from tqdm import tqdm


def prepare_label_emb(args, graph, labels, n_classes, train_node_nums, label_teacher_emb=None):
    if label_teacher_emb == None:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[:train_node_nums] = (
            F.one_hot(labels[:train_node_nums].to(torch.long), num_classes=n_classes).float().squeeze(1)
        )
        y = torch.FloatTensor(y)
    else:
        print("use teacher label")
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_node_nums:] = label_teacher_emb[train_node_nums:]
        y[:train_node_nums] = (
            F.one_hot(labels[:train_node_nums].to(torch.long), num_classes=n_classes).float().squeeze(1)
        )
        y = torch.FloatTensor(y)
    sparse_graph = ToSparseTensor()(graph)
    deg = sparse_graph.adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * sparse_graph.adj_t * deg_inv_sqrt.view(1, -1)
    for hop in range(args.label_num_hops):
        y = adj_t @ y
    del adj_t, sparse_graph
    return y


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval(
        {
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        }
    )["acc"]


def load_dataset(name, device, args, return_nid=False):
    """
    Load dataset and move graph and features to device
    """
    if name not in ["ogbn-products", "ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))

    dataset = PygNodePropPredDataset(name=name, root=args.data_dir)
    splitted_idx = dataset.get_idx_split()
    graph = dataset.data

    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    assert train_nid.max() <= val_nid.min()
    assert val_nid.max() <= test_nid.min()
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    evaluator = get_ogb_evaluator(name)

    print(
        f"# Nodes: {graph.num_nodes}\n"
        f"# Edges: {graph.num_edges}\n"
        f"# Train: {len(train_nid)}\n"
        f"# Val: {len(val_nid)}\n"
        f"# Test: {len(test_nid)}\n"
        f"# Classes: {47}\n"
    )

    if not return_nid:
        return graph, train_node_nums, valid_node_nums, test_node_nums, evaluator
    train_nid = torch.LongTensor(train_nid)
    val_nid = torch.LongTensor(val_nid)
    test_nid = torch.LongTensor(test_nid)
    return graph, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """

    data = load_dataset(args.dataset, device, args)

    graph, train_node_nums, valid_node_nums, test_node_nums, evaluator = data

    if args.giant_path is not None and args.giant:
        graph.x = torch.tensor(np.load(args.giant_path)).float()
        print("Pretrained node feature loaded! Path: {}".format(args.giant_path))
    elif args.bert_x_dir is not None and args.use_bert_x:
        graph.x = torch.load(args.bert_x_dir).float()
        print(f"graph.x.shape: {graph.x.shape}")

    # edge_index to adj_t
    sparse_graph = ToSparseTensor()(graph)
    deg = sparse_graph.adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * sparse_graph.adj_t * deg_inv_sqrt.view(1, -1)
    feats = [sparse_graph.x]

    for hop in tqdm(range(args.num_hops), desc="Compute neighbor-averaged feats"):
        feat = adj_t @ feats[-1]
        feats.append(feat)

    in_feats = feats[0].shape[1]
    del adj_t, sparse_graph
    gc.collect()

    if args.dataset == "ogbn-products":
        return (
            graph,
            feats,
            graph.y.view(-1),
            in_feats,
            47,  # num_classes
            train_node_nums,
            valid_node_nums,
            test_node_nums,
            evaluator,
        )
