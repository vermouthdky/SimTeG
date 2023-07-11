from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .modules.GAMLP import JK_GAMLP as PureGAMLP
from .modules.GAMLP import JK_GAMLP_RLU as PureGAMLPSLE
from .modules.GCN import GCN as PureGCN
from .modules.GraphSAGE import SAGE
from .modules.SAGN import SAGN as PureSAGN
from .modules.SAGN import SAGN_SLE as PureSAGNSLE
from .modules.SGC import SGC as PureSGC
from .modules.SIGN import SIGN as PureSIGN


class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        self.gnn_model = PureSGC(args.num_feats, args.num_labels, args.gnn_dropout)

    def forward(self, x0: torch.Tensor, x_emb: torch.Tensor, labels=None):
        xs = torch.cat([x0, x_emb], dim=-1)
        return self.gnn_model(xs)


class SAGN(nn.Module):
    def __init__(self, args):
        super(SAGN, self).__init__()
        self.gnn_model = PureSAGN(args)
        self.dim_feats = args.num_feats

    def forward(self, x0: torch.Tensor, x_emb: torch.Tensor, labels=None):
        xs = torch.cat([x0, x_emb], dim=-1)
        return self.gnn_model(xs)


class SIGN(nn.Module):
    def __init__(self, args):
        super(SIGN, self).__init__()
        self.gnn_model = PureSIGN(args, num_hops=args.gnn_num_layers + 1)
        self.dim_feats = args.num_feats

    def forward(self, x0: torch.Tensor, x_emb: torch.Tensor, labels=None):
        xs = torch.cat([x0, x_emb], dim=-1)
        return self.gnn_model(xs)


class GAMLP(nn.Module):
    def __init__(self, args):
        super(GAMLP, self).__init__()
        self.gnn_model = PureGAMLP(args)
        self.dim_feats = args.num_feats

    def forward(self, x0: torch.Tensor, x_emb: torch.Tensor, labels=None):
        xs = torch.cat([x0, x_emb], dim=-1)
        return self.gnn_model(xs)


class GraphSAGE(nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        if args.use_gpt_preds:
            self.gnn_model = SAGE(
                5 * args.hidden_size,
                args.hidden_size,
                args.num_labels,
                args.gnn_num_layers,
                args.gnn_dropout,
                use_gpt_preds=True,
            )
        else:
            self.gnn_model = SAGE(
                args.num_feats,
                args.hidden_size,
                args.num_labels,
                args.gnn_num_layers,
                args.gnn_dropout,
                use_gpt_preds=False,
            )

    def reset_parameters(self):
        self.gnn_model.reset_parameters()

    def forward(self, x, edge_index):
        return self.gnn_model(x, edge_index)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        xs = []
        for batch in subgraph_loader:
            x = self.gnn_model(batch.x.to(device), batch.edge_index.to(device))[: batch.batch_size]
            xs.append(x.cpu())
        return torch.cat(xs, dim=0)


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.gnn_model = PureGCN(
            args.num_feats, args.hidden_size, args.num_labels, args.gnn_num_layers, args.gnn_dropout
        )

    def reset_parameters(self):
        self.gnn_model.reset_parameters()

    def forward(self, x, edge_index):
        return self.gnn_model(x, edge_index)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        return self.gnn_model.inference(x_all, device, subgraph_loader)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.dense = nn.Linear(args.num_feats, args.hidden_size)
        self.dropout = nn.Dropout(args.gnn_dropout)
        self.out_proj = nn.Linear(args.hidden_size, args.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    @torch.no_grad()
    def inference(self, device, all_loader):
        x_list = []
        for x in all_loader:
            x = self.forward(x.to(device))
            x_list.append(x.cpu())
        return torch.cat(x_list, dim=0)
