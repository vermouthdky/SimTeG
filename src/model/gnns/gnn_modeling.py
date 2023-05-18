import torch
from torch import nn

from .modules.GAMLP import JK_GAMLP as PureGAMLP
from .modules.GAMLP import JK_GAMLP_RLU as PureGAMLPSLE
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
        self.gnn_model = SAGE(args.num_feats, args.hidden_size, args.num_labels, args.gnn_num_layers, args.gnn_dropout)

    def forward(self, x, edge_index):
        return self.gnn_model(x, edge_index)

    # @torch.no_grad()
    # def inference(self, x_all, device, subgraph_loader):
    #     return self.gnn_model.inference(x_all, device, subgraph_loader)
