import torch
from torch import nn

from .modules.GAMLP import JK_GAMLP as PureGAMLP
from .modules.SAGN import SAGN as PureSAGN
from .modules.SIGN import SIGN as PureSIGN


class SAGN(nn.Module):
    def __init__(self, args):
        super(SAGN, self).__init__()
        self.gnn_model = PureSAGN(args, num_hops=args.gnn_num_layers + 1)
        self.dim_feats = args.num_feats

    def forward(self, xs: torch.Tensor):
        xs = [x for x in torch.split(xs, self.dim_feats, -1)]
        return self.gnn_model(xs)


class SIGN(nn.Module):
    def __init__(self, args):
        super(SIGN, self).__init__()
        self.gnn_model = PureSIGN(args, num_hops=args.gnn_num_layers + 1)
        self.dim_feats = args.num_feats

    def forward(self, xs: torch.Tensor):
        xs = [x for x in torch.split(xs, self.dim_feats, -1)]
        return self.gnn_model(xs)


class GAMLP(nn.Module):
    def __init__(self, args):
        super(GAMLP, self).__init__()
        self.gnn_model = PureGAMLP(args)
        self.dim_feats = args.num_feats

    def forward(self, xs: torch.Tensor):
        xs = [x for x in torch.split(xs, self.dim_feats, -1)]
        return self.gnn_model(xs)
