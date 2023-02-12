from torch import nn

from .modules.SAGN import SAGN as PureSAGN
from .modules.SIGN import SIGN as PureSIGN


class SAGN(nn.Module):
    def __init__(self, args):
        super(SAGN, self).__init__()
        self.gnn_model = PureSAGN(args, num_hops=args.gnn_num_layers + 1)

    def forward(self, xs):
        return self.gnn_model(xs)


class SIGN(nn.Module):
    def __init__(self, args):
        super(SIGN, self).__init__()
        self.gnn_model = PureSIGN(args, num_hops=args.gnn_num_layers + 1)

    def forward(self, xs):
        return self.gnn_model(xs)
