from .GNNs import SAGN as PureSAGN
from torch import nn


class SAGN(nn.Module):
    def __init__(self, args):
        super(SAGN, self).__init__()
        self.gnn_model = PureSAGN(args, num_hops=args.gnn_num_layers + 1)

    def forward(self, xs, input_ids, att_mask):
        return self.gnn_model(xs)
