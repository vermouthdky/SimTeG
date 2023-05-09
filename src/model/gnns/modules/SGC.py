import torch
from torch import nn


class SGC(nn.Module):
    def __init__(self, num_feats, num_classes):
        super(SGC, self).__init__()
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.lin = nn.Linear(num_feats, num_classes)

    def forward(self, xs: torch.Tensor):
        xs = [x for x in torch.split(xs, self.num_feats, -1)]
        return self.lin(xs[-1])
