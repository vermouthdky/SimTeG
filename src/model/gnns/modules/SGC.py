import torch
from torch import nn


class SGC(nn.Module):
    def __init__(self, num_feats, num_labels, dropout):
        super(SGC, self).__init__()
        self.num_feats = hidden_size = num_feats
        self.dense = nn.Linear(num_feats, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, xs: torch.Tensor):
        xs = [x for x in torch.split(xs, self.num_feats, -1)]
        x = xs[-1]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
