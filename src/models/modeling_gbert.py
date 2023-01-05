import logging
import os
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.transforms import SIGN
from torch_sparse import SparseTensor
from transformers import AutoModel

from .GNNs import SAGN

"""
Procedure:
INPUT: adj_t, x
precomputing: x^1, ..., x^l
forward: \rho(x, x^1, ..., x^l, bert(x))
"""


class GBert(nn.Module):
    def __init__(self, args):
        super(GBert, self).__init__()
        self.bert_model = AutoModel.from_pretrained(args.pretrained_model)
        self.gnn_model = SAGN(args)

    def forward(self, xs, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        bert_out = bert_out[0]
        xs.append(bert_out)
        out = self.gnn_model(xs)
        return out
