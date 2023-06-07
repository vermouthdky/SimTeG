from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import SAGEConv
from tqdm.auto import tqdm

from .modules.GCN import GCN as PureGCN
from .modules.GraphSAGE import SAGE


class LinkGraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(LinkGraphSAGE, self).__init__()

        self.dropout = args.gnn_dropout
        self.num_layers = args.gnn_num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(args.num_feats, args.gnn_dim_hidden))
        for _ in range(args.gnn_num_layers - 2):
            self.convs.append(SAGEConv(args.gnn_dim_hidden, args.gnn_dim_hidden))
        self.convs.append(SAGEConv(args.gnn_dim_hidden, args.gnn_dim_hidden))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description("Evaluating")

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class LinkGCN(nn.Module):
    def __init__(self, args):
        super(LinkGCN, self).__init__()
        self.gnn_model = PureGCN(
            args.num_feats, args.hidden_size, args.hidden_size, args.gnn_num_layers, args.gnn_dropout
        )

    def reset_parameters(self):
        self.gnn_model.reset_parameters()

    def forward(self, x, edge_index):
        return self.gnn_model(x, edge_index)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        return self.gnn_model.inference(x_all, device, subgraph_loader)


class LinkMLP(nn.Module):
    def __init__(self, args):
        super(LinkMLP, self).__init__()
        self.dense = nn.Linear(args.num_feats, args.gnn_dim_hidden)
        self.dropout = nn.Dropout(args.gnn_dropout)
        self.out_proj = nn.Linear(args.gnn_hidden_hidden, args.gnn_dim_hidden)

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
