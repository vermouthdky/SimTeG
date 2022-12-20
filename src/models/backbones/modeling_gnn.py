import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv, GINConv


class GNNModel(torch.nn.Module):
    def __init__(self, config):
        super(GNNModel, self).__init__()
        self.model = None
        self.dropout = config.gnn_dropout
        self.num_layers = config.gnn_num_layers
        model_class = {"GIN": GINConv, "GAT": GATConv, "GraphSAGE": SAGEConv}
        self.convs = torch.nn.ModuleList(
            [model_class[config.gnn_type](config.hidden_size, config.hidden_size) for _ in range(self.num_layers)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x, edge_index, device):
        self.dataloader = NeighborSampler(edge_index, size=[-1], batch_size=100, shuffle=False)
        for i, conv in enumerate(self.convs):
            xs = []
            for _, n_id, adj in self.dataloader:
                edge_index, _, size = adj.to(device)
                x = x[n_id].to(device)
                x = conv(x, edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all
