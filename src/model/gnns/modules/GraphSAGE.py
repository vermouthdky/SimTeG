import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout):
        super().__init__()
        self.dropout_rate = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device, subgraph_loader: NeighborLoader) -> Tensor:
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[: batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)

        return x_all
