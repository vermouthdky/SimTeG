import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout, use_gpt_preds=False
    ):
        super().__init__()
        if use_gpt_preds:
            self.encoder = torch.nn.Embedding(out_channels + 1, hidden_channels)
        self.dropout_rate = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if hasattr(self, "encoder"):
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    # @torch.no_grad()
    # def inference(self, x_all: Tensor, device: torch.device, subgraph_loader: NeighborLoader) -> Tensor:
    #     # Compute representations of nodes layer by layer, using *all*
    #     # available edges. This leads to faster computation in contrast to
    #     # immediately computing the final representations of each batch:
    #     if hasattr(self, "encoder"):
    #         xs = []
    #         for batch in subgraph_loader:
    #             x = x_all[batch.n_id.to(x_all.device)].to(device)
    #             x = self.encoder(x)
    #             x = torch.flatten(x, start_dim=1)
    #             xs.append(x.cpu())
    #         x_all = torch.cat(xs, dim=0)
    #     for i, conv in enumerate(self.convs):
    #         xs = []
    #         for batch in subgraph_loader:
    #             x = x_all[batch.n_id.to(x_all.device)].to(device)
    #             x = conv(x, batch.edge_index.to(device))
    #             x = x[: batch.batch_size]
    #             if i < len(self.convs) - 1:
    #                 x = x.relu_()
    #             xs.append(x.cpu())
    #         x_all = torch.cat(xs, dim=0)

    #     return x_all
