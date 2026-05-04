import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGEConv
from torch.nn import Linear, Sequential, ReLU, Dropout, LayerNorm

class SimplePupilGNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super(SimplePupilGNN, self).__init__()
        # GCN Layers (Message Passing)
        # self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # SAGE Layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        # LayerNorm between conv blocks. LayerNorm normalizes per-node across the
        # feature dim so it works regardless of how many nodes a graph has and
        # regardless of batch size, unlike BatchNorm1d.
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        # Regression Head
        # LayerNorm (not BatchNorm1d) so batch_size=1 works in training mode.
        # Using concat(mean_pool, max_pool) so MLP input dim is 2 * hidden_dim.
        self.mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(0.3),
            Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            ReLU(),
            Dropout(0.1),
            Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = x.relu()

        # 2. Readout layer: Pool node features into a single graph-level vector
        x_mean = global_mean_pool(x, batch, size=data.num_graphs)
        x_max = global_max_pool(x, batch, size=data.num_graphs)
        x = torch.cat([x_mean, x_max], dim=-1)

        # 3. Final regression
        return self.mlp(x)