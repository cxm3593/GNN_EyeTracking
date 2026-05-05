import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGEConv, GINEConv
from torch.nn import Linear, Sequential, ReLU, Dropout, LayerNorm


def _gine_inner_mlp(in_dim: int, out_dim: int) -> Sequential:
    '''
    The MLP that GINEConv applies to (1+eps)*h_v + sum_u ReLU(h_u + e_uv).
    A small two-layer MLP is the conventional choice and gives the layer enough
    expressive power that GINE actually beats GCN/SAGE in practice.
    '''
    return Sequential(
        Linear(in_dim, out_dim),
        ReLU(),
        Linear(out_dim, out_dim),
    )


class SimplePupilGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 2,
        conv_dropout: float = 0.0,
        conv_type: str = "sage",   # "sage" | "gine"
        edge_dim: int = 3,         # Only used when conv_type == "gine".
    ):
        super(SimplePupilGNN, self).__init__()
        # GCN Layers (Message Passing)
        # self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.conv_type = conv_type.lower()
        if self.conv_type == "sage":
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        elif self.conv_type == "gine":
            # GINEConv passes edge_attr through an internal Linear(edge_dim, in_channels)
            # before adding it to the source-node features inside its message function.
            # This is the path that lets the network actually USE the relative-position
            # geometry that SAGE was throwing away.
            self.conv1 = GINEConv(_gine_inner_mlp(input_dim, hidden_dim), edge_dim=edge_dim)
            self.conv2 = GINEConv(_gine_inner_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
            self.conv3 = GINEConv(_gine_inner_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
        else:
            raise ValueError(f"Unknown conv_type='{conv_type}'. Use 'sage' or 'gine'.")

        # LayerNorm between conv blocks. LayerNorm normalizes per-node across the
        # feature dim so it works regardless of how many nodes a graph has and
        # regardless of batch size, unlike BatchNorm1d.
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        # Per-node dropout applied after each conv->norm->relu block. Acts as
        # regularization on the encoder side; default 0.0 keeps prior behavior
        # for callers that don't pass conv_dropout.
        self.conv_dropout = float(conv_dropout)

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

    def _conv(self, conv_layer, x, edge_index, edge_attr):
        '''Dispatch to the right call signature based on conv_type.'''
        if self.conv_type == "gine":
            return conv_layer(x, edge_index, edge_attr=edge_attr)
        return conv_layer(x, edge_index)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None) if self.conv_type == "gine" else None

        # 1. Obtain node embeddings
        x = self._conv(self.conv1, x, edge_index, edge_attr)
        x = self.norm1(x)
        x = x.relu()
        x = F.dropout(x, p=self.conv_dropout, training=self.training)
        x = self._conv(self.conv2, x, edge_index, edge_attr)
        x = self.norm2(x)
        x = x.relu()
        x = F.dropout(x, p=self.conv_dropout, training=self.training)
        x = self._conv(self.conv3, x, edge_index, edge_attr)
        x = self.norm3(x)
        x = x.relu()
        x = F.dropout(x, p=self.conv_dropout, training=self.training)

        # 2. Readout layer: Pool node features into a single graph-level vector
        x_mean = global_mean_pool(x, batch, size=data.num_graphs)
        x_max = global_max_pool(x, batch, size=data.num_graphs)
        x = torch.cat([x_mean, x_max], dim=-1)

        # 3. Final regression
        return self.mlp(x)