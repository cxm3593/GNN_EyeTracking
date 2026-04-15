import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

class SimplePupilGNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super(SimplePupilGNN, self).__init__()
        # GCN Layers (Message Passing)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Regression Head
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 2. Readout layer: Pool node features into a single graph-level vector
        # This is critical for predicting one (x,y) per window
        x = global_mean_pool(x, batch, size=data.num_graphs)

        # 3. Final regression
        return self.fc(x)