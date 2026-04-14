'''
This module is planned to be a toolbox for GNN operations
@author: Chengyi Ma
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_cluster import radius_graph


class GnnBuilder:
    '''
    GnnBuilder builds graph representations from point cloud data
    and provides utilities for inspection and visualization.
    '''

    def build_radius_graph(self, points: torch.Tensor, r: float, max_num_neighbors: int = 32) -> Data:
        '''
        Build an undirected radius graph from a set of 3D points.
        Each node is connected to all other nodes within Euclidean distance r.
        Args:
            points: (N, 3) float tensor with columns [x, y, t]
            r: radius threshold for edge creation
            max_num_neighbors: maximum edges per node (avoids quadratic blowup on dense regions)
        Returns:
            a torch_geometric.data.Data object with:
                - x:          (N, 3) node feature matrix (the points themselves)
                - edge_index: (2, E) connectivity in COO format
                - pos:        (N, 3) node positions (same as x)
        '''
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        else:
            points = points.float()

        edge_index = radius_graph(points, r=r, loop=False, max_num_neighbors=max_num_neighbors)

        return Data(x=points, edge_index=edge_index, pos=points)

    def visualize_graph_3d(self, graph: Data, title: str = "Graph Visualization", max_edges: int = 5000) -> None:
        '''
        Visualize a 3D point-cloud graph. Nodes are plotted as scatter points
        and edges as line segments. Colour encodes t (time).
        Axes are (t, x, y): horizontal t, depth x, vertical y — swapped from
        the stored order [x, y, t] for clearer inspection.
        Args:
            graph: a Data object produced by build_radius_graph
            title: plot title
            max_edges: cap on the number of edges drawn (random subsample if exceeded),
                       to keep rendering fast on dense graphs
        '''
        pos = graph.pos.numpy()          # (N, 3): x, y, t
        edge_index = graph.edge_index.numpy()  # (2, E)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # pos columns: 0=x, 1=y, 2=t — plot as (t, x, y)
        px, py, pt = pos[:, 0], pos[:, 1], pos[:, 2]

        # --- nodes (coloured by t) ---
        sc = ax.scatter(pt, px, py,
                        c=pt, cmap='viridis', s=8, alpha=0.7, zorder=3)
        plt.colorbar(sc, ax=ax, label='t (scaled time)', pad=0.1)

        # --- edges (subsample if too many) ---
        num_edges = edge_index.shape[1]
        indices = np.arange(num_edges)
        if num_edges > max_edges:
            indices = np.random.choice(indices, size=max_edges, replace=False)

        src, dst = edge_index[0, indices], edge_index[1, indices]
        for s, d in zip(src, dst):
            ax.plot([pt[s], pt[d]],
                    [px[s], px[d]],
                    [py[s], py[d]],
                    color='steelblue', alpha=0.15, linewidth=0.5)

        ax.set_xlabel('t (scaled time)')
        ax.set_ylabel('x (pixels)')
        ax.set_zlabel('y (pixels)')
        ax.set_title(f'{title}\n{pos.shape[0]} nodes, {num_edges} edges'
                     + (f' ({max_edges} shown)' if num_edges > max_edges else ''))
        plt.tight_layout()
        plt.show()
