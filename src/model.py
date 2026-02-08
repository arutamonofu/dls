# src/model.py

import torch
from torch import nn
from egnn_pytorch import EGNN_Sparse
from torch_geometric.nn import global_add_pool

class HybridEGNN(nn.Module):
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int,
                 out_dim: int = 1,
                 n_egnn_layers: int = 4,
                 tda_feature_dim: int = 0):
        super().__init__()
        self.tda_feature_dim = tda_feature_dim

        self.embedding = nn.Embedding(10, node_feature_dim)

        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(feats_dim=hidden_dim, pos_dim=3) for _ in range(n_egnn_layers)
        ])
        
        self.node_in_proj = nn.Linear(node_feature_dim, hidden_dim)

        mlp_in_dim = hidden_dim + tda_feature_dim
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, edge_index, tda_x: torch.Tensor = None) -> torch.Tensor:
        h = self.embedding(z)
        h = self.node_in_proj(h)

        for egnn_layer in self.egnn_layers:
            combined = torch.cat([pos, h], dim=-1)
            combined = egnn_layer(combined, edge_index, batch=batch)
            pos = combined[:, :3]
            h = combined[:, 3:]

        graph_vec = global_add_pool(h, batch)

        if tda_x is not None:
            if self.tda_feature_dim == 0:
                raise ValueError("Модель инициализирована без tda_dim, но tda_x есть.")
            
            if tda_x.shape[0] != graph_vec.shape[0]:
                 raise ValueError(f"Несовпадение размера батчей: Graph ({graph_vec.shape[0]}) vs TDA ({tda_x.shape[0]}).")
            
            combined_vec = torch.cat([graph_vec, tda_x], dim=-1)
        else:
            combined_vec = graph_vec

        prediction = self.mlp_head(combined_vec)
        return prediction