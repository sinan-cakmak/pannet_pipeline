"""
STAGE 5 — GIN (Graph Isomorphism Network) feature extractor.

The GIN was chosen for PanNET infiltration scoring because its SUM aggregation
can "count" the number of tumor–NNP contact points. More invasive tumors create
more interfaces, and GIN captures this magnitude naturally — unlike GAT's
weighted averaging which normalizes this signal away.

Architecture per layer:
  GINConv(
    MLP: Linear(256, 256) → RMSNorm → ReLU → Dropout(0.4)
       → Linear(256, 256) → RMSNorm → ReLU
  )
  With residual connection: x = conv(dropout(x)) + x

After all layers, ONLY PanNET nodes are pooled (via global_add_pool).
NNP nodes served their purpose during message passing — their information
now lives inside the updated PanNET node embeddings. Pooling NNP nodes
separately would dilute the tumor-specific signal.

Key finding from thesis (Table 5.7):
  1 layer + radius 3 = best performance. Deeper GNNs hurt at larger radii
  because the bipartite structure already provides sufficient context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

from src.constants import HIDDEN_DIM, PANNET_CLASS_ID


class GINFeatureExtractor(nn.Module):
    """
    GIN-based feature extractor for bipartite WSI graphs.

    Applies GIN message passing layers with residual connections,
    then pools ONLY the PanNET (tumor) node embeddings.

    Args:
        num_layers: Number of GIN message passing layers (default: 1)
        hidden_dim: Hidden dimension (default: 256, matches autoencoder output)
        dropout: Dropout rate (default: 0.4)
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.dropout = dropout

        # Build GIN layers, each with a 2-layer MLP inside
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.RMSNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.RMSNorm(hidden_dim),
                nn.ReLU(),
            )
            self.convs.append(GINConv(mlp))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        patch_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: message passing + PanNET-only pooling.

        Args:
            x: (N, hidden_dim) node features
            edge_index: (2, E) edge list
            batch: (N,) batch assignment per node
            patch_classes: (N,) tissue class per node

        Returns:
            (B, hidden_dim) graph-level embeddings (one per WSI in the batch)
        """
        for conv in self.convs:
            # Dropout before message passing (not after — matches thesis)
            x_dropped = F.dropout(x, p=self.dropout, training=self.training)
            # GIN message passing + residual connection
            x = conv(x_dropped, edge_index) + x

        # Pool ONLY PanNET nodes — this is the "context-aware" part.
        # After message passing, PanNET nodes have absorbed NNP context.
        # Pooling NNP nodes separately would dilute the tumor signal.
        pannet_mask = patch_classes == PANNET_CLASS_ID
        x_pannet = x[pannet_mask]
        batch_pannet = batch[pannet_mask]

        # Sum pooling (not mean) — captures the MAGNITUDE of the interface
        return global_add_pool(x_pannet, batch_pannet)
