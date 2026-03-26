"""
STAGE 5 — Cell-Conditioned GIN feature extractor.

Extends the bipartite GIN with a gate mechanism that modulates message
passing based on the cell composition of source and target patches.

The biological motivation: a message from an immune-rich stroma patch to a
tumor-dense PanNET patch carries different information than a message from
a cell-sparse connective patch. The gate learns WHICH cell composition
interactions matter for predicting infiltration patterns.

Gate mechanism per edge (j → i):
  cell_pair = [cell_j || cell_i]           # (E, 8) — concat both patches' cell info
  gate = sigmoid(MLP(cell_pair))           # (E, hidden_dim) — per-channel gate ∈ [0,1]
  msg = (W_msg @ h_j) * gate              # gated message
  h_i' = W_update @ [h_i || aggregate(msg)]  # update with residual

Adapted from pannet_gnn/src/models/gnn.py CellConditionedConv.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

from src.constants import HIDDEN_DIM, PANNET_CLASS_ID


class CellConditionedConv(MessagePassing):
    """
    Message passing layer where cell composition gates information flow.

    For each edge (j → i):
      1. Transform source features: msg = W_msg @ h_j
      2. Compute gate from cell compositions: gate = σ(MLP([cell_j || cell_i]))
      3. Apply gate: gated_msg = msg * gate  (element-wise)
      4. Aggregate: aggr = sum(gated_msg)  for all neighbors
      5. Update: h_i' = W_update @ [h_i || aggr]

    Args:
        in_channels: Input feature dimension (256)
        out_channels: Output feature dimension (256)
        cell_info_dim: Dimension of cell_information per node (4)
    """

    def __init__(
        self,
        in_channels: int = HIDDEN_DIM,
        out_channels: int = HIDDEN_DIM,
        cell_info_dim: int = 4,
    ):
        # aggr="add" = sum aggregation (same as GIN — counts interactions)
        super().__init__(aggr="add")

        # Gate network: takes concatenated cell info of source+target
        # Input: [cell_j || cell_i] = 2 * cell_info_dim = 8
        # Output: per-channel gate values in [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(cell_info_dim * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),  # Gate values ∈ [0, 1]
        )

        # Message transform: project source features
        self.msg_transform = nn.Linear(in_channels, out_channels)

        # Update: combine own features with aggregated gated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.RMSNorm(out_channels),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        cell_info: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge list
            cell_info: (N, cell_info_dim) cell composition per node
        """
        return self.propagate(edge_index, x=x, cell_info=cell_info)

    def message(
        self,
        x_j: torch.Tensor,
        cell_info_i: torch.Tensor,
        cell_info_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gated messages for each edge.

        x_j: source node features (E, in_channels)
        cell_info_i: target node cell composition (E, cell_info_dim)
        cell_info_j: source node cell composition (E, cell_info_dim)
        """
        # Concatenate source and target cell compositions
        cell_pair = torch.cat([cell_info_j, cell_info_i], dim=-1)  # (E, 8)

        # Compute per-channel gate
        gate = self.gate_net(cell_pair)  # (E, out_channels) ∈ [0, 1]

        # Transform and gate the message
        msg = self.msg_transform(x_j)   # (E, out_channels)
        return msg * gate               # element-wise gating

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update node representations with aggregated gated messages.

        aggr_out: (N, out_channels) — sum of gated messages from neighbors
        x: (N, in_channels) — own features from previous layer
        """
        combined = torch.cat([x, aggr_out], dim=-1)  # (N, in + out)
        return self.update_mlp(combined)              # (N, out_channels)


class CellConditionedGIN(nn.Module):
    """
    GIN feature extractor with cell-conditioned message passing.

    Same structure as GINFeatureExtractor but replaces GINConv with
    CellConditionedConv. Keeps:
      - Dropout before message passing
      - Residual connections
      - PanNET-only pooling via global_add_pool

    Args:
        num_layers: Number of message passing layers (default: 1)
        hidden_dim: Hidden dimension (default: 256)
        dropout: Dropout rate (default: 0.4)
        cell_info_dim: Dimension of cell_information per node (default: 4)
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.4,
        cell_info_dim: int = 4,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList([
            CellConditionedConv(hidden_dim, hidden_dim, cell_info_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        patch_classes: torch.Tensor,
        cell_information: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: cell-gated message passing + PanNET-only pooling.

        Args:
            x: (N, hidden_dim) node features
            edge_index: (2, E) edge list
            batch: (N,) batch assignment
            patch_classes: (N,) tissue class per node
            cell_information: (N, 4) cell composition per node
        """
        for conv in self.convs:
            x_dropped = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x_dropped, edge_index, cell_info=cell_information) + x  # residual

        # Pool only PanNET nodes
        pannet_mask = patch_classes == PANNET_CLASS_ID
        return global_add_pool(x[pannet_mask], batch[pannet_mask])
