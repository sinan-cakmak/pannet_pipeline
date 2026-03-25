"""
STAGE 3 — Bipartite adjacency matrix construction.

The key insight from Nusret's thesis: edges should ONLY connect PanNET
(tumor) patches to NNP (non-neoplastic parenchyma) patches. No tumor-to-tumor
or NNP-to-NNP edges. This forces the GNN to learn from the tumor–NNP
interaction, which is where the infiltration signal lives.

Distance metric: Chebyshev (L-infinity) distance in grid coordinates.
  max(|x1-x2|, |y1-y2|) / PATCH_SIZE ≤ hop_distance
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.constants import PANNET_CLASS_ID, PATCH_SIZE


def build_bipartite_edges(
    patch_locs: np.ndarray,
    patch_classes: np.ndarray,
    hop_distance: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a bipartite edge index between PanNET and non-PanNET patches.

    Two patches are connected if:
      1. One is PanNET and the other is not (bipartite constraint)
      2. Their Chebyshev distance ≤ hop_distance * PATCH_SIZE (spatial proximity)
      3. No self-loops

    Args:
        patch_locs: (N, 2) patch coordinates in pixels
        patch_classes: (N,) class labels
        hop_distance: Maximum grid distance for edge creation
        device: Torch device for computation

    Returns:
        edge_index: (2, E) tensor of undirected edges (both directions)
    """
    N = len(patch_locs)
    locs_t = torch.tensor(patch_locs, dtype=torch.float32, device=device)
    classes_t = torch.tensor(patch_classes, dtype=torch.long, device=device)

    # Compute pairwise Chebyshev (L-infinity) distances
    # diff[i, j] = |loc_i - loc_j| element-wise
    diff = (locs_t[:, None, :] - locs_t[None, :, :]).abs()
    cheb_dist = diff.amax(dim=2)  # (N, N) — max of |dx|, |dy|

    threshold = hop_distance * PATCH_SIZE

    # Spatial adjacency: within distance and not self-loop
    spatial_mask = (cheb_dist > 0) & (cheb_dist <= threshold)

    # Bipartite mask: one node is PanNET, the other is not
    is_pannet = (classes_t == PANNET_CLASS_ID)
    pannet_to_other = is_pannet[:, None] & ~is_pannet[None, :]
    other_to_pannet = ~is_pannet[:, None] & is_pannet[None, :]
    bipartite_mask = pannet_to_other | other_to_pannet

    # Final adjacency: spatial AND bipartite
    adj_mask = spatial_mask & bipartite_mask

    # Extract edge indices
    src, dst = torch.where(adj_mask)
    edge_index = torch.stack([src, dst], dim=0).cpu().long()

    return edge_index


def compute_edge_distances(data: Data) -> torch.Tensor:
    """
    Compute L-infinity distances for each edge, normalized by PATCH_SIZE.

    For adjacent patches (hop=1), distance ≈ 1.0.
    For diagonal neighbors, distance ≈ 1.0 (since L-inf = max(dx, dy)).

    Args:
        data: PyG Data object with pos (N, 2) and edge_index (2, E)

    Returns:
        edge_distances: (E, 1) tensor
    """
    src, dst = data.edge_index
    src_pos = data.pos[src].float()
    dst_pos = data.pos[dst].float()

    # L-infinity distance normalized by patch size
    diff = (src_pos - dst_pos).abs()
    dist = diff.amax(dim=1, keepdim=True) / PATCH_SIZE

    return dist
