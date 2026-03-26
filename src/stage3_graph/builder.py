"""
STAGE 3 — Graph construction pipeline.

Orchestrates the full graph construction for a single tissue region:
  1. Fill morphological holes in PanNET regions
  2. Detect tumor border patches (PanNET patches adjacent to NNP)
  3. Include nearby NNP patches within border_distance hops
  4. Build bipartite adjacency (PanNET ↔ NNP edges only)
  5. Filter connected components (remove small disconnected fragments)
  6. Project features through frozen autoencoder encoder (1280 → 256)
  7. Package as a PyG Data object

This is the core of what makes the "context-aware" approach work: by
restricting the graph to the tumor–NNP interface with bipartite edges,
we force the GNN to learn from the interaction zone rather than noise.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.constants import NEIGHBOR_CONNECTIVITY, PANNET_CLASS_ID, PATCH_SIZE
from src.stage3_graph.adjacency import build_bipartite_edges, compute_edge_distances
from src.stage3_graph.morphology import fill_pannet_holes, find_border_pannets


def build_graph(
    patch_locs: np.ndarray,
    patch_classes: np.ndarray,
    patch_feats: np.ndarray,
    slide_width: int,
    slide_height: int,
    grade_label: int,
    filename: str,
    hop_distance: int = 3,
    border_distance: int = 3,
    encoder: nn.Module | None = None,
    device: str = "cpu",
    cell_information: np.ndarray | None = None,
) -> Data | None:
    """
    Build a bipartite graph for one tissue region of one WSI.

    Args:
        patch_locs: (N, 2) patch (x, y) coordinates in pixels
        patch_classes: (N,) tissue class labels (0=stroma, 1=PanNET, 2=normal)
        patch_feats: (N, D) feature vectors (1280-d from VirChow2)
        slide_width: WSI width in pixels
        slide_height: WSI height in pixels
        grade_label: Infiltration grade (0-4, internally; 1-5 in clinical scale)
        filename: Source WSI filename
        hop_distance: Max Chebyshev distance for edge creation
        border_distance: How many hops of NNP neighbors to include
        encoder: Frozen autoencoder encoder (1280 → 256). If None, raw features used.
        device: Torch device for computation
        cell_information: (N, 4) cell counts per patch, or None if not available

    Returns:
        PyG Data object, or None if the tissue has no valid PanNET border patches.
    """
    # ---- Step 1: Skip if no PanNET patches ----
    if not np.any(patch_classes == PANNET_CLASS_ID):
        return None

    # ---- Step 2: Fill holes in PanNET regions ----
    # Gaps inside tumor regions break graph connectivity. We fill them with
    # synthetic zero-feature nodes so the graph stays connected.
    patch_locs, patch_classes, patch_feats = fill_pannet_holes(
        slide_width, slide_height, patch_locs, patch_classes, patch_feats,
    )
    # Extend cell_information to match (new hole-fill patches get zero counts)
    if cell_information is not None:
        n_new = len(patch_locs) - len(cell_information)
        if n_new > 0:
            cell_information = np.concatenate([
                cell_information,
                np.zeros((n_new, cell_information.shape[1]), dtype=cell_information.dtype),
            ])

    # ---- Step 3: Find border patches + nearby NNP ----
    border_indices, border_distances_arr, nnp_indices = find_border_pannets(
        patch_locs, patch_classes, include_neighbour=border_distance, device=device,
    )

    if len(border_indices) == 0:
        return None

    # ---- Step 4: Keep only border-relevant patches ----
    keep_indices = np.union1d(border_indices, nnp_indices)

    # Remove zero-feature filler patches (from hole filling)
    nonzero_mask = np.any(patch_feats[keep_indices] != 0, axis=1)
    keep_indices = keep_indices[nonzero_mask]

    if len(keep_indices) < 2:
        return None

    # Subset all arrays to kept patches
    sub_locs = patch_locs[keep_indices]
    sub_classes = patch_classes[keep_indices]
    sub_feats = patch_feats[keep_indices]
    sub_border_dist = border_distances_arr[keep_indices]
    sub_cell_info = cell_information[keep_indices] if cell_information is not None else None

    # ---- Step 5: Build bipartite edges ----
    edge_index = build_bipartite_edges(sub_locs, sub_classes, hop_distance, device)

    if edge_index.shape[1] == 0:
        return None

    # ---- Step 6: Filter connected components ----
    min_size = sum(NEIGHBOR_CONNECTIVITY * i for i in range(1, hop_distance + 1))
    keep_nodes = _filter_components(edge_index, len(sub_locs), min_size)

    if len(keep_nodes) < 2:
        return None

    # Reindex to keep_nodes
    sub_locs, sub_classes, sub_feats, sub_border_dist, edge_index = _reindex(
        sub_locs, sub_classes, sub_feats, sub_border_dist, edge_index, keep_nodes,
    )
    if sub_cell_info is not None:
        sub_cell_info = sub_cell_info[keep_nodes]

    # ---- Step 7: Project features through encoder ----
    features_t = torch.tensor(sub_feats, dtype=torch.float32)
    if encoder is not None:
        with torch.no_grad():
            features_t = encoder(features_t.to(device)).cpu()

    # ---- Step 8: Build PyG Data object ----
    data = Data(
        x=features_t,
        edge_index=edge_index,
        y=torch.tensor(grade_label, dtype=torch.long),
        pos=torch.tensor(sub_locs, dtype=torch.long),
        patch_classes=torch.tensor(sub_classes, dtype=torch.long),
        border_distances=torch.tensor(sub_border_dist, dtype=torch.float32).unsqueeze(-1),
        filename=filename,
        slide_width=slide_width,
        slide_height=slide_height,
    )

    # Attach cell_information if available
    if sub_cell_info is not None:
        data.cell_information = torch.tensor(sub_cell_info, dtype=torch.float32)

    # Compute and attach edge distances
    data.edge_distances = compute_edge_distances(data)

    return data


def _filter_components(
    edge_index: torch.Tensor,
    num_nodes: int,
    min_size: int,
) -> np.ndarray:
    """
    Keep only nodes belonging to connected components ≥ min_size.

    Uses NetworkX for connected component detection.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)

    keep = set()
    for component in nx.connected_components(G):
        if len(component) >= min_size:
            keep.update(component)

    return np.sort(np.array(list(keep)))


def _reindex(
    locs: np.ndarray,
    classes: np.ndarray,
    feats: np.ndarray,
    border_dist: np.ndarray,
    edge_index: torch.Tensor,
    keep_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Subset arrays to keep_nodes and remap edge_index accordingly.
    """
    # Create old→new index mapping
    mapping = {old: new for new, old in enumerate(keep_nodes)}

    # Subset arrays
    locs = locs[keep_nodes]
    classes = classes[keep_nodes]
    feats = feats[keep_nodes]
    border_dist = border_dist[keep_nodes]

    # Remap edges (only keep edges where both endpoints are in keep_nodes)
    keep_set = set(keep_nodes.tolist())
    src, dst = edge_index
    mask = torch.tensor(
        [(s.item() in keep_set and d.item() in keep_set) for s, d in zip(src, dst)]
    )
    new_src = torch.tensor([mapping[s.item()] for s, m in zip(src, mask) if m])
    new_dst = torch.tensor([mapping[d.item()] for d, m in zip(dst, mask) if m])
    edge_index = torch.stack([new_src, new_dst], dim=0)

    return locs, classes, feats, border_dist, edge_index
