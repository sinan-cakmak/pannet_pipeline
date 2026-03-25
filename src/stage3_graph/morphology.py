"""
STAGE 3 — Morphological operations for graph construction.

Two key operations:

1. fill_pannet_holes(): If there's a gap inside a PanNET region (a missing
   patch due to artifact, fold, or misclassification), we fill it so the graph
   stays connected. Uses scipy's binary_fill_holes on a 2D grid.

2. find_border_pannets(): Identifies PanNET patches at the tumor–NNP interface.
   A PanNET patch is a "border" patch if at least one of its 8 immediate
   neighbors is non-PanNET or doesn't exist. These border patches are where
   the bipartite graph is anchored.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from src.constants import NEIGHBOR_CONNECTIVITY, PANNET_CLASS_ID, PATCH_SIZE


def fill_pannet_holes(
    slide_width: int,
    slide_height: int,
    patch_locs: np.ndarray,
    patch_classes: np.ndarray,
    patch_feats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill enclosed gaps within PanNET (tumor) regions.

    Sometimes patches inside a tumor region are missing (tissue fold, staining
    artifact, or classification error). This leaves holes that break graph
    connectivity. We fill them by:

    1. Creating a 2D binary grid where PanNET patches = 1
    2. Running scipy.ndimage.binary_fill_holes (fills enclosed 0s)
    3. Adding new synthetic nodes at filled positions (with zero features)

    Args:
        slide_width: WSI width in pixels
        slide_height: WSI height in pixels
        patch_locs: (N, 2) patch coordinates
        patch_classes: (N,) patch class labels
        patch_feats: (N, D) patch features

    Returns:
        Updated (patch_locs, patch_classes, patch_feats) with holes filled.
        New patches have zero features and PANNET_CLASS_ID.
    """
    # Build a 2D grid. Value = class ID, -1 = empty position.
    grid_h = slide_height // PATCH_SIZE + 1
    grid_w = slide_width // PATCH_SIZE + 1
    grid = np.full((grid_h, grid_w), -1, dtype=np.int64)

    # Place patches on grid: grid[row, col] = class
    rows = patch_locs[:, 1] // PATCH_SIZE
    cols = patch_locs[:, 0] // PATCH_SIZE
    grid[rows, cols] = patch_classes

    # Create binary mask of PanNET patches and fill holes
    pannet_mask = grid == PANNET_CLASS_ID
    filled_mask = ndimage.binary_fill_holes(pannet_mask)

    # Find newly filled positions (were empty, now inside tumor region)
    holes = filled_mask & (grid == -1)
    hole_rows, hole_cols = np.where(holes)

    if len(hole_rows) == 0:
        return patch_locs, patch_classes, patch_feats

    # Create new patches at filled positions with zero features
    new_locs = np.stack([hole_cols * PATCH_SIZE, hole_rows * PATCH_SIZE], axis=1)
    new_classes = np.full(len(new_locs), PANNET_CLASS_ID, dtype=np.int64)
    new_feats = np.zeros((len(new_locs), patch_feats.shape[1]), dtype=patch_feats.dtype)

    # Append to existing data
    patch_locs = np.concatenate([patch_locs, new_locs], axis=0)
    patch_classes = np.concatenate([patch_classes, new_classes], axis=0)
    patch_feats = np.concatenate([patch_feats, new_feats], axis=0)

    return patch_locs, patch_classes, patch_feats


def find_border_pannets(
    patch_locs: np.ndarray,
    patch_classes: np.ndarray,
    include_neighbour: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find PanNET patches at the tumor–NNP boundary + optional NNP neighbors.

    A PanNET patch is a "border" patch if at least one of its 8 immediate
    grid neighbors is either non-PanNET or doesn't exist (edge of tissue).

    Optionally, also include non-PanNET patches within `include_neighbour`
    hops of the border (these become the NNP side of the bipartite graph).

    Args:
        patch_locs: (N, 2) patch coordinates in pixels
        patch_classes: (N,) class labels
        include_neighbour: How many hops of NNP neighbors to include (0 = border only)
        device: Torch device for distance computation

    Returns:
        (border_indices, border_distances, nnp_indices) where:
        - border_indices: indices of PanNET border patches in the input arrays
        - border_distances: (N,) distance of each patch to the border (0 = on border)
        - nnp_indices: indices of NNP patches to include (within include_neighbour hops)
    """
    locs_t = torch.tensor(patch_locs, dtype=torch.float32, device=device)
    classes_t = torch.tensor(patch_classes, dtype=torch.long, device=device)

    pannet_mask = classes_t == PANNET_CLASS_ID
    pannet_indices = torch.where(pannet_mask)[0]
    other_indices = torch.where(~pannet_mask)[0]

    if len(pannet_indices) == 0 or len(other_indices) == 0:
        return np.array([], dtype=np.int64), np.zeros(len(patch_locs)), np.array([], dtype=np.int64)

    pannet_locs = locs_t[pannet_indices]
    other_locs = locs_t[other_indices]

    # 8 directional offsets (up, down, left, right, 4 diagonals)
    offsets = torch.tensor(
        [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]],
        dtype=torch.float32, device=device,
    ) * PATCH_SIZE

    # For each PanNET patch, check if any 8-neighbor is non-PanNET or missing
    # Candidate neighbor positions: (num_pannet, 8, 2)
    candidates = pannet_locs[:, None, :] + offsets[None, :, :]

    # Check if any candidate matches a non-PanNET patch position
    # Using broadcasting: (num_pannet, 8, 1, 2) vs (1, 1, num_other, 2)
    # This can be memory-intensive for large slides, so process in chunks
    is_border = torch.zeros(len(pannet_indices), dtype=torch.bool, device=device)

    # Build a set of all patch locations for quick lookup
    all_locs_set = set(map(tuple, patch_locs.tolist()))
    other_locs_set = set(map(tuple, other_locs.cpu().numpy().tolist()))

    for i, ploc in enumerate(pannet_locs):
        for offset in offsets:
            neighbor = (ploc + offset).tolist()
            neighbor_tuple = (int(neighbor[0]), int(neighbor[1]))
            # Border if neighbor is non-PanNET OR doesn't exist at all
            if neighbor_tuple in other_locs_set or neighbor_tuple not in all_locs_set:
                is_border[i] = True
                break

    border_pannet_indices = pannet_indices[is_border].cpu().numpy()

    # Compute border distances for all patches (0 = border, higher = farther)
    border_distances = np.full(len(patch_locs), -1, dtype=np.int64)
    border_distances[border_pannet_indices] = 0

    # Find NNP patches within include_neighbour hops of border PanNET patches
    nnp_indices = np.array([], dtype=np.int64)
    if include_neighbour > 0 and len(border_pannet_indices) > 0:
        border_locs = locs_t[border_pannet_indices]

        # Chebyshev distance from each non-PanNET patch to nearest border patch
        # other_locs: (M, 2), border_locs: (K, 2)
        diff = other_locs[:, None, :] - border_locs[None, :, :]  # (M, K, 2)
        cheb_dist = (diff.abs() / PATCH_SIZE).amax(dim=2)  # (M, K)
        min_dist = cheb_dist.amin(dim=1)  # (M,)

        within_range = min_dist <= include_neighbour
        nnp_local_indices = torch.where(within_range)[0]
        nnp_indices = other_indices[nnp_local_indices].cpu().numpy()

        # Set border distances for NNP patches
        min_dist_np = min_dist[nnp_local_indices].cpu().numpy().astype(np.int64)
        border_distances[nnp_indices] = min_dist_np

    return border_pannet_indices, border_distances, nnp_indices
