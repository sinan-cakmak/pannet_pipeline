"""
STAGE 5 — Graph dataset for loading pre-built .pkl files.

Each .pkl file contains a PyG Data object representing one tissue region
from one WSI slide. The dataset loads them and provides grade labels for
the WeightedRandomSampler.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Dataset

from src.utils import parse_filename


class GraphDataset(Dataset):
    """
    Dataset of pre-built bipartite graphs stored as .pkl files.

    Filters files by a list of allowed case IDs (for train/val/test splitting).

    Args:
        graph_dir: Directory containing .pkl graph files
        case_ids: List of case IDs to include (for fold-based splitting)
    """

    def __init__(self, graph_dir: str | Path, case_ids: list[int] | None = None):
        super().__init__()
        graph_dir = Path(graph_dir)
        all_files = sorted(graph_dir.glob("*.pkl"))

        # Filter to specified case IDs if provided
        if case_ids is not None:
            case_set = set(case_ids)
            self.files = []
            for f in all_files:
                try:
                    case_id, _ = parse_filename(f.name)
                    if case_id in case_set:
                        self.files.append(f)
                except ValueError:
                    continue
        else:
            self.files = all_files

        # Pre-load grade labels for WeightedRandomSampler
        self._grade_labels = []
        for f in self.files:
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            self._grade_labels.append(data.y.item())
        self._grade_labels = np.array(self._grade_labels)

    @property
    def grade_labels(self) -> np.ndarray:
        """Grade labels (0-4) for all graphs. Used by WeightedRandomSampler."""
        return self._grade_labels

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int):
        with open(self.files[idx], "rb") as f:
            data = pickle.load(f)

        # Ensure correct dtypes
        data.x = data.x.float()
        data.y = data.y.long() if data.y.dim() == 0 else data.y.long().squeeze()
        if hasattr(data, "patch_classes"):
            data.patch_classes = torch.as_tensor(data.patch_classes).long()
        if hasattr(data, "pos"):
            data.pos = data.pos.float()

        return data
