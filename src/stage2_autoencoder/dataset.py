"""
Dataset for autoencoder training.

Loads all patch features from H5 files into memory as a flat list of
1280-dimensional vectors. Each vector is one training sample.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.constants import VIRCHOW2_DIM


class PatchFeatureDataset(Dataset):
    """
    Flat dataset of 1280-d VirChow2 feature vectors from H5 files.

    Loads all features into memory at init time. For a typical PanNET
    dataset (~200 slides, ~500 patches each), this is ~100K vectors × 1280
    floats ≈ 500 MB — fits comfortably in RAM.
    """

    def __init__(self, h5_files: list[Path]):
        super().__init__()
        all_features = []

        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                features = f["features"][:]
                # VirChow2 may output 2560-d (class_token + pooled_patches).
                # We only use the first 1280 dims for the autoencoder.
                if features.shape[1] > VIRCHOW2_DIM:
                    features = features[:, :VIRCHOW2_DIM]
                all_features.append(features)

        self.features = torch.from_numpy(
            np.concatenate(all_features, axis=0)
        ).float()
        print(f"Loaded {len(self.features)} patch features from {len(h5_files)} H5 files")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]
