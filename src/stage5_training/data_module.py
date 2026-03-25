"""
STAGE 5 — Data module for cross-validated training.

Handles the 4-fold cross-validation protocol:
  - s1, s2, s3, s4 rotate as test/val (val = test+1 mod 4)
  - s5 is ALWAYS added to training (incomplete patients, never evaluated)

Uses WeightedRandomSampler during training to balance grade classes.
Without this, the model would be biased toward the majority grade.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.stage5_training.dataset import GraphDataset
from src.utils import load_fold_config


class PanNETDataModule(L.LightningDataModule):
    """
    4-fold cross-validation data module for PanNET graph datasets.

    Args:
        graph_dir: Directory with .pkl graph files
        fold_config_path: YAML file with s1-s5 patient ID lists
        test_fold: Which fold (0-3) to use as test set
        batch_size: Training batch size (default: 8)
        num_workers: DataLoader workers (default: 4)
    """

    def __init__(
        self,
        graph_dir: str | Path,
        fold_config_path: str | Path = "data/fold_information.yaml",
        test_fold: int = 0,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.graph_dir = graph_dir
        self.fold_config_path = fold_config_path
        self.test_fold = test_fold
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """
        Split patient IDs into train/val/test based on fold configuration.

        Fold layout (4-fold):
          test  = s[test_fold]
          val   = s[(test_fold + 1) % 4]
          train = remaining s_i + s5 (always training)
        """
        config = load_fold_config(self.fold_config_path)
        splits = [
            config["s1_keys"],
            config["s2_keys"],
            config["s3_keys"],
            config["s4_keys"],
        ]
        s5 = config["s5_keys"]

        # Determine fold assignments
        test_ids = splits[self.test_fold]
        val_idx = (self.test_fold + 1) % 4
        val_ids = splits[val_idx]

        # Train = remaining folds + s5 (always training data)
        train_ids = []
        for i in range(4):
            if i != self.test_fold and i != val_idx:
                train_ids.extend(splits[i])
        train_ids.extend(s5)

        print(f"Fold {self.test_fold}: test={len(test_ids)} patients, "
              f"val={len(val_ids)} patients, train={len(train_ids)} patients")

        # Create datasets filtered by patient IDs
        self.train_dataset = GraphDataset(self.graph_dir, case_ids=train_ids)
        self.val_dataset = GraphDataset(self.graph_dir, case_ids=val_ids)
        self.test_dataset = GraphDataset(self.graph_dir, case_ids=test_ids)

        print(f"  Train graphs: {len(self.train_dataset)}, "
              f"Val graphs: {len(self.val_dataset)}, "
              f"Test graphs: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        # WeightedRandomSampler: oversample rare grades so the model sees
        # balanced batches. Weight per sample = 1 / count_of_its_class.
        labels = self.train_dataset.grade_labels
        class_counts = Counter(labels.tolist())
        sample_weights = [1.0 / class_counts[int(l)] for l in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
