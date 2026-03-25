"""
STAGE 2 entry point: Train the autoencoder on all patch features.

Splits H5 files 90/10 for train/val (by file, not by patch — prevents
data leakage since patches from the same slide are correlated).

Usage:
  uv run python -m src.stage2_autoencoder.train \\
      --h5-dir "/path/to/features" \\
      --batch-size 4096 \\
      --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.rich_model_summary import RichModelSummary
from torch.utils.data import DataLoader

from src.stage2_autoencoder.dataset import PatchFeatureDataset
from src.stage2_autoencoder.model import AutoEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Train AutoEncoder")
    parser.add_argument("--h5-dir", required=True, help="Directory with H5 feature files")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints/autoencoder")
    args = parser.parse_args()

    L.seed_everything(args.seed)

    # Collect all H5 files
    h5_dir = Path(args.h5_dir)
    h5_files = sorted(h5_dir.glob("**/*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_dir}")

    # Split 90/10 by file (not by individual patches)
    split_idx = int(len(h5_files) * 0.9)
    train_files = h5_files[:split_idx]
    val_files = h5_files[split_idx:]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create datasets and dataloaders
    train_dataset = PatchFeatureDataset(train_files)
    val_dataset = PatchFeatureDataset(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = AutoEncoder()

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.01,
        mode="min",
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb, RichProgressBar(), RichModelSummary()],
        precision="bf16-mixed",
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
