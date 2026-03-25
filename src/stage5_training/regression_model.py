"""
STAGE 5 — Infiltration pattern scoring model (LightningModule).

Combines three components:
  1. Projector: Frozen autoencoder encoder (1280 → 256)
  2. Feature extractor: GIN with bipartite message passing
  3. Regression head: MLP that outputs a single infiltration grade

The model predicts grades on a 0-4 scale (corresponding to clinical grades
1-5). During patient-level evaluation, predictions are shifted back to 1-5
and summed across 3 slides to compute IPS.

Loss function: Huber loss (δ=2)
  - Like MSE for small errors, like MAE for large errors
  - More robust to noisy labels than pure MSE
  - The thesis showed Huber significantly outperforms cross-entropy
    for this ordinal regression task (Table 5.4)
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError

from src.constants import AUTOENCODER_DIM


class InfiltrationModel(L.LightningModule):
    """
    End-to-end model: frozen projector → GNN → regression head.

    Args:
        projector: Frozen autoencoder encoder (1280 → 256)
        feature_extractor: GIN feature extractor
        lr: Learning rate (default: 1e-4)
        weight_decay: L2 regularization (default: 1e-3)
    """

    def __init__(
        self,
        projector: nn.Module,
        feature_extractor: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["projector", "feature_extractor"])

        # Frozen projector: compresses 1280-d VirChow2 → 256-d
        self.projector = projector
        for param in self.projector.parameters():
            param.requires_grad = False

        # Input normalization after projection
        self.input_norm = nn.RMSNorm(AUTOENCODER_DIM)

        # GNN feature extractor (trainable)
        self.feature_extractor = feature_extractor

        # Regression head: 256 → 1 (single grade prediction)
        self.head = nn.Sequential(
            nn.RMSNorm(AUTOENCODER_DIM),
            nn.Linear(AUTOENCODER_DIM, AUTOENCODER_DIM // 2),  # 256 → 128
            nn.RMSNorm(AUTOENCODER_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(AUTOENCODER_DIM // 2, 1),
        )

        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, data) -> torch.Tensor:
        """
        Full forward pass: project → normalize → GNN → head.

        Args:
            data: PyG Data batch with x, edge_index, batch, patch_classes

        Returns:
            (B,) predictions on the 0-4 scale
        """
        # Project features: 1280 → 256 (frozen)
        x = self.projector(data.x[:, :1280])
        x = self.input_norm(x)

        # GNN message passing + PanNET-only pooling → (B, 256)
        h = self.feature_extractor(x, data.edge_index, data.batch, data.patch_classes)

        # Regression → single scalar per graph
        pred = self.head(h).squeeze(-1)  # (B,)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch.y.float()
        loss = F.huber_loss(pred, target, delta=2.0)

        self.train_mae(pred, target)
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log("train_mae", self.train_mae, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch.y.float()
        loss = F.huber_loss(pred, target, delta=2.0)

        self.val_mae(pred, target)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log("val_mae", self.val_mae, batch_size=batch.num_graphs)
        return loss

    def predict_step(self, batch, batch_idx):
        pred = self(batch)
        # Shift from 0-4 internal scale to 1-5 clinical scale
        return {
            "predictions": pred + 1.0,
            "targets": batch.y.float() + 1.0,
            "filenames": batch.filename,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
