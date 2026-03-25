"""
STAGE 2: AutoEncoder for feature dimensionality reduction.

VirChow2 produces 1280-dimensional feature vectors per patch. Training a GNN
directly on 1280-d features risks overfitting given our small dataset (~73
patients). This autoencoder compresses features to 256 dimensions.

Architecture (symmetric):
  Encoder: 1280 → 768 → 512 → 256  (with RMSNorm, GELU, Dropout)
  Decoder: 256 → 512 → 768 → 1280  (with RMSNorm, GELU, no dropout)

The encoder is trained on ALL patches (no labels used — unsupervised), then
frozen and used as a fixed feature projector during GNN training. This avoids
label leakage and keeps the projection stable across folds.

Loss = MSE(reconstruction, input) + 1e-3 * variance_regularization
  The variance term encourages the latent space to have unit variance per
  dimension, preventing feature collapse.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import AUTOENCODER_DIM, VIRCHOW2_DIM


class AutoEncoder(L.LightningModule):
    """
    Symmetric autoencoder for patch feature compression.

    Encoder: 1280 → 768 → 512 → 256
    Decoder: 256 → 512 → 768 → 1280
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # ---- Encoder ----
        # Each block: Linear → RMSNorm → GELU → Dropout
        # The bottleneck (final layer) has no activation/dropout
        self.encoder = nn.Sequential(
            nn.Linear(VIRCHOW2_DIM, 768),
            nn.RMSNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.RMSNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, AUTOENCODER_DIM),
        )

        # ---- Decoder (mirrors encoder, no dropout) ----
        self.decoder = nn.Sequential(
            nn.Linear(AUTOENCODER_DIM, 512),
            nn.RMSNorm(512),
            nn.GELU(),
            nn.Linear(512, 768),
            nn.RMSNorm(768),
            nn.GELU(),
            nn.Linear(768, VIRCHOW2_DIM),
        )

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (reconstruction, latent) for loss computation.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        reconstruction, latent = self(batch)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, batch)

        # Variance regularization: encourage unit variance in latent space.
        # Without this, the encoder might collapse all features to similar values.
        latent_var = latent.var(dim=0)
        var_loss = F.mse_loss(latent_var, torch.ones_like(latent_var))

        loss = recon_loss + 1e-3 * var_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon", recon_loss)
        self.log("train_var", var_loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        reconstruction, latent = self(batch)
        loss = F.mse_loss(reconstruction, batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
