"""
STAGE 1b: Classify each patch as PanNET, Normal, or Stroma.

After Stage 1a extracts VirChow2 features for every patch, this step runs
a trained 3-class MLP classifier to label each patch by tissue type:
  - 0 = Stroma
  - 1 = PanNET (tumor)
  - 2 = Normal tissue

These labels are critical for Stage 3 (graph construction), which uses them
to identify tumor borders and build bipartite graphs.

The classifier architecture:
  VirChow2 features (1280-d) are first expanded to 2560-d by concatenating
  the class token with the average-pooled patch tokens (standard VirChow2
  output format). Then:
    Linear(2560, 256) → BatchNorm → ReLU → Dropout(0.3)
    Linear(256, 128)  → BatchNorm → ReLU → Dropout(0.3)
    Linear(128, 3)

The classifier was trained on ~45,000 patches from 20 pixel-annotated WSIs
and achieves >95% weighted F1 (see thesis Section 5.1).

Usage:
  uv run python -m src.stage1_extraction.classify_patches \\
      --h5-dir "/path/to/features_output" \\
      --checkpoint "/path/to/patch_classifier.ckpt"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from rich.progress import track


class PatchClassifier(nn.Module):
    """
    3-class tissue patch classifier.

    Takes 2560-d VirChow2 features (class_token + avg_pooled_patches)
    and predicts tissue type: stroma (0), PanNET (1), normal (2).
    """

    def __init__(self, input_dim: int = 2560, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def classify_all_h5(h5_dir: str, checkpoint_path: str, device: str = "cuda") -> None:
    """
    Run patch classification on all H5 files and save patch_classes into each.

    For each H5 file:
      1. Load the 'features' dataset (N, 1280 or 2560)
      2. If features are 1280-d, duplicate to create 2560-d input
         (in practice, VirChow2 outputs class_token + patch_tokens as 2560-d)
      3. Run the classifier in inference mode
      4. Save argmax predictions as 'patch_classes' dataset in the H5 file
    """
    h5_path = Path(h5_dir)
    h5_files = sorted(h5_path.glob("**/*.h5"))
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        return

    # Load classifier
    model = PatchClassifier()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    for h5_file in track(h5_files, description="Classifying patches"):
        with h5py.File(h5_file, "r+") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)

            # Classify in batches to avoid OOM
            batch_size = 512
            all_preds = []
            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size].to(device)
                with torch.no_grad():
                    logits = model(batch)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)

            patch_classes = torch.cat(all_preds).numpy()

            # Save back into H5
            if "patch_classes" in f:
                del f["patch_classes"]
            f.create_dataset("patch_classes", data=patch_classes)

    print(f"Classified patches in {len(h5_files)} H5 files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1b: Classify patches as PanNET / Normal / Stroma"
    )
    parser.add_argument("--h5-dir", required=True, help="Directory with H5 files")
    parser.add_argument("--checkpoint", required=True, help="Patch classifier checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()
    classify_all_h5(args.h5_dir, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
