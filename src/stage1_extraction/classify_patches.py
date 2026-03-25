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


def classify_all_h5(
    h5_dir: str, checkpoint_path: str, device: str = "cuda",
    chunk: int = 0, num_chunks: int = 1,
) -> None:
    """
    Run patch classification on H5 files and save patch_classes into each.

    Supports chunked processing for parallel execution:
      chunk=0, num_chunks=8 → process files 0, 8, 16, ...
    """
    h5_path = Path(h5_dir)
    all_h5_files = sorted(h5_path.glob("**/*.h5"))
    h5_files = all_h5_files[chunk::num_chunks]
    print(f"Chunk {chunk}/{num_chunks}: classifying {len(h5_files)}/{len(all_h5_files)} H5 files")
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
    parser.add_argument("--chunk", type=int, default=0, help="Which chunk (0-indexed)")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total parallel chunks")
    args = parser.parse_args()
    classify_all_h5(args.h5_dir, args.checkpoint, args.device, args.chunk, args.num_chunks)


if __name__ == "__main__":
    main()
