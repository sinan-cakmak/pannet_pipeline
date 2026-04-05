"""
STAGE 1.5g: Extract virtual protein counts per patch using GigaTIME.

For each patch in the WSI, runs GigaTIME (UNet++) to predict 23-channel
virtual multiplex immunofluorescence masks, then counts positive pixels
per functional channel.

GigaTIME predicts 23 channels; we exclude TRITC (idx=1) and Cy5 (idx=2)
as background, keeping 21 functional protein channels:
  DAPI, PD-1, CD14, CD4, T-bet, CD34, CD68, CD16, CD11c, CD138, CD20,
  CD3, CD8, PD-L1, CK, Ki67, Tryptase, Actin-D, Caspase3-D, PHH3-B,
  Transgelin

Each 1024×1024 patch is split into 16 non-overlapping 256×256 tiles
(GigaTIME's native resolution), pixel counts are summed across tiles.

Requires:
  - HF_TOKEN env var for downloading model weights from HuggingFace
  - WSI .tiff files accessible (for reading patch images via OpenSlide)
  - GPU recommended (CPU works but slow)

Usage:
  uv run python -m src.stage1_extraction.extract_gigatime \\
      --h5-dir "/path/to/features_virchow2" \\
      --wsi-dir "/path/to/PANNET Slides" \\
      --chunk 0 --num-chunks 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.constants import GIGATIME_FUNCTIONAL_INDICES, PATCH_SIZE

# ImageNet normalization (same as GigaTIME training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TILE_SIZE = 256  # GigaTIME's native input resolution
TILES_PER_SIDE = PATCH_SIZE // TILE_SIZE  # 1024 / 256 = 4
TILES_PER_PATCH = TILES_PER_SIDE ** 2     # 16


def load_gigatime(device: str = "cuda") -> torch.nn.Module:
    """
    Load GigaTIME model with pre-trained weights from HuggingFace.

    Requires HF_TOKEN environment variable for model access.
    Weights are cached in ~/.cache/huggingface after first download.
    """
    from huggingface_hub import snapshot_download

    from src.stage1_extraction.gigatime_arch import GigaTIME

    model = GigaTIME(num_classes=23, input_channels=3)

    # Download weights from HuggingFace
    local_dir = snapshot_download(repo_id="prov-gigatime/GigaTIME")
    weights_path = os.path.join(local_dir, "model.pth")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def extract_protein_counts(
    model: torch.nn.Module,
    img_np: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run GigaTIME on a 1024×1024 patch and count protein-positive pixels.

    Splits the patch into 16 non-overlapping 256×256 tiles, runs GigaTIME
    on all tiles as a batch, thresholds at 0.5, and sums positive pixels
    across tiles for each of the 21 functional channels.

    Args:
        model: GigaTIME model (eval mode, on device)
        img_np: (1024, 1024, 3) uint8 RGB image from OpenSlide
        device: torch device string

    Returns:
        (21,) int64 array of protein-positive pixel counts per channel
    """
    # Normalize: uint8 → float32 with ImageNet stats
    img_float = img_np.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # Tile into 4×4 grid of 256×256
    # Reshape: (1024, 1024, 3) → (4, 256, 4, 256, 3) → (4, 4, 256, 256, 3) → (16, 256, 256, 3)
    tiles = img_norm.reshape(
        TILES_PER_SIDE, TILE_SIZE, TILES_PER_SIDE, TILE_SIZE, 3
    ).transpose(0, 2, 1, 3, 4).reshape(TILES_PER_PATCH, TILE_SIZE, TILE_SIZE, 3)

    # HWC → CHW and to tensor: (16, 3, 256, 256)
    tiles_tensor = torch.from_numpy(tiles.transpose(0, 3, 1, 2)).to(device)

    # Batch inference
    with torch.no_grad():
        logits = model(tiles_tensor)  # (16, 23, 256, 256)

    # Sigmoid → threshold → binary masks
    masks = (torch.sigmoid(logits) > 0.5).float()

    # Select functional channels (exclude TRITC=1, Cy5=2)
    masks = masks[:, GIGATIME_FUNCTIONAL_INDICES, :, :]  # (16, 21, 256, 256)

    # Sum positive pixels across tiles and spatial dims → (21,)
    counts = masks.sum(dim=(0, 2, 3)).cpu().numpy().astype(np.int64)

    return counts


def process_h5_file(
    h5_path: Path,
    wsi_dir: Path,
    model: torch.nn.Module,
    device: str = "cuda",
) -> bool:
    """
    Extract GigaTIME protein counts for all patches in one H5 file.

    Opens the corresponding WSI, reads each patch at its stored coordinates,
    runs GigaTIME, and saves gigatime_features back into the H5 file.

    Returns True on success, False on failure.
    """
    from openslide import OpenSlide

    try:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            num_patches = len(coords)

            # Check if already processed
            if "gigatime_features" in f:
                print(f"  Skipping {h5_path.name} — gigatime_features already exists")
                return True

        # Find the corresponding WSI file
        # H5 name: "#1-1 7817B8509.h5" → WSI: "#1-1 7817B8509.tiff"
        wsi_name = h5_path.stem + ".tiff"
        wsi_path = wsi_dir / wsi_name
        if not wsi_path.exists():
            print(f"  WARNING: WSI not found: {wsi_path}")
            return False

        # Open WSI
        wsi = OpenSlide(str(wsi_path))

        # Process each patch
        protein_counts = np.zeros((num_patches, len(GIGATIME_FUNCTIONAL_INDICES)), dtype=np.int64)

        for i in tqdm(range(num_patches), desc=f"  {h5_path.name}", leave=False):
            x, y = int(coords[i, 0]), int(coords[i, 1])

            # Read patch image from WSI at stored coordinates
            img = wsi.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            img_np = np.array(img)

            # Run GigaTIME and count protein-positive pixels
            try:
                protein_counts[i] = extract_protein_counts(model, img_np, device)
            except Exception as e:
                # If GigaTIME fails on this patch, leave as zeros
                print(f"    WARNING: GigaTIME failed on patch ({x}, {y}): {e}")

        wsi.close()

        # Save back into H5 file
        with h5py.File(h5_path, "r+") as f:
            if "gigatime_features" in f:
                del f["gigatime_features"]
            f.create_dataset("gigatime_features", data=protein_counts)

        return True

    except Exception as e:
        print(f"  ERROR processing {h5_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1.5g: Extract virtual protein counts per patch using GigaTIME"
    )
    parser.add_argument("--h5-dir", required=True, help="Directory with H5 feature files")
    parser.add_argument("--wsi-dir", required=True, help="Directory with WSI .tiff files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    wsi_dir = Path(args.wsi_dir)

    # Get this chunk's H5 files
    all_h5 = sorted(h5_dir.glob("*.h5"))
    my_h5 = all_h5[args.chunk :: args.num_chunks]
    print(f"Chunk {args.chunk}/{args.num_chunks}: processing {len(my_h5)}/{len(all_h5)} H5 files")

    if not my_h5:
        print("No H5 files assigned to this chunk.")
        return

    # Load GigaTIME model
    print("Loading GigaTIME model from HuggingFace...")
    model = load_gigatime(args.device)
    print("GigaTIME model loaded.")

    # Process each H5 file
    success = 0
    failed = 0
    for h5_path in my_h5:
        print(f"Processing: {h5_path.name}")
        if process_h5_file(h5_path, wsi_dir, model, args.device):
            success += 1
        else:
            failed += 1

    print(f"\nDone. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
