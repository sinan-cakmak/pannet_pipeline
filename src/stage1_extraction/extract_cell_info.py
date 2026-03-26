"""
STAGE 1.5: Extract cell counts per patch using HoVer-Net.

For each patch in the WSI, runs HoVer-Net nucleus instance segmentation
to detect and classify individual cells, then counts them by type.

HoVer-Net (trained on PanNuke dataset) classifies nuclei into 5 types:
  1 = Neoplastic (tumor cells)
  2 = Inflammatory (immune cells)
  3 = Connective (stromal cells)
  4 = Dead cells
  5 = Non-neoplastic epithelial

We collapse these into a 4-dim vector per patch:
  [0] = neoplastic count      (type 1)
  [1] = inflammatory count    (type 2)
  [2] = other count           (types 3 + 4 + 5)
  [3] = 0                     (reserved / binary flag)

Requires:
  - tiatoolbox installed (uv pip install -e ../tiatoolbox --no-deps)
  - WSI .tiff files accessible (for reading patch images via OpenSlide)
  - GPU for HoVer-Net inference

Usage:
  uv run python -m src.stage1_extraction.extract_cell_info \\
      --h5-dir "/path/to/features_virchow2" \\
      --wsi-dir "/path/to/PANNET Slides" \\
      --chunk 0 --num-chunks 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.constants import PATCH_SIZE


def load_hovernet(device: str = "cuda") -> torch.nn.Module:
    """
    Load HoVer-Net model pre-trained on PanNuke dataset.

    Uses a workaround from Nusret's code: extract the .model from
    NucleusInstanceSegmentor since its predict() is unreliable, and
    call infer_batch() + postproc() directly.
    """
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

    segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=2,
        num_postproc_workers=2,
        batch_size=4,
    )
    model = segmentor.model.to(device)
    model.eval()
    return model


def load_stain_normalizer(reference_image_path: str | None = None):
    """
    Load Macenko stain normalizer.

    If a reference image is provided, fit the normalizer to it.
    Otherwise return None (skip normalization).
    """
    if reference_image_path is None:
        return None

    from PIL import Image
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer

    normalizer = MacenkoNormalizer()
    ref_img = np.array(Image.open(reference_image_path).convert("RGB"))
    normalizer.fit(ref_img)
    return normalizer


def prep_patch_for_hovernet(img_np: np.ndarray) -> torch.Tensor:
    """
    Prepare a 1024×1024 RGB patch for HoVer-Net inference.

    HoVer-Net expects padded input (48px reflect padding on each side)
    to handle nuclei at patch borders. Input format: (1, H+96, W+96, 3).
    """
    x = torch.from_numpy(img_np)                          # (H, W, 3)
    x = x.permute(2, 0, 1)                                # (3, H, W)
    x = F.pad(x, (48, 48, 48, 48), mode="reflect")        # (3, H+96, W+96)
    x = x.permute(1, 2, 0).unsqueeze(0)                   # (1, H+96, W+96, 3)
    return x


def count_cells_in_patch(
    model: torch.nn.Module,
    img_np: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run HoVer-Net on a single patch and count cells by type.

    Returns a 4-dim array: [neoplastic, inflammatory, other, reserved].

    PanNuke cell types:
      1 = Neoplastic (tumor)
      2 = Inflammatory (immune)
      3 = Connective (stromal)
      4 = Dead
      5 = Non-neoplastic epithelial
    """
    x = prep_patch_for_hovernet(img_np)

    # Run HoVer-Net inference
    # Returns: np_map (nuclei probability), hv (horizontal/vertical gradients), tp (type prediction)
    np_map, hv, tp = model.infer_batch(model, x, device)

    # Remove padding artifacts (48px was added on each side, but output has 2px border effect)
    np_map = np_map[0][2:-2, 2:-2]
    hv = hv[0][2:-2, 2:-2]
    tp = tp[0][2:-2, 2:-2]

    # Post-process: instance segmentation + cell type assignment
    # Returns: (instance_map, instance_info_dict)
    # instance_info_dict: {id: {"centroid": (x,y), "type": int, "contour": [...], ...}}
    _, maps = model.postproc([np_map, hv, tp])

    # Count cells by type
    counts = np.zeros(4, dtype=np.int64)
    for nucleus_info in maps.values():
        cell_type = nucleus_info["type"]
        if cell_type == 1:
            counts[0] += 1  # Neoplastic (tumor)
        elif cell_type == 2:
            counts[1] += 1  # Inflammatory (immune)
        else:
            counts[2] += 1  # Other (connective + dead + epithelial)
    # counts[3] stays 0 (reserved flag)

    return counts


def process_h5_file(
    h5_path: Path,
    wsi_dir: Path,
    model: torch.nn.Module,
    normalizer=None,
    device: str = "cuda",
) -> bool:
    """
    Extract cell counts for all patches in one H5 file.

    Opens the corresponding WSI, reads each patch at its stored coordinates,
    runs HoVer-Net, and saves cell_information back into the H5 file.

    Returns True on success, False on failure.
    """
    from openslide import OpenSlide

    try:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            num_patches = len(coords)

            # Check if already processed
            if "cell_information" in f:
                print(f"  Skipping {h5_path.name} — cell_information already exists")
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
        cell_info = np.zeros((num_patches, 4), dtype=np.int64)

        for i in tqdm(range(num_patches), desc=f"  {h5_path.name}", leave=False):
            x, y = int(coords[i, 0]), int(coords[i, 1])

            # Read patch image from WSI at stored coordinates
            img = wsi.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            img_np = np.array(img)

            # Stain normalization (if enabled)
            if normalizer is not None:
                try:
                    img_np = normalizer.transform(img_np)
                except Exception:
                    pass  # Skip normalization if it fails (e.g., too few tissue pixels)

            # Run HoVer-Net and count cells
            try:
                cell_info[i] = count_cells_in_patch(model, img_np, device)
            except Exception as e:
                # If HoVer-Net fails on this patch, leave as zeros
                print(f"    WARNING: HoVer-Net failed on patch ({x}, {y}): {e}")

        wsi.close()

        # Save back into H5 file
        with h5py.File(h5_path, "r+") as f:
            if "cell_information" in f:
                del f["cell_information"]
            f.create_dataset("cell_information", data=cell_info)

        return True

    except Exception as e:
        print(f"  ERROR processing {h5_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1.5: Extract cell counts per patch using HoVer-Net"
    )
    parser.add_argument("--h5-dir", required=True, help="Directory with H5 feature files")
    parser.add_argument("--wsi-dir", required=True, help="Directory with WSI .tiff files")
    parser.add_argument("--stain-ref", default=None, help="Reference image for Macenko normalization")
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

    # Load HoVer-Net model
    print("Loading HoVer-Net (hovernet_fast-pannuke)...")
    model = load_hovernet(args.device)

    # Load stain normalizer
    normalizer = load_stain_normalizer(args.stain_ref)
    if normalizer:
        print("Macenko stain normalization enabled.")
    else:
        print("Stain normalization disabled (no --stain-ref provided).")

    # Process each H5 file
    success = 0
    failed = 0
    for h5_path in my_h5:
        print(f"Processing: {h5_path.name}")
        if process_h5_file(h5_path, wsi_dir, model, normalizer, args.device):
            success += 1
        else:
            failed += 1

    print(f"\nDone. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
