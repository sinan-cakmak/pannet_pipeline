"""
STAGE 1a: WSI → H5 feature files.

Extracts patch-level features from whole-slide images using the Trident
Processor API with the VirChow2 foundation model.

What happens (3 sequential Trident tasks):
  1. seg:    GrandQC segments tissue from glass background → GeoJSON contours
  2. coords: Tile tissue regions into 1024×1024 patches at 40x → coordinate lists
  3. feat:   VirChow2 encodes each patch → H5 files with (N, 1280) features

Supports chunked processing for parallel execution across multiple GPUs:
  --chunk 0 --num-chunks 8  → processes slides 0, 8, 16, 24, ...

Usage:
  uv run python -m src.stage1_extraction.extract_features \\
      --wsi-dir "/path/to/PANNET Slides" \\
      --output-dir "/path/to/features_output" \\
      --chunk 0 --num-chunks 8
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.constants import MAGNIFICATION, MIN_TISSUE_RATIO, PATCH_SIZE


def run_trident_pipeline(wsi_dir: str, output_dir: str, gpu: int = 0) -> None:
    """
    Run all 3 Trident tasks (seg → coords → feat) using the Processor API.

    Args:
        wsi_dir: Directory containing WSI .tiff files to process
        output_dir: Directory to save all outputs (segmentation, coords, features)
        gpu: GPU index to use
    """
    from trident import Processor
    from trident.segmentation_models.load import segmentation_model_factory
    from trident.patch_encoder_models.load import encoder_factory

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    # Initialize Trident Processor
    processor = Processor(
        job_dir=output_dir,
        wsi_source=wsi_dir,
        wsi_ext=[".tiff"],
    )

    # Task 1: Tissue segmentation via GrandQC
    print("  [seg] Running GrandQC tissue segmentation...")
    seg_model = segmentation_model_factory("grandqc")
    artifact_remover = segmentation_model_factory("grandqc_artifact")
    processor.run_segmentation_job(
        seg_model,
        seg_mag=seg_model.target_mag,
        holes_are_tissue=True,
        artifact_remover_model=artifact_remover,
        batch_size=32,
        device=device,
    )

    # Task 2: Patch coordinate extraction
    print("  [coords] Extracting patch coordinates at 40x, 1024px...")
    coords_dir = f"{MAGNIFICATION}x_{PATCH_SIZE}px_0px_overlap"
    processor.run_patching_job(
        target_magnification=MAGNIFICATION,
        patch_size=PATCH_SIZE,
        overlap=0,
        saveto=coords_dir,
        min_tissue_proportion=MIN_TISSUE_RATIO,
    )

    # Task 3: VirChow2 feature extraction → H5 files
    print("  [feat] Extracting VirChow2 features...")
    encoder = encoder_factory("virchow2")
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir,
        patch_encoder=encoder,
        device=device,
        saveas="h5",
        batch_limit=32,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Stage 1a: Extract VirChow2 features from WSIs using Trident"
    )
    parser.add_argument("--wsi-dir", required=True, help="Directory containing raw WSI .tiff files")
    parser.add_argument("--output-dir", required=True, help="Directory to save output")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--chunk", type=int, default=0, help="Which chunk to process (0-indexed)")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total number of parallel chunks")
    args = parser.parse_args()

    wsi_dir = Path(args.wsi_dir)
    if not wsi_dir.exists():
        print(f"ERROR: WSI directory not found: {wsi_dir}")
        sys.exit(1)

    # Get all WSI files and select this chunk's subset
    all_wsis = sorted(wsi_dir.glob("*.tiff"))
    my_wsis = all_wsis[args.chunk :: args.num_chunks]

    print(f"Chunk {args.chunk}/{args.num_chunks}: processing {len(my_wsis)}/{len(all_wsis)} WSIs")

    if not my_wsis:
        print("No WSIs assigned to this chunk.")
        return

    if args.num_chunks == 1:
        # Single chunk — process entire directory directly
        run_trident_pipeline(str(wsi_dir), args.output_dir, args.gpu)
    else:
        # Multi-chunk — create a temp directory with symlinks to this chunk's WSIs.
        # Trident's Processor scans its wsi_source directory, so we give it only our subset.
        with tempfile.TemporaryDirectory(prefix=f"trident_chunk{args.chunk}_") as tmp_dir:
            for wsi_path in my_wsis:
                os.symlink(wsi_path, Path(tmp_dir) / wsi_path.name)
            print(f"Created symlink directory with {len(my_wsis)} WSIs")
            run_trident_pipeline(tmp_dir, args.output_dir, args.gpu)

    print(f"Chunk {args.chunk} complete.")


if __name__ == "__main__":
    main()
