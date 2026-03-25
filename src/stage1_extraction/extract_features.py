"""
STAGE 1a: WSI → H5 feature files.

Extracts patch-level features from whole-slide images using the Trident
pipeline with the VirChow2 foundation model.

What happens here:
  1. Each WSI (.tiff) is opened at 40x magnification.
  2. GrandQC segments the tissue from the glass background.
  3. Tissue regions are tiled into 1024×1024 non-overlapping patches.
  4. Patches with <60% tissue are discarded.
  5. VirChow2 encodes each patch into a 1280-dimensional feature vector.
  6. Features, coordinates, and metadata are saved as an H5 file per slide.

Supports chunked processing for parallel execution across multiple GPUs:
  --chunk 0 --num-chunks 8  → process slides 0, 8, 16, 24, ...
  --chunk 1 --num-chunks 8  → process slides 1, 9, 17, 25, ...

Output H5 structure per slide:
  features      (N, 1280)  — VirChow2 embeddings
  coords        (N, 2)     — (x, y) top-left pixel coordinates of each patch

Usage:
  uv run python -m src.stage1_extraction.extract_features \\
      --wsi-dir "/path/to/PANNET Slides" \\
      --output-dir "/path/to/features_output" \\
      --chunk 0 --num-chunks 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from src.constants import MAGNIFICATION, MIN_TISSUE_RATIO, PATCH_SIZE


def extract_features(wsi_dir: str, output_dir: str) -> None:
    """
    Run Trident's batch feature extraction on all WSIs in a directory.
    """
    from trident.run_batch_of_slides import main as trident_main

    sys.argv = [
        "run_batch_of_slides",
        "--wsi_dir", wsi_dir,
        "--save_dir", output_dir,
        "--patch_encoder", "virchow2",
        "--segmenter", "grandqc",
        "--mag", str(MAGNIFICATION),
        "--patch_size", str(PATCH_SIZE),
        "--min_tissue_proportion", str(MIN_TISSUE_RATIO),
        "--batch_size", "32",
        "--remove_artifacts",
    ]
    trident_main()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Stage 1a: Extract VirChow2 features from WSIs using Trident"
    )
    parser.add_argument("--wsi-dir", required=True, help="Directory containing raw WSI .tiff files")
    parser.add_argument("--output-dir", required=True, help="Directory to save H5 feature files")
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
        extract_features(str(wsi_dir), args.output_dir)
    else:
        # Multi-chunk — create a temp directory with symlinks to this chunk's WSIs.
        # Trident processes everything in its input directory, so we give it only our subset.
        with tempfile.TemporaryDirectory(prefix=f"trident_chunk{args.chunk}_") as tmp_dir:
            for wsi_path in my_wsis:
                os.symlink(wsi_path, Path(tmp_dir) / wsi_path.name)
            print(f"Created symlink directory with {len(my_wsis)} WSIs: {tmp_dir}")
            extract_features(tmp_dir, args.output_dir)

    print(f"Chunk {args.chunk} complete.")


if __name__ == "__main__":
    main()
