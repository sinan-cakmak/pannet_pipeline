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

Output H5 structure per slide:
  features      (N, 1280)  — VirChow2 embeddings
  coords        (N, 2)     — (x, y) top-left pixel coordinates of each patch
  + GeoJSON contour files in {output_dir}/contours_geojson/

Requirements:
  - Trident must be installed (editable local dependency)
  - HF_TOKEN env var must be set for VirChow2 model download
  - Must run on GPU (HPC cluster)

Usage:
  uv run python -m src.stage1_extraction.extract_features \\
      --wsi-dir "/path/to/PANNET Slides" \\
      --output-dir "/path/to/features_output"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.constants import MAGNIFICATION, MIN_TISSUE_RATIO, PATCH_SIZE


def extract_features(wsi_dir: str, output_dir: str) -> None:
    """
    Run Trident's batch feature extraction on all WSIs in a directory.

    This wraps Trident's CLI by constructing the argument list and calling
    its main() function directly. Trident handles:
      - Reading WSI files (OpenSlide)
      - Tissue segmentation via GrandQC
      - Patch extraction at the specified magnification
      - Feature encoding via VirChow2
      - Saving to H5 format
    """
    # Trident expects CLI-style arguments via sys.argv
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
    parser.add_argument(
        "--wsi-dir", required=True,
        help="Directory containing raw WSI .tiff files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save H5 feature files",
    )
    args = parser.parse_args()

    wsi_dir = Path(args.wsi_dir)
    if not wsi_dir.exists():
        print(f"ERROR: WSI directory not found: {wsi_dir}")
        sys.exit(1)

    print(f"Extracting features from: {wsi_dir}")
    print(f"Saving to: {args.output_dir}")
    extract_features(str(wsi_dir), args.output_dir)
    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
