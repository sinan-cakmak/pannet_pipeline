"""
STAGE 1c: Assign tissue IDs to patches.

A single WSI may contain multiple disconnected tissue pieces on the glass
slide. This step groups patches into connected tissue regions by:

  1. Reading GeoJSON contour files produced by GrandQC (via Trident).
  2. Merging nearby polygons (buffer → union → unbuffer) to bridge small gaps.
  3. Filtering out tiny tissue fragments (<30% of the largest region's area).
  4. Spatially joining each patch coordinate to its enclosing tissue polygon.

Each patch gets a tissue_id (0, 1, 2, ...) indicating which piece of tissue
it belongs to. In Stage 3, only the largest tissue (tissue_id=0) is used.

Requirements:
  - GeoJSON contour files from Trident/GrandQC
  - geopandas must be installed for spatial operations

Usage:
  uv run python -m src.stage1_extraction.assign_tissue_ids \\
      --h5-dir "/path/to/features_output"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from rich.progress import track
from shapely.ops import unary_union


def merge_polygons(geojson_path: str | Path) -> gpd.GeoDataFrame:
    """
    Read tissue contours from a GeoJSON file and merge nearby polygons.

    Steps:
      1. Buffer each polygon by 100 pixels (bridges small gaps between fragments)
      2. Union all overlapping polygons into single shapes
      3. Explode multi-polygons into individual polygons
      4. Remove small fragments (<30% of the largest polygon's area)
      5. Un-buffer by -100 pixels (restore original boundary)

    Returns:
        GeoDataFrame with one row per merged tissue region, sorted by area.
    """
    gdf = gpd.read_file(geojson_path)

    # Buffer, merge, and explode
    buffered = gdf.geometry.buffer(100)
    merged = unary_union(buffered)
    exploded = gpd.GeoDataFrame(geometry=gpd.GeoSeries(merged).explode(index_parts=False))

    # Filter small fragments
    exploded["area"] = exploded.geometry.area
    max_area = exploded["area"].max()
    clean = exploded[exploded["area"] >= 0.3 * max_area].copy()

    # Un-buffer to restore original size
    clean["geometry"] = clean.geometry.buffer(-100)

    # Sort by area descending (tissue_id=0 is the largest)
    clean = clean.sort_values("area", ascending=False).reset_index(drop=True)
    return clean


def assign_patches_to_tissue(
    tissue_gdf: gpd.GeoDataFrame,
    patch_coords: np.ndarray,
) -> np.ndarray:
    """
    Assign each patch to the nearest tissue polygon via spatial join.

    Args:
        tissue_gdf: GeoDataFrame of merged tissue polygons
        patch_coords: (N, 2) array of patch (x, y) coordinates

    Returns:
        (N,) integer array of tissue IDs
    """
    points_df = pd.DataFrame(patch_coords, columns=["x", "y"])
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df["x"], points_df["y"]),
    )

    # Nearest spatial join — every patch gets assigned to the closest polygon
    joined = gpd.sjoin_nearest(points_gdf, tissue_gdf, how="left")
    return joined["index_right"].fillna(0).astype(np.int64).values


def assign_all_h5(h5_dir: str) -> None:
    """
    Process all H5 files: read contours, assign tissue_ids, save back.
    """
    h5_path = Path(h5_dir)
    h5_files = sorted(h5_path.glob("**/*.h5"))
    contour_dir = h5_path / "contours_geojson"

    for h5_file in track(h5_files, description="Assigning tissue IDs"):
        geojson_file = contour_dir / f"{h5_file.stem}.geojson"
        if not geojson_file.exists():
            print(f"WARNING: No contour file for {h5_file.name}, skipping.")
            continue

        tissue_gdf = merge_polygons(geojson_file)

        with h5py.File(h5_file, "r+") as f:
            coords = f["coords"][:]
            tissue_ids = assign_patches_to_tissue(tissue_gdf, coords)

            if "tissue_id" in f:
                del f["tissue_id"]
            f.create_dataset("tissue_id", data=tissue_ids)

    print(f"Assigned tissue IDs for {len(h5_files)} H5 files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1c: Assign tissue IDs to patches from GeoJSON contours"
    )
    parser.add_argument("--h5-dir", required=True, help="Directory with H5 files")
    args = parser.parse_args()
    assign_all_h5(args.h5_dir)


if __name__ == "__main__":
    main()
