"""
STAGE 3+4 entry point: Build bipartite graphs from H5 files and save as .pkl.

For each WSI:
  1. Load features, coords, patch_classes, tissue_ids from H5
  2. Select tissue_id=0 (largest tissue piece)
  3. Build bipartite graph via builder.build_graph()
  4. Save as a .pkl file (PyG Data object)

Usage:
  uv run python -m src.stage3_graph.build_all \\
      --h5-dir "/path/to/features" \\
      --output-dir "/path/to/graphs" \\
      --ae-checkpoint "checkpoints/autoencoder/best.ckpt" \\
      --hop-distance 3 \\
      --border-distance 3
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import json

import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from rich.progress import track

from src.stage2_autoencoder.model import AutoEncoder
from src.stage3_graph.builder import build_graph
from src.utils import load_wsi_info


def parse_gigatime_features(h5_file: h5py.File) -> np.ndarray | None:
    """
    Load GigaTIME protein counts from H5 file.

    Returns (N, 21) float32 array, or None if the dataset doesn't exist.
    """
    if "gigatime_features" not in h5_file:
        return None
    return h5_file["gigatime_features"][:].astype(np.float32)


def parse_cell_types_v2(h5_file: h5py.File) -> np.ndarray | None:
    """
    Parse patch_cell_types_v2 from H5 into a (N, 4) cell_information array.

    Each entry in patch_cell_types_v2 is a JSON string like:
      {"neoplastic": 41, "inflammatory": 2, "connective": 67, "dead": 0,
       "macrophage_cell": 0, "macrophage_nuc": 0, "position": [...], "is_complete": true}

    We collapse into 4 dims:
      [0] = neoplastic count
      [1] = inflammatory count
      [2] = other (connective + dead + macrophage_cell + macrophage_nuc)
      [3] = 0 (reserved)

    Returns None if the dataset doesn't exist.
    """
    if "patch_cell_types_v2" not in h5_file:
        return None

    raw = h5_file["patch_cell_types_v2"][:]
    n = len(raw)
    cell_info = np.zeros((n, 4), dtype=np.float32)

    for i, entry in enumerate(raw):
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8")
        d = json.loads(entry)
        cell_info[i, 0] = d.get("neoplastic", 0)
        cell_info[i, 1] = d.get("inflammatory", 0)
        cell_info[i, 2] = (
            d.get("connective", 0)
            + d.get("dead", 0)
            + d.get("macrophage_cell", 0)
            + d.get("macrophage_nuc", 0)
        )
        # cell_info[i, 3] stays 0 (reserved)

    return cell_info


def load_encoder(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load the frozen autoencoder encoder from a Lightning checkpoint."""
    autoencoder = AutoEncoder.load_from_checkpoint(checkpoint_path, map_location=device)
    encoder = autoencoder.encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder.to(device)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 3+4: Build graphs and save as .pkl")
    parser.add_argument("--h5-dir", required=True, help="Directory with H5 feature files")
    parser.add_argument("--output-dir", required=True, help="Directory to save .pkl graphs")
    parser.add_argument("--ae-checkpoint", required=True, help="AutoEncoder checkpoint path")
    parser.add_argument("--wsi-info", default="data/wsi_information.csv", help="WSI info CSV")
    parser.add_argument("--hop-distance", type=int, default=3)
    parser.add_argument("--border-distance", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cell-source", default="none", choices=["none", "hovernet", "gigatime"],
                        help="Source of cell/protein info: none, hovernet (patch_cell_types_v2), gigatime")
    parser.add_argument("--chunk", type=int, default=0, help="Which chunk (0-indexed)")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total parallel chunks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load WSI metadata (need grade labels)
    wsi_info = load_wsi_info(args.wsi_info)
    # Build filename→grade lookup (grades are 1-5 in CSV, stored as 0-4 internally)
    grade_lookup = dict(zip(wsi_info["filename"], wsi_info["grade"] - 1))

    # Load frozen autoencoder encoder
    print(f"Loading autoencoder from: {args.ae_checkpoint}")
    encoder = load_encoder(args.ae_checkpoint, args.device)

    # Process H5 files (chunked for parallel execution)
    h5_dir = Path(args.h5_dir)
    all_h5_files = sorted(h5_dir.glob("**/*.h5"))
    h5_files = all_h5_files[args.chunk :: args.num_chunks]
    print(f"Chunk {args.chunk}/{args.num_chunks}: processing {len(h5_files)}/{len(all_h5_files)} H5 files")

    saved = 0
    skipped = 0

    for h5_file in track(h5_files, description="Building graphs"):
        # Derive the WSI filename (e.g., "#1-1 7817B8509.tiff")
        wsi_filename = h5_file.stem + ".tiff"

        # Look up grade label
        grade = grade_lookup.get(wsi_filename)
        if grade is None:
            print(f"WARNING: No grade found for {wsi_filename}, skipping.")
            skipped += 1
            continue

        # Load data from H5
        with h5py.File(h5_file, "r") as f:
            features = f["features"][:]
            coords = f["coords"][:]
            patch_classes = f["patch_classes"][:]
            tissue_ids = f["tissue_id"][:] if "tissue_id" in f else np.zeros(len(features), dtype=np.int64)
            slide_width = int(f.attrs.get("slide_width", coords[:, 0].max() + 1024))
            slide_height = int(f.attrs.get("slide_height", coords[:, 1].max() + 1024))
            # Parse cell/protein information based on source
            if args.cell_source == "gigatime":
                cell_info = parse_gigatime_features(f)
            elif args.cell_source == "hovernet":
                cell_info = parse_cell_types_v2(f)
            else:
                cell_info = None

        # Only use first 1280 dims (VirChow2 output)
        if features.shape[1] > 1280:
            features = features[:, :1280]

        # Select largest tissue region (tissue_id=0)
        tissue_mask = tissue_ids == 0
        if not tissue_mask.any():
            tissue_mask = np.ones(len(features), dtype=bool)  # fallback: use all

        t_locs = coords[tissue_mask]
        t_classes = patch_classes[tissue_mask]
        t_feats = features[tissue_mask]
        t_cell_info = cell_info[tissue_mask] if cell_info is not None else None

        # Build graph
        data = build_graph(
            patch_locs=t_locs,
            patch_classes=t_classes,
            patch_feats=t_feats,
            slide_width=slide_width,
            slide_height=slide_height,
            grade_label=grade,
            filename=wsi_filename,
            hop_distance=args.hop_distance,
            border_distance=args.border_distance,
            encoder=encoder,
            device=args.device,
            cell_information=t_cell_info,
        )

        if data is None:
            skipped += 1
            continue

        # Save as .pkl (Stage 4)
        stem = h5_file.stem
        pkl_name = f"{stem}_hop_{args.hop_distance}_border_{args.border_distance}.pkl"
        pkl_path = output_dir / pkl_name

        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        saved += 1

    print(f"\nDone. Saved {saved} graphs, skipped {skipped}.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
