"""
Shared utility functions used across pipeline stages.

Includes filename parsing, IPS grading logic, and config loading.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml


# =============================================================================
# Filename parsing
# =============================================================================

def parse_filename(filename: str) -> tuple[int, int]:
    """
    Extract case and slide IDs from a WSI filename.

    Filename format: '#<case>-<slide> <accession>.tiff'
    Example: '#1-1 7817B8509.tiff' → (case=1, slide=1)

    Returns:
        (case_id, slide_id) as integers
    """
    match = re.match(r"#(\d+)-(\d+)\s", filename)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")
    return int(match.group(1)), int(match.group(2))


# =============================================================================
# IPS grading
# =============================================================================

def total_to_ips_gold(total: int) -> int:
    """
    Convert a summed slide-grade total (integer) to an IPS category.

    This is used for ground-truth labels where grades are exact integers.
    The thresholds follow the clinical protocol from Taskin et al., 2022:
      - IPS-A (0): total ∈ [3, 6]  → non/minimally infiltrative
      - IPS-B (1): total ∈ [7, 9]  → moderately infiltrative
      - IPS-C (2): total ∈ [10, 15] → highly infiltrative
    """
    if total < 7:
        return 0  # IPS-A
    if total < 10:
        return 1  # IPS-B
    return 2      # IPS-C


def total_to_ips_pred(total: float) -> int:
    """
    Convert a summed predicted-grade total (float) to an IPS category.

    Uses shifted thresholds (midpoints) to handle non-integer predictions:
      - IPS-A (0): total ≤ 6.5
      - IPS-B (1): total ≤ 9.5
      - IPS-C (2): total > 9.5
    """
    if total <= 6.5:
        return 0  # IPS-A
    if total <= 9.5:
        return 1  # IPS-B
    return 2      # IPS-C


# =============================================================================
# Config loading
# =============================================================================

def load_fold_config(yaml_path: str | Path) -> dict[str, list[int]]:
    """
    Load cross-validation fold definitions from a YAML file.

    Returns a dict with keys 's1_keys' through 's5_keys', each mapping
    to a list of patient (case) IDs.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def load_wsi_info(csv_path: str | Path) -> pd.DataFrame:
    """
    Load WSI metadata from a CSV file.

    Expected columns: filename, case, slide, patient, grade, ips
    Drops any unnamed/empty trailing columns.

    Returns:
        DataFrame with one row per WSI slide.
    """
    df = pd.read_csv(csv_path)
    # Drop unnamed trailing columns (artifacts from spreadsheet export)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Strip whitespace from patient IDs (some entries have trailing spaces)
    df["patient"] = df["patient"].astype(str).str.strip()
    return df
