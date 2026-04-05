"""
Global constants shared across all pipeline stages.

These values are fixed by the data acquisition and annotation protocol
and should NOT be changed between experiments.
"""

# =============================================================================
# Stage 1: Patch extraction
# =============================================================================

PATCH_SIZE = 1024       # Patch side length in pixels (1024×1024 at 40x mag)
MAGNIFICATION = 40      # Extraction magnification (40x)
MIN_TISSUE_RATIO = 0.60 # Minimum tissue proportion to keep a patch

# =============================================================================
# Stage 3: Graph construction
# =============================================================================

# Patch tissue class IDs (output of the 3-class patch classifier)
STROMA_CLASS_ID = 0
PANNET_CLASS_ID = 1
NORMAL_CLASS_ID = 2

# 8-connected neighborhood for border detection and component filtering
NEIGHBOR_CONNECTIVITY = 8

# =============================================================================
# Stage 5: Training
# =============================================================================

# Feature dimensions at each pipeline stage
VIRCHOW2_DIM = 1280      # VirChow2 foundation model output
AUTOENCODER_DIM = 256     # After autoencoder compression
HIDDEN_DIM = 256          # GNN hidden dimension (same as AE output)

# GigaTIME virtual protein extraction
GIGATIME_DIM = 21  # 23 total channels minus 2 background (TRITC, Cy5)
GIGATIME_FUNCTIONAL_INDICES = [
    0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
]  # Exclude index 1 (TRITC) and 2 (Cy5)
GIGATIME_CHANNEL_NAMES = [
    "DAPI", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16",
    "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
    "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin",
]

# The 25 seeds used for statistical evaluation (matches thesis protocol)
EVAL_SEEDS = [
    42, 777, 5999, 3232, 2832, 4211, 1819, 1412, 1997, 14273,
    31764, 29145, 86392, 3231, 3233, 4, 6, 11, 13, 105,
    107, 112, 116, 117, 119,
]
