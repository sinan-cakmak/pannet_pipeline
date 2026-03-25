#!/bin/bash
#SBATCH --job-name=PanNET_Extract
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/extraction_%j.out
#SBATCH --error=logs/extraction_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hcakmak20@ku.edu.tr

# =============================================================================
# STAGE 1: WSI → H5 feature files
#
# Runs three sequential steps:
#   1a. Extract VirChow2 features from WSIs using Trident
#   1b. Classify patches as PanNET/Normal/Stroma
#   1c. Assign tissue IDs from GeoJSON contours
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configurable paths (override via --export)
WSI_DIR="${WSI_DIR:-/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/hcakmak20/pannet_pipeline/features}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-checkpoints/patch_classifier/best.ckpt}"

mkdir -p logs "$OUTPUT_DIR"

echo "=== Stage 1a: Extracting VirChow2 features ==="
uv run python -m src.stage1_extraction.extract_features \
    --wsi-dir "$WSI_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "=== Stage 1b: Classifying patches ==="
uv run python -m src.stage1_extraction.classify_patches \
    --h5-dir "$OUTPUT_DIR" \
    --checkpoint "$CLASSIFIER_CKPT"

echo "=== Stage 1c: Assigning tissue IDs ==="
uv run python -m src.stage1_extraction.assign_tissue_ids \
    --h5-dir "$OUTPUT_DIR"

echo "=== Stage 1 complete ==="
