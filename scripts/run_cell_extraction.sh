#!/bin/bash
#SBATCH --job-name=PanNET_Cell_%a
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=/scratch/hcakmak20/pannet_pipeline/logs/cell_extract_%a_%j.out
#SBATCH --error=/scratch/hcakmak20/pannet_pipeline/logs/cell_extract_%a_%j.err

# =============================================================================
# STAGE 1.5: Extract cell counts per patch using HoVer-Net (8 GPUs parallel)
# =============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -v '^$' | xargs)
fi

CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=${NUM_CHUNKS:-8}

WSI_DIR="${WSI_DIR:-/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides}"
H5_DIR="${H5_DIR:-${WSI_FEATURES_DIR:-/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/pannet_wsi_features/40x_1024px_0px_overlap/features_virchow2}}"
STAIN_REF="${STAIN_REF:-}"  # Optional: path to reference image for Macenko normalization

mkdir -p "$PROJECT_DIR/logs"

STAIN_ARG=""
if [ -n "$STAIN_REF" ]; then
    STAIN_ARG="--stain-ref $STAIN_REF"
fi

echo "=== Chunk $CHUNK/$NUM_CHUNKS: Extracting cell counts ==="
uv run python -m src.stage1_extraction.extract_cell_info \
    --h5-dir "$H5_DIR" \
    --wsi-dir "$WSI_DIR" \
    --chunk "$CHUNK" \
    --num-chunks "$NUM_CHUNKS" \
    $STAIN_ARG

echo "=== Chunk $CHUNK complete ==="
