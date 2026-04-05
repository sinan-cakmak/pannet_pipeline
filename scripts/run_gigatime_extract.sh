#!/bin/bash
#SBATCH --job-name=PanNET_GigaTIME_%a
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=/scratch/hcakmak20/pannet_pipeline/logs/gigatime_%a_%j.out
#SBATCH --error=/scratch/hcakmak20/pannet_pipeline/logs/gigatime_%a_%j.err

# =============================================================================
# STAGE 1.5g: Extract virtual protein counts per patch using GigaTIME (8 GPUs)
#
# IMPORTANT: Download the model ONCE on the login node before submitting:
#   cd /scratch/hcakmak20/pannet_pipeline
#   source .env
#   uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('prov-gigatime/GigaTIME')"
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

mkdir -p "$PROJECT_DIR/logs"

echo "=== Chunk $CHUNK/$NUM_CHUNKS: Extracting GigaTIME protein counts ==="
uv run python -m src.stage1_extraction.extract_gigatime \
    --h5-dir "$H5_DIR" \
    --wsi-dir "$WSI_DIR" \
    --chunk "$CHUNK" \
    --num-chunks "$NUM_CHUNKS"

echo "=== Chunk $CHUNK complete ==="
