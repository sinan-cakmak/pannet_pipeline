#!/bin/bash
#SBATCH --job-name=PanNET_Extract_%a
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-7
#SBATCH --output=/scratch/hcakmak20/pannet_pipeline/logs/extraction_%a_%j.out
#SBATCH --error=/scratch/hcakmak20/pannet_pipeline/logs/extraction_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hcakmak20@ku.edu.tr

# =============================================================================
# STAGE 1: WSI → H5 feature files (PARALLEL: 8 GPUs, each handles ~32 slides)
#
# Array job: 252 slides split across 8 tasks.
# Each task runs all 3 sub-stages on its chunk of slides.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"
cd "$PROJECT_DIR"

# Load HF_TOKEN from .env for model downloads
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=${NUM_CHUNKS:-8}

WSI_DIR="${WSI_DIR:-/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/hcakmak20/pannet_pipeline/features}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-checkpoints/patch_classifier/best.ckpt}"

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

echo "=== Chunk $CHUNK/$NUM_CHUNKS: Stage 1a — Extracting VirChow2 features ==="
uv run python -m src.stage1_extraction.extract_features \
    --wsi-dir "$WSI_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --chunk "$CHUNK" \
    --num-chunks "$NUM_CHUNKS"

echo "=== Chunk $CHUNK/$NUM_CHUNKS: Stage 1b — Classifying patches ==="
uv run python -m src.stage1_extraction.classify_patches \
    --h5-dir "$OUTPUT_DIR" \
    --checkpoint "$CLASSIFIER_CKPT" \
    --chunk "$CHUNK" \
    --num-chunks "$NUM_CHUNKS"

echo "=== Chunk $CHUNK/$NUM_CHUNKS: Stage 1c — Assigning tissue IDs ==="
uv run python -m src.stage1_extraction.assign_tissue_ids \
    --h5-dir "$OUTPUT_DIR" \
    --chunk "$CHUNK" \
    --num-chunks "$NUM_CHUNKS"

echo "=== Chunk $CHUNK complete ==="
