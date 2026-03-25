#!/bin/bash
#SBATCH --job-name=PanNET_AE
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/autoencoder_%j.out
#SBATCH --error=logs/autoencoder_%j.err

# =============================================================================
# STAGE 2: Train the autoencoder (1280 → 256 feature compression)
# =============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"
cd "$PROJECT_DIR"

H5_DIR="${H5_DIR:-/scratch/hcakmak20/pannet_pipeline/features}"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/checkpoints/autoencoder"

uv run python -m src.stage2_autoencoder.train \
    --h5-dir "$H5_DIR" \
    --batch-size 4096 \
    --epochs 50 \
    --seed 42

echo "=== Stage 2 complete ==="
