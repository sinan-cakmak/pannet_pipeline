#!/bin/bash
#SBATCH --job-name=PanNET_GIN_%a
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-3
#SBATCH --output=logs/train_fold_%a_%j.out
#SBATCH --error=logs/train_fold_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hcakmak20@ku.edu.tr

# =============================================================================
# STAGE 5: Train the bipartite GIN model
#
# Array job: index 0-3 = 4 CV folds
# Loops over 25 seeds per fold (matching thesis evaluation protocol)
# =============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"
cd "$PROJECT_DIR"

# Configurable via --export
TEST_FOLD=${SLURM_ARRAY_TASK_ID:-${TEST_FOLD:-0}}
GRAPH_DIR="${GRAPH_DIR:-/scratch/hcakmak20/pannet_pipeline/graphs}"
AE_CKPT="${AE_CKPT:-checkpoints/autoencoder/best.ckpt}"
NUM_LAYERS="${NUM_LAYERS:-1}"
BORDER_DISTANCE="${BORDER_DISTANCE:-3}"

# 25 seeds from thesis evaluation protocol
SEEDS="42 777 5999 3232 2832 4211 1819 1412 1997 14273 31764 29145 86392 3231 3233 4 6 11 13 105 107 112 116 117 119"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/experiments"

echo "=== Fold $TEST_FOLD, $NUM_LAYERS layer(s), border=$BORDER_DISTANCE ==="

for SEED in $SEEDS; do
    echo "--- Seed: $SEED ---"
    uv run python -m src.stage5_training.train \
        --seed "$SEED" \
        --test-fold "$TEST_FOLD" \
        --graph-dir "$GRAPH_DIR" \
        --ae-checkpoint "$AE_CKPT" \
        --num-gnn-layers "$NUM_LAYERS" \
        --batch-size 8 \
        --max-epochs 200 \
        --output-dir experiments
done

echo "=== Fold $TEST_FOLD complete ==="
