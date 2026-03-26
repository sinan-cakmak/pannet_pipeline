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
#SBATCH --array=0-7
#SBATCH --output=/scratch/hcakmak20/pannet_pipeline/logs/train_%a_%j.out
#SBATCH --error=/scratch/hcakmak20/pannet_pipeline/logs/train_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hcakmak20@ku.edu.tr

# =============================================================================
# STAGE 5: Train the bipartite GIN model (PARALLEL: 8 GPUs)
#
# With 4 folds × 25 seeds = 100 experiments, we split across 8 GPUs.
# Each array task handles ~12-13 (fold, seed) combinations.
# =============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"
cd "$PROJECT_DIR"

GRAPH_DIR="${GRAPH_DIR:-/scratch/hcakmak20/pannet_pipeline/graphs}"
# Auto-detect latest autoencoder checkpoint
AE_CKPT="${AE_CKPT:-$(ls -t "$PROJECT_DIR"/checkpoints/autoencoder/autoencoder-*.ckpt 2>/dev/null | head -1)}"
if [ -z "$AE_CKPT" ]; then echo "ERROR: No autoencoder checkpoint found"; exit 1; fi
echo "Using AE checkpoint: $AE_CKPT"
NUM_LAYERS="${NUM_LAYERS:-1}"
BORDER_DISTANCE="${BORDER_DISTANCE:-3}"
CELL_INFO_MODE="${CELL_INFO_MODE:-none}"

CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=8

# 25 seeds from thesis evaluation protocol
# SEEDS=(42 777 5999 3232 2832 4211 1819 1412 1997 14273 31764 29145 86392 3231 3233 4 6 11 13 105 107 112 116 117 119)
SEEDS=(42 128 1234 5 777)
FOLDS=(0 1 2 3)

# Build all (fold, seed) combinations and select this chunk's subset
ALL_COMBOS=()
for FOLD in "${FOLDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        ALL_COMBOS+=("$FOLD:$SEED")
    done
done

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/experiments"

echo "=== Chunk $CHUNK/$NUM_CHUNKS: ${NUM_LAYERS} layer(s), border=$BORDER_DISTANCE ==="
echo "=== Total combos: ${#ALL_COMBOS[@]}, this chunk handles every ${NUM_CHUNKS}th ==="

IDX=0
for COMBO in "${ALL_COMBOS[@]}"; do
    if (( IDX % NUM_CHUNKS == CHUNK )); then
        FOLD="${COMBO%%:*}"
        SEED="${COMBO##*:}"
        echo "--- Fold: $FOLD, Seed: $SEED ---"
        uv run python -m src.stage5_training.train \
            --seed "$SEED" \
            --test-fold "$FOLD" \
            --graph-dir "$GRAPH_DIR" \
            --ae-checkpoint "$AE_CKPT" \
            --num-gnn-layers "$NUM_LAYERS" \
            --hop-distance "${BORDER_DISTANCE}" \
            --border-distance "${BORDER_DISTANCE}" \
            --batch-size 8 \
            --max-epochs 200 \
            --output-dir experiments \
            --skip-existing
    fi
    IDX=$((IDX + 1))
done

echo "=== Chunk $CHUNK complete ==="
