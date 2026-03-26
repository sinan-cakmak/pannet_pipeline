#!/bin/bash
#SBATCH --job-name=PanNET_Graph_%a
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=/scratch/hcakmak20/pannet_pipeline/logs/graph_build_%a_%j.out
#SBATCH --error=/scratch/hcakmak20/pannet_pipeline/logs/graph_build_%a_%j.err

# =============================================================================
# STAGE 3+4: Build bipartite graphs (PARALLEL: 8 GPUs, each handles ~32 slides)
# =============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"
cd "$PROJECT_DIR"

# Load env vars
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -v '^$' | xargs)
fi

CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=${NUM_CHUNKS:-8}

H5_DIR="${H5_DIR:-${WSI_FEATURES_DIR:-/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/pannet_wsi_features/40x_1024px_0px_overlap/features_virchow2}}"
GRAPH_DIR="${GRAPH_DIR:-/scratch/hcakmak20/pannet_pipeline/graphs}"
# Auto-detect latest autoencoder checkpoint (lowest val_loss)
AE_CKPT="${AE_CKPT:-$(ls -t "$PROJECT_DIR"/checkpoints/autoencoder/autoencoder-*.ckpt 2>/dev/null | head -1)}"
if [ -z "$AE_CKPT" ]; then echo "ERROR: No autoencoder checkpoint found"; exit 1; fi
echo "Using AE checkpoint: $AE_CKPT"

mkdir -p "$PROJECT_DIR/logs" "$GRAPH_DIR"

for DISTANCE in 1 2 3; do
    echo "=== Chunk $CHUNK/$NUM_CHUNKS: hop=$DISTANCE, border=$DISTANCE ==="
    uv run python -m src.stage3_graph.build_all \
        --h5-dir "$H5_DIR" \
        --output-dir "$GRAPH_DIR" \
        --ae-checkpoint "$AE_CKPT" \
        --hop-distance "$DISTANCE" \
        --border-distance "$DISTANCE" \
        --device cuda \
        --chunk "$CHUNK" \
        --num-chunks "$NUM_CHUNKS"
done

echo "=== Chunk $CHUNK complete ==="
