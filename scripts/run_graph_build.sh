#!/bin/bash
#SBATCH --job-name=PanNET_Graph
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/graph_build_%j.out
#SBATCH --error=logs/graph_build_%j.err

# =============================================================================
# STAGE 3+4: Build bipartite graphs from H5 files → save as .pkl
# =============================================================================

set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

H5_DIR="${H5_DIR:-/scratch/hcakmak20/pannet_pipeline/features}"
GRAPH_DIR="${GRAPH_DIR:-/scratch/hcakmak20/pannet_pipeline/graphs}"
AE_CKPT="${AE_CKPT:-checkpoints/autoencoder/best.ckpt}"

mkdir -p logs "$GRAPH_DIR"

# Build graphs for each hop/border distance combination
for DISTANCE in 1 2 3; do
    echo "=== Building graphs: hop=$DISTANCE, border=$DISTANCE ==="
    uv run python -m src.stage3_graph.build_all \
        --h5-dir "$H5_DIR" \
        --output-dir "$GRAPH_DIR" \
        --ae-checkpoint "$AE_CKPT" \
        --hop-distance "$DISTANCE" \
        --border-distance "$DISTANCE" \
        --device cuda
done

echo "=== Stage 3+4 complete ==="
