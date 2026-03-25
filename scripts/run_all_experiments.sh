#!/bin/bash
# =============================================================================
# Launch the full experiment sweep matching the thesis ablation study.
#
# Configurations:
#   - border_distance: 1, 2, 3
#   - num_layers: 1, 2, 3
#   Total: 9 configurations × 4 folds × 25 seeds = 900 experiments
#
# Usage:
#   cd pannet_pipeline
#   bash scripts/run_all_experiments.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

for BORDER in 1 2 3; do
    for LAYERS in 1 2 3; do
        echo "Submitting: border=$BORDER, layers=$LAYERS"
        sbatch \
            --export=ALL,BORDER_DISTANCE=$BORDER,NUM_LAYERS=$LAYERS \
            scripts/run_train.sh
    done
done

echo "All 9 configurations submitted (4 folds each = 36 array jobs)."
