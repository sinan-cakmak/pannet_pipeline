# CLAUDE.md — PanNET Bipartite GNN Pipeline

## Project Overview

End-to-end pipeline for PanNET infiltration pattern scoring using a bipartite GNN. Implements the methodology from Nusret Özateş's MSc thesis (Koç University, Jan 2026). Predicts patient-level Infiltration Pattern Scores (IPS A/B/C) from whole-slide images.

## Pipeline Stages

```
Stage 1: WSI → H5 files        (Trident + VirChow2 + GrandQC)
Stage 2: H5 → AutoEncoder      (1280-d → 256-d compression)
Stage 3: H5 → .pkl graphs      (bipartite graph construction + caching)
Stage 5: .pkl → IPS prediction  (GIN + Huber regression + patient aggregation)
```

## Build & Run

Package manager is **uv**. All commands from project root.

```bash
uv sync                          # Install dependencies
# Stage 2 (if H5 files available):
uv run python -m src.stage2_autoencoder.train --h5-dir /path/to/features
# Stage 3:
uv run python -m src.stage3_graph.build_all --h5-dir /path --output-dir graphs --ae-checkpoint ckpt.ckpt
# Stage 5:
uv run python -m src.stage5_training.train --graph-dir graphs --ae-checkpoint ckpt.ckpt --test-fold 0 --seed 42
```

SLURM (HPC): `sbatch scripts/run_train.sh` (4-fold array, 25 seeds each)

## Key Design Decisions

- **Bipartite graph**: Edges only between PanNET↔NNP patches. Thesis proves this beats full graphs.
- **PanNET-only pooling**: `global_add_pool` over tumor nodes only (NNP context absorbed via message passing).
- **Huber loss (δ=2)**: Ordinal regression. Outperforms classification (cross-entropy) significantly.
- **1-layer GIN**: Best at radius 3. Deeper architectures over-smooth.
- **4-fold CV, 25 seeds**: Matches thesis statistical evaluation protocol.

## Architecture

```
src/
├── constants.py                # PATCH_SIZE=1024, class IDs, dimensions
├── utils.py                    # Filename parsing, IPS grading, config loading
├── stage1_extraction/          # WSI → H5 (Trident wrapper, patch classifier, tissue IDs)
├── stage2_autoencoder/         # 1280→256 compression (LightningModule)
├── stage3_graph/               # Bipartite graph construction (morphology, adjacency, builder)
└── stage5_training/            # GIN model, regression head, evaluation, training loop
```

## Environment

- `WSI_DIR`: Raw WSI slides (HPC: `/userfiles/cgunduz/.../PANNET Slides`)
- `WSI_FEATURES_DIR`: Trident H5 output
- `WSI_INFO_PATH`: CSV with filename, case, slide, patient, grade, ips
- `WSI_SPLIT_PATH`: YAML with fold definitions (s1-s5 patient IDs)

## Data

- 252 WSIs from 73 patients, Koç University Hospital
- Grades 1-5 (stored internally as 0-4)
- IPS: sum of 3 slides → [3-6]=IPS-A, [7-9]=IPS-B, [10-15]=IPS-C
- Expected thesis result: ~70.74% macro F1 (1-layer GIN, r=3, bipartite)
