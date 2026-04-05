# CLAUDE.md — PanNET Bipartite GNN Pipeline

## Project Overview

End-to-end pipeline for PanNET (Pancreatic Neuroendocrine Tumor) infiltration pattern scoring using a bipartite Graph Neural Network. Implements the methodology from Nusret Özateş's MSc thesis (Koç University, Jan 2026, advisor Prof. Çiğdem Gündüz Demir).

**Goal:** Predict patient-level Infiltration Pattern Scores (IPS-A/B/C) from whole-slide images by constructing bipartite graphs at the tumor–NNP (non-neoplastic parenchyma) interface and training a GIN model on them.

## Medical Context

- **Infiltration pattern grading (1-5):** Assessed at the tumor–NNP interface per slide. Grade 1 = fully demarcated, Grade 5 = highly infiltrative with distant clusters.
- **IPS:** Sum 3 slide grades per patient → [3-6]=IPS-A (non-infiltrative), [7-9]=IPS-B (moderate), [10-15]=IPS-C (highly infiltrative).
- **Inter-observer agreement is moderate** (Fleiss' κ = 0.526) → labels are inherently noisy.
- **Dataset:** 252 H&E-stained WSIs from 73 patients, Koç University Hospital. 61 patients with 3 WSIs (evaluation), 12 with <3 (training only).

## Pipeline Stages

```
Stage 1: WSI → H5 files        (Trident + VirChow2 + GrandQC)
Stage 2: H5 → AutoEncoder      (1280-d → 256-d, frozen projector)
Stage 3: H5 → .pkl graphs      (bipartite graph construction + caching)
Stage 5: .pkl → IPS prediction  (GIN + Huber regression + patient aggregation)
```

## Build & Run

Package manager is **uv**. Project runs on **SLURM HPC cluster** (KUIS AI, A40 GPUs).

```bash
uv sync                          # Install dependencies
uv pip install poetry-core       # Needed for Trident
uv pip install -e ../trident --no-build-isolation  # Install Trident

# Stage 2: Train autoencoder (~5 min, 1 GPU)
sbatch scripts/run_autoencoder.sh

# Stage 3: Build bipartite graphs (~2 min, 8 GPUs parallel)
sbatch scripts/run_graph_build.sh

# Stage 5: Train GIN (4 folds × 25 seeds, 8 GPUs parallel, ~1-2h)
sbatch scripts/run_train.sh

# Full ablation sweep (9 configs: 3 radii × 3 depths)
bash scripts/run_all_experiments.sh
```

**Local development** (CPU, from project root):
```bash
uv run python -m src.stage5_training.train --graph-dir graphs --ae-checkpoint ckpt.ckpt --test-fold 0 --seed 42 --device cpu
```

## Key Architecture Details

### Bipartite Graph (Stage 3)
- **Only PanNET↔NNP edges** — no tumor-tumor or stroma-stroma connections
- Border detection: 8-connected neighborhood check on PanNET patches
- Morphological hole filling before border detection (scipy.ndimage.binary_fill_holes)
- Connected component filtering: remove clusters < `sum(8*i for i in 1..hop_distance)` nodes
- Distance metric: Chebyshev (L-infinity), threshold = hop_distance × 1024

### GIN Model (Stage 5)
- **Frozen projector:** AutoEncoder encoder (1280→256), never updated
- **GINConv layers:** MLP(256→256→256) with RMSNorm, ReLU, Dropout(0.4), residual connections
- **PanNET-only pooling:** global_add_pool over tumor nodes only (NNP context absorbed via message passing)
- **Regression head:** RMSNorm→Linear(256,128)→RMSNorm→ReLU→Dropout(0.4)→Linear(128,1)
- **Loss:** Huber (δ=2) — ordinal regression, NOT classification
- **Optimizer:** AdamW (lr=1e-4, wd=1e-3), ReduceLROnPlateau (factor=0.8, patience=5)
- **Training:** batch_size=8, max 200 epochs, early stopping patience=15, bf16-mixed, WeightedRandomSampler
- **Best config (from thesis):** 1 layer, r=3, achieves ~70.7% macro F1

### Cross-Validation
- **4-fold CV:** s1-s4 rotate as test/val (val = test+1 mod 4), s5 always train
- **25 seeds** per configuration for statistical robustness
- Patient IDs per fold defined in `data/fold_information.yaml`
- IPS thresholds — gold: <7/<10, predicted: ≤6.5/≤9.5

## Architecture

```
src/
├── constants.py                 # PATCH_SIZE=1024, class IDs, dims, 25 eval seeds
├── utils.py                     # Filename parsing, IPS grading, config loading
├── stage1_extraction/           # WSI → H5 (Trident/VirChow2/GrandQC wrappers)
│   ├── extract_features.py      # Trident Processor API: seg → coords → feat
│   ├── classify_patches.py      # 3-class MLP: stroma(0)/PanNET(1)/normal(2)
│   └── assign_tissue_ids.py     # GeoJSON contour → tissue_id via spatial join
├── stage2_autoencoder/          # Feature compression (LightningModule)
│   ├── model.py                 # Encoder: 1280→768→512→256 (RMSNorm, GELU, Dropout)
│   ├── dataset.py               # Loads H5 features[:, :1280] as flat vectors
│   └── train.py                 # AdamW, MSE+variance loss, 90/10 split by file
├── stage3_graph/                # Bipartite graph construction
│   ├── morphology.py            # fill_pannet_holes(), find_border_pannets()
│   ├── adjacency.py             # build_bipartite_edges(), compute_edge_distances()
│   ├── builder.py               # build_graph() — full pipeline per tissue region
│   └── build_all.py             # Entry point: H5 → .pkl for all slides
└── stage5_training/             # GIN training & evaluation
    ├── models/gin.py            # GINFeatureExtractor (PanNET-only pooling)
    ├── regression_model.py      # InfiltrationModel (LightningModule)
    ├── dataset.py               # GraphDataset — loads .pkl, filters by hop/border
    ├── data_module.py           # PanNETDataModule — fold splits, WeightedRandomSampler
    ├── evaluation.py            # Patient-level IPS aggregation, F1, QWK, MAE
    ├── db.py                    # PostgreSQL logging (Neon DB)
    └── train.py                 # Main entry point
```

## HPC Environment

- **Cluster:** KUIS AI Center, SLURM scheduler
- **GPUs:** NVIDIA A40 (Tensor Cores, use `torch.set_float32_matmul_precision('high')`)
- **Project path:** `/scratch/hcakmak20/pannet_pipeline`
- **WSI slides:** `/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides`
- **Pre-extracted H5 features:** `/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/pannet_wsi_features/40x_1024px_0px_overlap/features_virchow2` (252 files, VirChow2 2560-d)
- **Trident:** `/scratch/hcakmak20/trident` (cloned from github.com/mahmoodlab/TRIDENT)
- **CUDA compatibility:** torch 2.4.x (driver is CUDA 12.0.9 on some nodes)
- **SSH:** `ssh hcakmak20@172.20.240.205` (key-based auth, requires VPN)
- **SLURM:** partition=ai, qos=ai, account=ai, max 8 GPUs concurrent
- **All scripts use:** `PROJECT_DIR="/scratch/hcakmak20/pannet_pipeline"` with hardcoded cd (BASH_SOURCE doesn't work under SLURM)
- **RichProgressBar crashes on SLURM** — removed from all trainers

## Database

- **Neon PostgreSQL** for experiment logging
- Connection string in `src/stage5_training/db.py`
- Table: `bipartite_experiments` (run_id, fold, seed, layers, hop/border distance, metrics)
- Fallback: JSON files in `experiments/results/`
- Migration script: `scripts/migrate_json_to_db.py`
- **Important:** Metrics must be plain `float()`, not `np.float64` (PostgreSQL rejects numpy types)
- **Experiment identity columns:** The DB skip check (`--skip-existing`) and experiment naming use these columns to uniquely identify a run: `fold, seed, num_gnn_layers, hop_distance, cell_info_mode, cell_info_dim, log_normalize`. When adding a new experimental parameter, you MUST: (1) add the column to `CREATE_TABLE_SQL`, (2) add it to `INSERT_SQL`, (3) add it to `build_log_entry()`, (4) add it to the skip check query in `train.py`, (5) add it to the experiment name string, (6) add an env var in `run_train.sh`, and (7) run `ALTER TABLE` on the live DB.

## Data

### H5 files (from Trident, 252 files)
```
features         (N, 2560)  VirChow2 embeddings (use first 1280 for autoencoder)
coords           (N, 2)     patch (x, y) top-left pixel coordinates
patch_classes    (N,)       0=stroma, 1=PanNET, 2=normal
tissue_id        (N,)       connected tissue region ID (use tissue_id=0)
slide_width      attr       WSI dimensions in pixels
slide_height     attr
```

### Graph .pkl files (PyG Data objects)
```
x               (N, 256)    autoencoder-compressed features
edge_index      (2, E)      bipartite edges (PanNET ↔ NNP only)
y               scalar      grade label (0–4 internally, 1–5 clinical)
pos             (N, 2)      patch pixel coordinates
patch_classes   (N,)        tissue type per node
border_distances (N, 1)     distance to tumor–NNP border
edge_distances  (E, 1)      L∞ distance / 1024
filename        string      source WSI filename
```
Naming: `{slide_name}_hop_{r}_border_{d}.pkl` — dataset filters by this pattern.

### WSI info: `data/wsi_information.csv`
Columns: filename, case, slide, patient, grade (1-5), ips (A/B/C)

### Folds: `data/fold_information.yaml`
s1-s4 rotate test/val, s5 always train. Patient IDs (case numbers) per split.

## Key Design Decisions (from thesis ablation)

| Decision | Choice | Evidence |
|----------|--------|----------|
| Graph type | Bipartite only | Bipartite > full graph at every radius (Table 5.5) |
| Pooling | Sum over PanNET nodes only | NNP info absorbed via message passing |
| Loss | Huber (δ=2) | Regression >> classification (Table 5.4) |
| GNN layers | 1 (default) | 1 > 2 > 3 at r=3 (Table 5.7) |
| Radius | 3 (default) | r=3 > r=2 > r=1 (Table 5.5) |
| GNN backbone | GIN | GIN ≈ GATv2, but simpler and lower MAE |
| Feature model | VirChow2 (1280-d) | Pathology foundation model |
| Dim reduction | AutoEncoder (→256-d) | Decoupled, avoids overfitting |
| CV protocol | 4-fold, 25 seeds | Statistical robustness, Wilcoxon test |

## Thesis Results (target to reproduce)

| Model | Macro F1 | QWK |
|-------|----------|-----|
| DeepSets (MIL) | 63.1% | 54.1 |
| AB-MIL | 63.2% | 55.5 |
| Patch-GCN (full graph) | 61.4% | 50.7 |
| Context-Aware MIL | 63.3% | 62.9 |
| **Bipartite GIN (1L, r=3)** | **70.7%** | **69.1** |

## Known Issues & Gotchas

- **GraphDataset must filter by hop_distance** via filename glob pattern. Loading all .pkl files mixes graph structures and hurts performance significantly.
- **torch version must be 2.4.x** for CUDA 12.0 driver compatibility on some HPC nodes. torch>=2.8 crashes.
- **SLURM scripts use hardcoded PROJECT_DIR** because `BASH_SOURCE` doesn't resolve correctly when SLURM copies scripts.
- **`((IDX++))` in bash** returns exit code 1 when IDX=0 → kills script under `set -e`. Use `IDX=$((IDX + 1))` instead.
- **RichProgressBar** crashes on SLURM (no interactive terminal). Removed from all Lightning trainers.
- **H5 features are 2560-d** (VirChow2 class_token + pooled_patches). Autoencoder uses `features[:, :1280]`.
- **Graphs already contain 256-d features** (projected in Stage 3). The regression model auto-detects: if `x.shape[1] > 256`, applies projector; otherwise skips it.
- **psycopg2 requires plain `float()`**, not `np.float64`. All metrics in evaluation.py are cast to float.

## Experiment Tracking

All experiment iterations with their methodology and results must be documented in `docs/experiment_log.md`. Update this file after every training run with: trial name, what changed, why, results, and takeaways. This ensures reproducibility and prevents re-running failed approaches.

## Future Work

- **Cell information gating:** Extract cell counts per patch via HoVer-Net (tiatoolbox), add CellConditionedConv that gates message passing based on cell composition of source/target patches. This combines Nusret's bipartite approach with cell-level information from Koc et al. 2025.
- **Additional conv types:** Test GATv2, SAGE, Transformer on bipartite graphs.
- **Multi-scale graphs:** Combine graphs at different hop distances.
- **Autoresearch:** Autonomous agent-driven hyperparameter/architecture exploration.

## Related Projects

- `pannet_gnn/` — Sinan's earlier project with full-graph GNN, 7 conv types, cell_info_mode (gate/concat/none), UNI v2 features (2560-d raw)
- `context-aware-gnn/` — Nusret's original implementation (reference, not used directly)
- `trident/` — WSI preprocessing tool (github.com/mahmoodlab/TRIDENT)
- Nusret's thesis PDF: `/Users/sinan/Downloads/Nusret Pannet thesis.pdf`
