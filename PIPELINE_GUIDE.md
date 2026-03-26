# PanNET Bipartite GNN Pipeline — Complete Guide

## The Problem

- **Pancreatic Neuroendocrine Tumors (PanNETs)** are rare tumors graded by their infiltration pattern at the **tumor–NNP interface** (where tumor meets non-neoplastic parenchyma)
- A pathologist examines 3 representative H&E-stained whole-slide images (WSIs) per patient
- Each slide gets a grade (1–5) based on how aggressively the tumor infiltrates surrounding tissue:
  - **Grade 1**: Fully demarcated, round/capsulated
  - **Grade 2**: Mildly irregular borders, early capsular penetration
  - **Grade 3**: Satellite nodules/projections, still connected to main mass
  - **Grade 4**: Small peri-tumoral clusters, no distant invasion
  - **Grade 5**: Non-demarcated, prominent infiltration deep into surrounding tissue
- 3 slide grades are summed → **Infiltration Pattern Score (IPS)**:
  - **IPS-A** (sum 3–6): non/minimally infiltrative
  - **IPS-B** (sum 7–9): moderately infiltrative
  - **IPS-C** (sum 10–15): highly infiltrative
- Manual assessment is subjective (inter-observer κ = 0.526) → need automated scoring

## The Dataset

- **252 H&E-stained WSIs** from **73 patients**, Koç University Hospital
- 61 patients have 3 WSIs → used for IPS evaluation
- 12 patients have <3 WSIs → training only, never evaluated
- Grade distribution: G1(64), G2(43), G3(36), G4(34), G5(20)
- IPS distribution: IPS-A(12 patients), IPS-B(21), IPS-C(28)
- Each WSI is gigapixel-scale (often >100,000 × 100,000 pixels)

## The Core Idea

- Infiltration is a **spatial/relational** property — it's about the interaction between tumor and surrounding tissue, not isolated cell morphology
- Standard MIL (Multiple Instance Learning) treats patches independently → loses spatial relationships
- Full-graph GNNs connect all patches → introduces noise from tumor-to-tumor and stroma-to-stroma edges
- **This pipeline**: construct a **bipartite graph** at the tumor–NNP interface, with edges **only** between tumor and non-tumor patches, then use a GNN to learn from this interaction zone

---

## Stage 1: WSI → H5 Feature Files

**Tool**: Trident ([https://github.com/mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT))

### 1a. Tissue Segmentation

- **GrandQC** model detects tissue vs. glass background
- Output: GeoJSON contour polygons per slide
- Patches with <60% tissue are discarded

### 1b. Patch Extraction

- WSI tiled into **1024×1024 pixel** non-overlapping patches at **40x magnification**
- Each patch gets an **(x, y) coordinate** — the top-left corner pixel position on the WSI
- The **3** in 1024×1024×3 = RGB color channels (Red, Green, Blue)
- A typical slide yields 200–2,000 tissue patches

### 1c. Feature Extraction (VirChow2)

- **VirChow2**: pathology foundation model pre-trained on millions of histopathology images
- Input: 1024×1024×3 raw patch pixels
- Output: **1280-dimensional feature vector** per patch
- This vector encodes cell morphology, tissue structure, staining patterns — everything a pathologist sees, compressed into 1280 numbers
- Two patches with similar tissue will have similar vectors

### 1d. Patch Classification

- 3-class MLP classifier: `2560 → 256 → 128 → 3` (BatchNorm, ReLU, Dropout 0.3)
- Trained on ~45,000 patches from 20 pixel-annotated WSIs
- Labels each patch: **Stroma (0)**, **PanNET/tumor (1)**, **Normal (2)**
- Hard argmax — no soft labels. Mixed patches get whichever class has highest probability
- Accuracy: PanNET F1 97.2%, Normal F1 96.2%, Stroma F1 90.9%

### 1e. Tissue ID Assignment

- GeoJSON contours from GrandQC → merged via buffer(100px) + union + unbuffer(-100px)
- Fragments <30% of largest region's area are removed
- Each patch is spatially joined to its enclosing tissue polygon → **tissue_id** (0, 1, 2, ...)
- Only **tissue_id=0** (largest tissue piece) is used downstream

### Output: 1 H5 file per slide (252 total)

```
features      (N, 1280)   VirChow2 embeddings
coords        (N, 2)      patch (x, y) pixel coordinates
patch_classes  (N,)        0=stroma, 1=PanNET, 2=normal
tissue_id      (N,)        which tissue piece
slide_width    scalar      WSI dimensions
slide_height   scalar
```

**Files**: `src/stage1_extraction/extract_features.py`, `classify_patches.py`, `assign_tissue_ids.py`

---

## Stage 2: AutoEncoder (Feature Compression)

### Why compress?

- 1280-d features → overfitting risk on 73 patients
- Compress to **256-d** via a trained autoencoder
- Trained on ALL patches (unsupervised, no labels used) → no label leakage
- After training, encoder is **frozen** and used as a fixed projector

### Architecture

```
Encoder: 1280 → 768 → 512 → 256   (Linear → RMSNorm → GELU → Dropout 0.2)
Decoder: 256 → 512 → 768 → 1280   (Linear → RMSNorm → GELU, no dropout)
```

### Training

- **Loss**: MSE(reconstruction, input) + 1e-3 × variance_regularization
  - Variance term prevents feature collapse (encourages diverse latent representations)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Batch size**: 4096
- **Epochs**: 50, early stopping patience=10
- **Data split**: 90/10 by H5 file (not by patch — prevents data leakage)
- **Weight init**: Xavier uniform

### Output

- Checkpoint file with trained encoder weights
- ~~3M parameters total, only encoder used downstream (~~1.5M)

**Files**: `src/stage2_autoencoder/model.py`, `dataset.py`, `train.py`

---

## Stage 3: Graph Construction (The Core Innovation)

### Processing pipeline per slide

#### Step 1: Morphological Hole Filling

- Sometimes patches inside a tumor region are missing (tissue fold, staining artifact, misclassification)
- Creates a 2D binary grid (PanNET=1, else=0)
- Runs `scipy.ndimage.binary_fill_holes` — enclosed gaps get filled
- New synthetic nodes added at filled positions with zero features (removed later, only needed for correct border detection)

#### Step 2: Border Detection

- A PanNET patch is a **"border" patch** if ≥1 of its 8 immediate neighbors is non-PanNET or missing
- Uses 8-connected grid neighborhood (up, down, left, right, 4 diagonals)
- These border patches are where the tumor–NNP interface lives

```
S S S S S S
S B B B B S      B = border PanNET (touches non-PanNET neighbor)
S B T T B S      T = interior PanNET (all neighbors are PanNET)
S B T T B S      S = stroma
S B B B B S
S S S S S S
```

#### Step 3: NNP Neighbor Inclusion

- Include non-PanNET patches within **r hops** of the border (Chebyshev distance ≤ r × 1024)
- **r** = `border_distance` parameter (default: 3)
- These NNP patches become the other side of the bipartite graph

#### Step 4: Bipartite Adjacency

- Edges **only** between PanNET and non-PanNET patches (no tumor↔tumor, no stroma↔stroma)
- Two patches connected if Chebyshev distance ≤ `hop_distance` × 1024
- **Chebyshev distance** = max(|Δx|, |Δy|) — grid distance including diagonals
- Hop distance neighborhoods:
  - r=1: up to 8 neighbors
  - r=2: up to 24 neighbors
  - r=3: up to 48 neighbors

#### Step 5: Connected Component Filtering

- Remove disconnected clusters with < `sum(8×i for i in 1..hop_distance)` nodes
- At r=3: minimum 48 nodes per component
- Small isolated patches = noise, not meaningful for message passing

#### Step 6: Feature Projection

- Project features through frozen autoencoder encoder: 1280 → 256

### Why bipartite?

- Forces the GNN to learn from the tumor–NNP **interaction**, not texture
- Tumor-to-tumor edges carry no infiltration signal (homogeneous)
- Thesis proves: bipartite consistently outperforms full graph at every radius

### Output: 1 .pkl file per slide

```
x               (N, 256)    autoencoder-compressed features
edge_index      (2, E)      bipartite edges (PanNET ↔ NNP only)
y               scalar      grade label (0–4, internally)
pos             (N, 2)      patch pixel coordinates
patch_classes   (N,)        tissue type per node
border_distances (N, 1)     distance to tumor–NNP border
edge_distances  (E, 1)      L∞ distance between connected nodes / 1024
filename        string      source WSI filename
slide_width     int
slide_height    int
```

**Files**: `src/stage3_graph/morphology.py`, `adjacency.py`, `builder.py`, `build_all.py`

---

## Stage 5: GIN Training & Evaluation

### Model Architecture

#### Frozen Projector

- AutoEncoder encoder (1280 → 256), weights frozen, no gradients

#### Input Normalization

- RMSNorm(256) after projection

#### GIN Feature Extractor

- **GINConv** (Graph Isomorphism Network):
  ```
  h_v^(k) = MLP( (1+ε) · h_v^(k-1) + Σ h_u^(k-1) )
  ```
- MLP per layer: `Linear(256,256) → RMSNorm → ReLU → Dropout(0.4) → Linear(256,256) → RMSNorm → ReLU`
- Residual connection: `x = GINConv(dropout(x, 0.4)) + x`
- Default: **1 layer** (thesis shows deeper hurts at r=3)
- Why GIN? **Sum aggregation** counts tumor–NNP contact points — more invasion = more interfaces. Mean/max would normalize this signal away.

#### PanNET-Only Pooling

- `global_add_pool` over **tumor nodes only**
- After message passing, PanNET nodes have absorbed NNP context
- Pooling NNP nodes would dilute the tumor-specific signal
- Output: 1 vector of size 256 per graph (per WSI)

#### Regression Head

- `RMSNorm(256) → Linear(256,128) → RMSNorm(128) → ReLU → Dropout(0.4) → Linear(128,1)`
- Predicts a single grade on 0–4 scale

### Training Configuration


| Parameter      | Value                                            |
| -------------- | ------------------------------------------------ |
| Loss           | Huber (δ=2) — robust to noisy ordinal labels     |
| Optimizer      | AdamW (lr=1e-4, weight_decay=1e-3)               |
| Scheduler      | ReduceLROnPlateau (factor=0.8, patience=5)       |
| Batch size     | 8                                                |
| Max epochs     | 200                                              |
| Early stopping | patience=15, min_delta=0.01 on val_loss          |
| Precision      | bf16-mixed                                       |
| Sampler        | WeightedRandomSampler (1/class_count per sample) |


### Why Huber loss instead of cross-entropy?

- Grades are **ordinal** — Grade 3 is between Grade 2 and Grade 4
- Cross-entropy treats all misclassifications equally (3→1 same penalty as 3→2)
- Huber penalizes proportionally to error magnitude
- Handles noisy boundary cases (pathologist subjectivity) better than hard classification
- Thesis shows: regression >> classification across all configurations

### Model Size

- AutoEncoder (frozen): 3,021,312 params
- GIN (1 layer): 132,096 params
- Regression head: 33,665 params
- **Total trainable: 165,761 params** — very lightweight

### Cross-Validation Protocol

- **4-fold CV** using splits s1, s2, s3, s4 (rotate as test/val)
- **s5** always added to training (patients with <3 WSIs, never evaluated)
- Per fold: test=s[i], val=s[(i+1)%4], train=remaining+s5
- **25 random seeds** per configuration for statistical robustness
- Wilcoxon signed-rank test for significance

### Patient-Level IPS Evaluation

1. Model predicts per-graph grade (0–4 scale, continuous)
2. Shift to clinical scale: prediction + 1 → (1–5)
3. Per slide: average predictions across graphs, clip to [1,5], round to integer
4. Per patient (3 slides): sum the 3 slide grades
5. Map sum to IPS:
  - Gold standard (integer sum): <7 → IPS-A, <10 → IPS-B, ≥10 → IPS-C
  - Predictions (float sum): ≤6.5 → IPS-A, ≤9.5 → IPS-B, >9.5 → IPS-C

### Evaluation Metrics

- **Per-class F1**: IPS-A, IPS-B, IPS-C
- **Macro F1**: unweighted mean of per-class F1
- **Weighted F1**: class-size-weighted mean
- **QWK** (Quadratic Weighted Kappa): penalizes distant misclassifications (IPS-A→IPS-C worse than IPS-A→IPS-B)
- **MAE**: mean absolute error on grade scale

**Files**: `src/stage5_training/models/gin.py`, `regression_model.py`, `dataset.py`, `data_module.py`, `evaluation.py`, `train.py`

---

## Results (from thesis)


| Model                       | Macro F1  | QWK      | MAE      |
| --------------------------- | --------- | -------- | -------- |
| DeepSets (MIL)              | 63.1%     | 54.1     | 1.87     |
| AB-MIL                      | 63.2%     | 55.5     | 1.80     |
| CLAM (classification)       | 48.0%     | 37.0     | 2.42     |
| Patch-GCN (full graph)      | 61.4%     | 50.7     | 1.84     |
| Context-Aware MIL           | 63.3%     | 62.9     | 1.78     |
| **Bipartite GIN (1L, r=3)** | **70.7%** | **69.1** | **1.60** |


### Key Ablation Findings

- **Bipartite > Full Graph** at every radius — tumor↔tumor edges are noise
- **Regression (Huber) >> Classification (CE)** — ordinal labels need ordinal loss
- **1 layer > 2 > 3 layers** at r=3 — bipartite structure already captures the receptive field
- **r=3 > r=2 > r=1** — more spatial context consistently helps
- **GIN ≈ GATv2** — sum aggregation matches attention; attention adds complexity without benefit in low-data regime

---

## Project Structure

```
pannet_pipeline/
├── data/
│   ├── wsi_information.csv          # 252 slides: filename, case, slide, patient, grade, ips
│   └── fold_information.yaml        # s1-s5 patient ID lists for cross-validation
├── src/
│   ├── constants.py                 # PATCH_SIZE=1024, class IDs, feature dims, 25 seeds
│   ├── utils.py                     # Filename parsing, IPS grading, config loading
│   ├── stage1_extraction/           # WSI → H5 (Trident + VirChow2 + GrandQC)
│   ├── stage2_autoencoder/          # 1280→256 compression (LightningModule)
│   ├── stage3_graph/                # Bipartite graph construction → .pkl
│   └── stage5_training/             # GIN model, regression, evaluation
├── scripts/
│   ├── run_extraction.sh            # SLURM: 8 GPUs parallel, ~32 slides each
│   ├── run_autoencoder.sh           # SLURM: 1 GPU, ~5 min
│   ├── run_graph_build.sh           # SLURM: 8 GPUs parallel
│   ├── run_train.sh                 # SLURM: 8 GPUs, 100 fold×seed combos split
│   └── run_all_experiments.sh       # Full 9-config sweep (3 radii × 3 depths)
├── checkpoints/                     # Saved model weights
└── experiments/                     # Results JSON files
```

## Execution Order

```
sbatch scripts/run_extraction.sh       # Stage 1: ~1h with 8 GPUs
sbatch scripts/run_autoencoder.sh      # Stage 2: ~5 min
sbatch scripts/run_graph_build.sh      # Stage 3: ~2 min with 8 GPUs
sbatch scripts/run_train.sh            # Stage 5: ~2h with 8 GPUs (1 config)
bash scripts/run_all_experiments.sh    # Full sweep: 9 configs × above
```

