# How the GIN Model Works — From Graph to Patient Prediction

This document explains Stage 5 of the pipeline step by step: how a bipartite graph turns into an infiltration pattern score. No prior GNN knowledge needed.

---

## What You Already Know

At this point you have a **bipartite graph** per tissue region:

- **Nodes**: patches at the tumor-NNP border, each with a 256-dim feature vector (from the autoencoder)
- **Edges**: only connect PanNET (tumor) patches to NNP (non-neoplastic) patches — never tumor-to-tumor or NNP-to-NNP
- **Label**: a grade (0-4 internally, corresponding to clinical grades 1-5)

The question is: **how do you turn this graph into a single number (the predicted grade)?**

---

## Step 1: Message Passing — The Core Idea

### The Intuition

Imagine you're a tumor patch sitting at the border. You can "see" the NNP patches around you through the edges. A highly infiltrative tumor (grade 5) has many irregular contact points with surrounding tissue, while a well-demarcated tumor (grade 1) has few, clean edges.

**Message passing** is how each node collects information from its neighbors. After one round, every tumor node "knows" about the NNP patches it touches, and every NNP node "knows" about the tumor patches it touches.

### What Happens Concretely

For **every node** in the graph, we:

1. **Collect** the feature vectors of all its neighbors
2. **Sum** them up
3. **Add** its own features
4. **Transform** the result through a small neural network (MLP)

Written out for a single node `v`:

```
new_features(v) = MLP( features(v) + SUM of features(u) for all neighbors u of v )
```

That's it. That's GIN.

### A Concrete Example

Say tumor node T has three NNP neighbors: N1, N2, N3. Each has a 256-dim feature vector.

```
Step 1: Sum the neighbors
  neighbor_sum = features(N1) + features(N2) + features(N3)

  This is a 256-dim vector. If T had 10 neighbors instead of 3,
  the sum would be BIGGER in magnitude. This is important —
  it means more contact = larger signal.

Step 2: Add own features
  combined = features(T) + neighbor_sum

Step 3: Transform through MLP
  new_features(T) = MLP(combined)
```

After this, `new_features(T)` encodes:

- What T itself looks like (its own tissue appearance)
- How many NNP patches it touches (magnitude of the sum)
- What those NNP patches look like (their combined appearance)

### The MLP Inside GINConv

The MLP is a small 2-layer network:

```
Input (256-dim)
  → Linear layer (256 → 256)     multiply by a weight matrix + add bias
  → RMSNorm                      normalize the scale
  → ReLU                         zero out negatives: max(0, x)
  → Dropout (40%)                randomly zero out 40% of values (training only)
  → Linear layer (256 → 256)     another weight matrix + bias
  → RMSNorm
  → ReLU
Output (256-dim)
```

The MLP's weights are **learned during training**. It learns WHAT to extract from the sum of neighbors. For example, it might learn that certain patterns of tumor-NNP contact indicate higher infiltration.

### Residual Connection

After the MLP, we add the **original** features back:

```
final_features(v) = MLP(features(v) + neighbor_sum) + features(v)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
                     what the GIN layer computed          skip connection
```

Why? If the GIN layer accidentally learns something unhelpful, the network can fall back on the original features. It makes training more stable — the network only needs to learn the **difference** (residual) that message passing adds, not reconstruct the features from scratch.

### Why SUM and Not AVERAGE?

This is the key insight behind GIN and why it works for this specific problem.

**Average pooling (what GAT/GCN do):**

```
Node A: 3 neighbors, avg of their features = some vector v
Node B: 30 neighbors, avg of their features = some vector v

→ A and B look THE SAME after aggregation!
```

**Sum pooling (what GIN does):**

```
Node A: 3 neighbors, sum of their features = small vector
Node B: 30 neighbors, sum of their features = MUCH LARGER vector

→ B's result is 10x larger, reflecting its 10x more contact
```

For infiltration scoring, the **number** of tumor-NNP contacts matters. A grade 5 tumor has many fragmented contact points (more edges, larger sums). GIN preserves this count. GAT would normalize it away.

### Why Only 1 Layer?

With 1 layer at radius r=3:

- Tumor nodes hear from their NNP neighbors (up to 3 hops away)
- That's already the full border context

With 2 layers:

- In layer 1: tumor hears from NNP
- In layer 2: tumor hears from NNP's updated features (which now contain info from tumor nodes)

This creates a feedback loop that can blur the signal. The thesis showed diminishing returns: **1 layer > 2 layers > 3 layers** at r=3.

---

## Step 2: PanNET-Only Pooling — Graph to Vector

After message passing, we have updated features for every node. But we need a **single vector per graph** (per tissue region) to feed into the regression head.

### What Pooling Does

**Pooling** = combine all node features into one vector.

The simplest approach would be to sum ALL nodes. But this pipeline does something smarter:

```
graph_embedding = SUM of features(v) for all tumor nodes v only
                  (NNP nodes are thrown away at this point)
```

### Why Only Tumor Nodes?

After message passing, each tumor node's features already contain information about its NNP neighbors. The NNP information is **absorbed into the tumor representations**. Including NNP nodes in the pool would:

1. **Double-count** NNP information (once directly, once through tumor nodes)
2. **Dilute** the tumor-specific signal with stroma/normal tissue features that are less relevant to grading

Think of it this way: you want to judge how infiltrative the tumor is. You ask each tumor node "what does your border look like?" — that answer already incorporates the NNP context. You don't also need to ask the NNP nodes.

### The Resulting Vector

After pooling, you have one **256-dim vector** per graph. This vector encodes:

- How many tumor nodes are at the border (sum magnitude)
- What the tumor-NNP interface looks like overall
- The aggregate border complexity

---

## Step 3: Regression Head — Vector to Grade

The 256-dim graph embedding is fed through a small MLP to produce a single number:

```
Input: graph embedding (256-dim)
  → RMSNorm(256)                  normalize
  → Linear(256 → 128)             compress
  → RMSNorm(128)                  normalize
  → ReLU                          max(0, x)
  → Dropout(40%)                  regularization
  → Linear(128 → 1)               single output neuron
Output: predicted grade (1 number, roughly 0–4)
```

The output is a **continuous number** (e.g., 2.37), not a discrete class. This is regression, not classification.

### Why Regression Instead of Classification?

The grades (1-5) are **ordinal** — grade 3 is between grade 2 and grade 4. If you treat this as classification:

- Predicting grade 2 when the truth is grade 3 → WRONG
- Predicting grade 5 when the truth is grade 3 → EQUALLY WRONG

But clearly the second error is much worse! Classification treats all wrong answers equally.

With regression, the model learns that grade 3 is numerically between 2 and 4. The loss naturally penalizes a 2-grade error more than a 1-grade error.

The thesis showed regression with Huber loss **significantly outperforms** cross-entropy classification (Table 5.4).

---

## Step 4: Huber Loss — How the Model Learns

During training, the model makes a prediction and gets penalized based on how far off it is. The **Huber loss** (with delta=2) combines the best of two worlds:

### For Small Errors (error within 2 grades)

Behaves like **squared error** (MSE):

```
loss = 0.5 * error^2

Examples:
  predicted 2.5, actual 3.0 → error = 0.5 → loss = 0.125
  predicted 1.0, actual 3.0 → error = 2.0 → loss = 2.0
```

Squared error gives smooth gradients near zero, helping the model fine-tune.

### For Large Errors (error beyond 2 grades)

Behaves like **absolute error** (MAE):

```
loss = delta * (|error| - 0.5 * delta) = 2 * (|error| - 1)

Examples:
  predicted 0.0, actual 3.0 → error = 3.0 → loss = 4.0
  predicted 0.0, actual 4.0 → error = 4.0 → loss = 6.0
```

This prevents **outliers from dominating training**. Since label noise is inherent (inter-observer kappa = 0.526), some labels may be off by 1-2 grades. Huber loss is robust to this.

### Visual Intuition

```
loss
  |         Huber (delta=2)
  |        /
  |       /
  |      /  ← linear growth (large errors)
  |     /
  |    /
  |  _/    ← quadratic growth (small errors)
  | /
  |/_____________________________ error
  0    1    2    3    4
            ^
            delta=2 (transition point)
```

---

## Step 5: From Predicted Grade to Patient IPS

### The Problem

The model predicts a **grade per tissue region** (0-4 internal scale). But the clinical metric is **IPS per patient** (A, B, or C), computed from 3 slides.

### Step by Step

**1. Shift from internal to clinical scale:**

```
clinical_grade = predicted_grade + 1    (0-4 → 1-5)
```

**2. Group predictions by patient:**

```
Patient #1:
  Slide 1, Region A: predicted 2.3
  Slide 1, Region B: predicted 2.7  ← multiple regions per slide possible
  Slide 2, Region A: predicted 3.1
  Slide 3, Region A: predicted 1.8
```

**3. Average per slide, then round:**

```
Slide 1: avg(2.3, 2.7) = 2.5 → round → 3 (grade 3 = moderate infiltration)
Slide 2: avg(3.1) = 3.1       → round → 3
Slide 3: avg(1.8) = 1.8       → round → 2
```

**4. Sum the 3 slide grades:**

```
total = 3 + 3 + 2 = 8
```

**5. Map total to IPS category:**

```
For predictions (slightly shifted thresholds to account for rounding):
  total <= 6.5  →  IPS-A  (non-infiltrative)
  total <= 9.5  →  IPS-B  (moderate)
  total >  9.5  →  IPS-C  (highly infiltrative)

For ground truth:
  total <  7    →  IPS-A
  total < 10    →  IPS-B
  total >= 10   →  IPS-C

Our patient: total = 8 → IPS-B
```

### Why Different Thresholds for Prediction vs Ground Truth?

Ground truth grades are integers (always exactly 1, 2, 3, 4, or 5). Predicted grades are rounded from continuous values, which can introduce small biases. The slightly shifted thresholds (6.5 and 9.5 instead of 7 and 10) account for this.

---

## Step 6: Evaluation — How We Know It Works

### 4-Fold Cross-Validation

The 73 patients are split into 5 groups (s1-s5). S5 is always in training. The other 4 rotate:

```
Fold 0: test=s1, val=s2, train=s3+s4+s5
Fold 1: test=s2, val=s3, train=s1+s4+s5
Fold 2: test=s3, val=s4, train=s1+s2+s5
Fold 3: test=s4, val=s1, train=s2+s3+s5
```

Every patient gets tested exactly once across the 4 folds.

### 25 Seeds

Each fold is trained 25 times with different random seeds (weight initialization, data shuffling). This gives 25 predictions per patient. We report the **mean and standard deviation** across seeds to show the result isn't a lucky initialization.

Total training runs: 4 folds x 25 seeds = **100 models**.

### Metrics

**Per-class F1 Score** (IPS-A, IPS-B, IPS-C):

```
F1 = 2 * (precision * recall) / (precision + recall)

precision = true positives / (true positives + false positives)
    "of all patients I predicted as IPS-B, how many actually are IPS-B?"

recall = true positives / (true positives + false negatives)
    "of all actual IPS-B patients, how many did I correctly predict?"
```

**Macro F1** = average of the three per-class F1 scores (treats each class equally, even if IPS-C has very few patients).

**Weighted F1** = weighted average by class size (gives more weight to classes with more patients).

**QWK (Quadratic Weighted Kappa):**

Measures agreement between predicted and true IPS, **penalizing distant errors more**:

```
kappa = 1 - (weighted observed disagreement) / (weighted expected disagreement)

Weight for confusing class i with class j:
  w(i,j) = (i - j)^2 / (num_classes - 1)^2

Examples:
  Predict IPS-A, true IPS-B → weight = (0-1)^2 / (3-1)^2 = 1/4 = 0.25
  Predict IPS-A, true IPS-C → weight = (0-2)^2 / (3-1)^2 = 4/4 = 1.0  (4x worse!)
```

QWK ranges from -1 to 1:

- 1.0 = perfect agreement
- 0.0 = agreement no better than random chance
- <0 = worse than random

### Thesis Results (Target)


| Model                       | Macro F1  | QWK      |
| --------------------------- | --------- | -------- |
| DeepSets (MIL)              | 63.1%     | 54.1     |
| AB-MIL (attention)          | 63.2%     | 55.5     |
| Patch-GCN (full graph)      | 61.4%     | 50.7     |
| Context-Aware MIL           | 63.3%     | 62.9     |
| **Bipartite GIN (1L, r=3)** | **70.7%** | **69.1** |


The bipartite GIN beats everything by ~7% macro F1. The key insight: by restricting edges to only the tumor-NNP interface and using sum aggregation, the model focuses on exactly the right biological signal — the infiltration boundary pattern.

---

## Putting It All Together — One Forward Pass

```
INPUT: A bipartite graph with 150 nodes (80 tumor, 70 NNP), 320 edges
       Each node has a 256-dim feature vector

1. NORMALIZE
   Each node's 256-dim vector → RMSNorm → still 256-dim
   (makes magnitudes consistent across graphs)

2. GIN MESSAGE PASSING (1 round)
   For each of the 150 nodes:
     neighbor_sum = sum of all neighbor feature vectors
     combined = own features + neighbor_sum
     new_features = MLP(combined) + own features   ← residual

   Now: each tumor node "knows about" its NNP neighbors
         each NNP node "knows about" its tumor neighbors

3. PANNET-ONLY POOLING
   Take only the 80 tumor nodes
   Sum their 256-dim vectors → one 256-dim vector for the whole graph

   This 256-dim vector IS the graph embedding.
   It encodes: border complexity, contact patterns, tissue appearance at the interface.

4. REGRESSION HEAD
   256-dim → Linear → 128-dim → ReLU → Dropout → Linear → 1 number

   Output: 2.37 (predicted grade on 0-4 scale)

5. LOSS (training only)
   True grade: 3 (moderately infiltrative)
   Error: |2.37 - 3| = 0.63
   Huber loss: 0.5 * 0.63^2 = 0.198   (small error → MSE-like)

   Backpropagate this loss → update MLP weights, GINConv MLP weights,
   and regression head weights. The autoencoder stays frozen.

6. PREDICTION (inference)
   predicted_grade = 2.37 + 1 = 3.37 (shift to clinical 1-5 scale)
   This gets rounded to 3 and summed with the other 2 slides
   to compute the patient's IPS.
```

