"""Quick sanity check for GigaTIME extraction outputs."""

import numpy as np
from pathlib import Path

NAMES = [
    "DAPI", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16",
    "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
    "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin",
]

npy_dir = Path("gigatime_features")
files = sorted(npy_dir.glob("*.npy"))
print(f"Total .npy files: {len(files)}")
print(f"Expected: 252 (one per slide)\n")

# Aggregate stats across all files
all_means = []
total_patches = 0
all_zero_files = []

for f in files:
    data = np.load(f)
    total_patches += len(data)
    all_means.append(data.mean(axis=0))
    if data.max() == 0:
        all_zero_files.append(f.name)

print(f"Total patches across all slides: {total_patches}")
print(f"All-zero files (failed extraction): {len(all_zero_files)}")
if all_zero_files:
    for name in all_zero_files[:10]:
        print(f"  - {name}")

# Global per-channel averages
global_mean = np.mean(all_means, axis=0)
print(f"\n{'='*55}")
print(f"{'Channel':15s} {'Avg pixel count':>15s}  Interpretation")
print(f"{'='*55}")
for i, name in enumerate(NAMES):
    val = global_mean[i]
    note = ""
    if name == "DAPI":
        note = "nuclei (should be highest)"
    elif name == "CK":
        note = "tumor epithelium"
    elif name in ("CD3", "CD8"):
        note = "T cells (key for IPS)"
    elif name == "CD68":
        note = "macrophages"
    elif name == "CD20":
        note = "B cells"
    elif name == "PD-L1":
        note = "immune checkpoint"
    elif name == "Ki67":
        note = "proliferation"
    print(f"  {name:12s}  {val:15.1f}  {note}")

# Show 3 example slides in detail
print(f"\n{'='*55}")
print("Sample slides:")
print(f"{'='*55}")
for f in files[:3]:
    data = np.load(f)
    nonzero = (data.sum(axis=1) > 0).sum()
    print(f"\n{f.name}")
    print(f"  shape: {data.shape}, patches with signal: {nonzero}/{len(data)}")
    print(f"  Top channels (mean across patches):")
    means = data.mean(axis=0)
    ranked = np.argsort(means)[::-1]
    for idx in ranked[:5]:
        print(f"    {NAMES[idx]:12s}: {means[idx]:.1f} pixels/patch")
