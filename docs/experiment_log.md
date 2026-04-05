# Experiment Log

Tracks every cell-information trial: what changed, why, results, and takeaways.

## Baseline: No Cell Information

- **ID:** `none_4d`
- **Config:** `cell_info_mode=none`, `cell_info_dim=4`
- **Description:** Standard bipartite GIN (1 layer, r=3) with no cell composition information. Pure VirChow2 morphological features (1280-d -> 256-d via autoencoder). This is the thesis baseline.
- **Runs:** 4 folds x 5 seeds = 20

| Metric | Value |
|--------|-------|
| Macro F1 | 0.6445 |
| Weighted F1 | 0.6850 |
| QWK | 0.6786 |
| F1 IPS-1 | 0.5732 |
| F1 IPS-2 | 0.5495 |
| F1 IPS-3 | 0.8109 |

---

## Trial 1: Gate with HoVer-Net Cell Counts (4-dim)

- **ID:** `gate_4d`
- **Config:** `cell_info_mode=gate`, `cell_info_dim=4`
- **Description:** CellConditionedConv gates message passing using HoVer-Net nucleus counts. Each patch gets a 4-dim vector: [neoplastic, inflammatory, other, reserved]. Gate network: MLP(8->256->256) with Sigmoid. Only 73% of slides had cell data (27% zero-filled from missing HoVer-Net extractions via Nusret's `patch_cell_types_v2`).
- **Cell info source:** `patch_cell_types_v2` from H5 files (Nusret's prior HoVer-Net runs)
- **Runs:** 4 folds x 5 seeds = 20

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Macro F1 | 0.6032 | -0.0413 |
| Weighted F1 | 0.6416 | -0.0434 |
| QWK | 0.5916 | -0.0870 |
| F1 IPS-1 | 0.5137 | -0.0595 |
| F1 IPS-2 | 0.5398 | -0.0097 |
| F1 IPS-3 | 0.7562 | -0.0547 |

**Takeaway:** Gate mechanism hurts across the board. Possible causes: 27% missing data (zero-filled adds noise), 4-dim vector too coarse, extra parameters cause overfitting, or VirChow2 features already encode cell composition.

---

## Trial 2: Gate with GigaTIME Virtual Protein Counts (21-dim, raw)

- **ID:** `gate_21d`
- **Config:** `cell_info_mode=gate`, `cell_info_dim=21`
- **Description:** Replaced HoVer-Net with GigaTIME (Microsoft, Cell 2025) virtual multiplex immunofluorescence. Each 1024x1024 patch is split into 16 non-overlapping 256x256 tiles, GigaTIME predicts 23-channel protein masks, pixel counts summed across tiles for 21 functional channels. 100% slide coverage (no missing data). Gate network: MLP(42->256->256) with Sigmoid. Raw pixel counts used (no normalization).
- **Cell info source:** GigaTIME .npy files in `gigatime_features/`
- **GigaTIME channels:** DAPI, PD-1, CD14, CD4, T-bet, CD34, CD68, CD16, CD11c, CD138, CD20, CD3, CD8, PD-L1, CK, Ki67, Tryptase, Actin-D, Caspase3-D, PHH3-B, Transgelin
- **Runs:** 4 folds x 5 seeds = 20

| Metric | Value | vs Baseline | vs Gate-4d |
|--------|-------|-------------|------------|
| Macro F1 | 0.5771 | -0.0674 | -0.0261 |
| Weighted F1 | 0.6303 | -0.0547 | -0.0113 |
| QWK | 0.6117 | -0.0669 | +0.0201 |
| F1 IPS-1 | 0.4968 | -0.0764 | -0.0169 |
| F1 IPS-2 | 0.4785 | -0.0710 | -0.0613 |
| F1 IPS-3 | 0.7561 | -0.0548 | -0.0001 |

**Takeaway:** Worse than both baseline and gate-4d on F1, though slightly better than gate-4d on QWK. Root cause identified: **massive scale mismatch** in raw pixel counts. Channel values range from ~67 (Tryptase) to ~217K (CK) — a 3000x spread. The sigmoid gate is dominated by high-count channels (CK, DAPI, Caspase3-D) while biologically important immune markers (CD3=4.7K, CD8=14.7K) are effectively invisible. Also, Caspase3-D (178K avg) is likely a GigaTIME artifact on PanNET tissue.

---

## Trial 3: Gate with GigaTIME + Log Normalization (21-dim, log1p)

- **ID:** `gate_21d_log`
- **Config:** `cell_info_mode=gate`, `cell_info_dim=21`, log1p normalization applied
- **Description:** Same as Trial 2 but applies `log1p()` to cell_information before passing to the gate network. This compresses the scale: 67->4.2, 4723->8.5, 216568->12.3 — all within a comparable range. The gate can now distinguish between channels meaningfully.
- **Status:** PENDING
- **Hypothesis:** Log normalization will fix the scale mismatch and allow the gate to learn meaningful per-edge protein-based modulation.

---

## Future Trials (Planned)

- **concat_21d_log:** Concat mode with GigaTIME 21-dim + log1p (simpler than gate, may overfit less)
- **gate_selected_channels:** Gate with only 8-10 immune/tumor channels (drop Caspase3-D, structural markers)
- **gate_smaller_mlp:** Reduce gate MLP from (42->256->256) to (42->64->256) to reduce overfitting
