"""
STAGE 5 — Patient-level IPS evaluation.

The model predicts per-graph (per-tissue-region) grades. But the clinical
metric is patient-level IPS, computed by:
  1. Grouping predictions by patient
  2. Filtering to patients with exactly 3 slides
  3. Averaging predictions per slide (if multiple tissue regions per slide)
  4. Rounding each slide's average to the nearest integer grade (1-5)
  5. Summing the 3 slide grades → total score
  6. Mapping total to IPS: [3-6]=IPS-A, [7-9]=IPS-B, [10-15]=IPS-C

Metrics reported:
  - Per-class F1 (IPS-A, IPS-B, IPS-C)
  - Macro F1 (unweighted mean of per-class F1)
  - Weighted F1 (class-size-weighted mean)
  - QWK (Quadratic Weighted Kappa — penalizes distant misclassifications)
  - MAE (Mean Absolute Error on the grade scale)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from src.utils import parse_filename, total_to_ips_gold, total_to_ips_pred


def aggregate_to_patient_ips(
    predictions: list[float],
    targets: list[float],
    filenames: list[str],
) -> tuple[list[int], list[int]]:
    """
    Aggregate graph-level predictions to patient-level IPS categories.

    Args:
        predictions: Per-graph predicted grades (1-5 scale, may be float)
        targets: Per-graph ground-truth grades (1-5 scale, integer)
        filenames: Per-graph WSI filenames (e.g., '#1-1 7817B8509.tiff')

    Returns:
        (y_true_ips, y_pred_ips) — lists of IPS categories (0, 1, 2) for
        patients with exactly 3 slides.
    """
    # Group by patient (case_id)
    patient_slides_pred: dict[int, dict[int, list[float]]] = {}
    patient_slides_true: dict[int, dict[int, list[float]]] = {}

    for pred, target, fname in zip(predictions, targets, filenames):
        case_id, slide_id = parse_filename(fname)

        if case_id not in patient_slides_pred:
            patient_slides_pred[case_id] = {}
            patient_slides_true[case_id] = {}

        patient_slides_pred[case_id].setdefault(slide_id, []).append(pred)
        patient_slides_true[case_id].setdefault(slide_id, []).append(target)

    # For each patient with 3 slides: average per slide, round, sum → IPS
    y_true_ips = []
    y_pred_ips = []

    for case_id in sorted(patient_slides_pred.keys()):
        slides_pred = patient_slides_pred[case_id]
        slides_true = patient_slides_true[case_id]

        # Only evaluate patients with exactly 3 slides (clinical protocol)
        if len(slides_pred) < 3:
            continue

        # Take the first 3 slides (by slide_id order)
        slide_ids = sorted(slides_pred.keys())[:3]

        # Sum predicted grades (average per slide, clip to [1,5], round)
        total_pred = 0.0
        total_true = 0
        for sid in slide_ids:
            avg_pred = np.mean(slides_pred[sid])
            avg_pred = np.clip(avg_pred, 1.0, 5.0)
            total_pred += round(float(avg_pred))

            avg_true = np.mean(slides_true[sid])
            total_true += round(float(avg_true))

        y_pred_ips.append(total_to_ips_pred(total_pred))
        y_true_ips.append(total_to_ips_gold(total_true))

    return y_true_ips, y_pred_ips


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """
    Compute all evaluation metrics for patient-level IPS classification.

    Returns dict with keys: f1_ips_a, f1_ips_b, f1_ips_c, f1_macro,
    f1_weighted, qwk
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Per-class F1 scores
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=[0, 1, 2],
                            average=None, zero_division=0)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    # Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(y_true_arr, y_pred_arr, n_classes=3)

    return {
        "f1_ips_a": per_class_f1[0],
        "f1_ips_b": per_class_f1[1],
        "f1_ips_c": per_class_f1[2],
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "qwk": qwk,
        "num_patients": len(y_true),
    }


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    QWK penalizes disagreements proportionally to the squared distance
    between predicted and actual categories. A misclassification of
    IPS-A as IPS-C is penalized more heavily than IPS-A as IPS-B.

    κ = 1 - Σ(w_ij * O_ij) / Σ(w_ij * E_ij)

    where w_ij = (i-j)² / (N-1)², O = observed confusion matrix,
    E = expected (random chance) confusion matrix.
    """
    # Build confusion matrix
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    # Weight matrix: (i-j)² / (N-1)²
    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2

    # Expected matrix (outer product of marginals)
    row_sum = O.sum(axis=1)
    col_sum = O.sum(axis=0)
    total = O.sum()
    if total == 0:
        return 0.0
    E = np.outer(row_sum, col_sum) / total

    # QWK
    numerator = (W * O).sum()
    denominator = (W * E).sum()
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator
