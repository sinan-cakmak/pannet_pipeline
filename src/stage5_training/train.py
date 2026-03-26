"""
STAGE 5 entry point: Train the bipartite GIN model and evaluate.

This is the main training script. It:
  1. Loads the frozen autoencoder encoder
  2. Creates the GIN feature extractor + regression model
  3. Sets up 4-fold cross-validation data
  4. Trains with early stopping on validation loss
  5. Runs prediction on the test set
  6. Aggregates to patient-level IPS and computes metrics

Usage:
  uv run python -m src.stage5_training.train \\
      --seed 42 \\
      --test-fold 0 \\
      --graph-dir "/path/to/graphs" \\
      --ae-checkpoint "checkpoints/autoencoder/best.ckpt"

  # Full sweep (all folds, all seeds):
  for fold in 0 1 2 3; do
    for seed in 42 777 5999 ...; do
      uv run python -m src.stage5_training.train --test-fold $fold --seed $seed
    done
  done
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.stage2_autoencoder.model import AutoEncoder
from src.stage5_training.data_module import PanNETDataModule
from src.stage5_training.db import build_log_entry, log_to_db
from src.stage5_training.evaluation import aggregate_to_patient_ips, compute_metrics
from src.stage5_training.models.cell_gin import CellConditionedGIN
from src.stage5_training.models.gin import GINFeatureExtractor
from src.stage5_training.regression_model import InfiltrationModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: Train bipartite GIN model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fold", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--graph-dir", required=True, help="Directory with .pkl graph files")
    parser.add_argument("--ae-checkpoint", required=True, help="AutoEncoder checkpoint")
    parser.add_argument("--fold-config", default="data/fold_information.yaml")
    parser.add_argument("--num-gnn-layers", type=int, default=1)
    parser.add_argument("--hop-distance", type=int, default=3)
    parser.add_argument("--border-distance", type=int, default=3)
    parser.add_argument("--cell-info-mode", default="none", choices=["none", "concat", "gate"],
                        help="How to use cell_information: none=ignore, concat=append, gate=cell-conditioned conv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if this (fold, seed, layers, hop) combo already has results")
    args = parser.parse_args()

    # ---- Skip if already run ----
    if args.skip_existing:
        experiment_name = f"gin_{args.num_gnn_layers}layer_{args.cell_info_mode}_fold{args.test_fold}_seed{args.seed}"
        json_path = Path(args.output_dir) / "results" / f"{experiment_name}.json"
        if json_path.exists():
            print(f"SKIP: {experiment_name} (JSON exists)")
            return
        # Also check DB
        try:
            import psycopg2
            from src.stage5_training.db import CONN_STRING
            conn = psycopg2.connect(CONN_STRING)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM bipartite_experiments "
                "WHERE fold=%s AND seed=%s AND num_gnn_layers=%s AND hop_distance=%s AND cell_info_mode=%s",
                (args.test_fold, args.seed, args.num_gnn_layers, args.hop_distance, args.cell_info_mode),
            )
            if cur.fetchone()[0] > 0:
                print(f"SKIP: {experiment_name} (already in DB)")
                cur.close()
                conn.close()
                return
            cur.close()
            conn.close()
        except Exception:
            pass  # DB unreachable, fall through to JSON check only

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    # ---- Load frozen autoencoder encoder ----
    print(f"Loading autoencoder from: {args.ae_checkpoint}")
    autoencoder = AutoEncoder.load_from_checkpoint(
        args.ae_checkpoint, map_location="cpu"
    )
    projector = autoencoder.encoder.eval()
    for p in projector.parameters():
        p.requires_grad = False

    # ---- Create model ----
    if args.cell_info_mode == "gate":
        feature_extractor = CellConditionedGIN(num_layers=args.num_gnn_layers)
    else:
        feature_extractor = GINFeatureExtractor(num_layers=args.num_gnn_layers)

    model = InfiltrationModel(
        projector=projector,
        feature_extractor=feature_extractor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cell_info_mode=args.cell_info_mode,
    )

    # ---- Data ----
    data_module = PanNETDataModule(
        graph_dir=args.graph_dir,
        fold_config_path=args.fold_config,
        test_fold=args.test_fold,
        hop_distance=args.hop_distance,
        border_distance=args.border_distance,
        batch_size=args.batch_size,
    )

    # ---- Callbacks ----
    experiment_name = f"gin_{args.num_gnn_layers}layer_{args.cell_info_mode}_fold{args.test_fold}_seed{args.seed}"
    checkpoint_dir = Path(args.output_dir) / "checkpoints" / experiment_name

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, min_delta=0.01, mode="min"),
        checkpoint_cb,
    ]

    # ---- Train ----
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision="bf16-mixed",
        accelerator=args.device,
        devices=1,
        log_every_n_steps=1,
        logger=False,  # We log to PostgreSQL, not CSV
    )

    trainer.fit(model, datamodule=data_module)

    # ---- Test: predict on held-out test set ----
    print("\nRunning predictions on test set...")
    predictions_list = trainer.predict(model, datamodule=data_module)

    # Flatten prediction batches
    all_preds = []
    all_targets = []
    all_filenames = []
    for batch_result in predictions_list:
        all_preds.extend(batch_result["predictions"].cpu().tolist())
        all_targets.extend(batch_result["targets"].cpu().tolist())
        all_filenames.extend(batch_result["filenames"])

    # ---- Patient-level IPS evaluation ----
    y_true_ips, y_pred_ips = aggregate_to_patient_ips(all_preds, all_targets, all_filenames)

    if len(y_true_ips) == 0:
        print("WARNING: No patients with 3 slides found in test set.")
        return

    metrics = compute_metrics(y_true_ips, y_pred_ips)

    # ---- Report results ----
    print(f"\n{'='*60}")
    print(f"Results: fold={args.test_fold}, seed={args.seed}, layers={args.num_gnn_layers}")
    print(f"{'='*60}")
    print(f"  Patients evaluated: {metrics['num_patients']}")
    print(f"  F1 IPS-A: {metrics['f1_ips_a']:.4f}")
    print(f"  F1 IPS-B: {metrics['f1_ips_b']:.4f}")
    print(f"  F1 IPS-C: {metrics['f1_ips_c']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"  QWK: {metrics['qwk']:.4f}")
    print(f"{'='*60}")

    # ---- Log to PostgreSQL (fallback to JSON) ----
    best_ckpt = checkpoint_cb.best_model_path or ""
    best_val_loss = checkpoint_cb.best_model_score
    best_val_loss = float(best_val_loss) if best_val_loss is not None else None

    log_entry = build_log_entry(
        seed=args.seed,
        test_fold=args.test_fold,
        num_gnn_layers=args.num_gnn_layers,
        hop_distance=args.hop_distance,
        border_distance=args.border_distance,
        cell_info_mode=args.cell_info_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        epochs_trained=trainer.current_epoch,
        val_loss=best_val_loss,
        metrics=metrics,
        model_path=best_ckpt,
    )

    db_logged = log_to_db(log_entry)

    # Fallback: save as JSON if DB fails
    if not db_logged:
        results_dir = Path(args.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{experiment_name}.json"
        json_data = {k: (str(v) if not isinstance(v, (int, float, str, type(None))) else v)
                     for k, v in log_entry.items()}
        with open(results_file, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"[Fallback] Results saved to: {results_file}")


if __name__ == "__main__":
    main()
