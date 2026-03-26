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
from src.stage5_training.evaluation import aggregate_to_patient_ips, compute_metrics
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()

    L.seed_everything(args.seed)

    # ---- Load frozen autoencoder encoder ----
    print(f"Loading autoencoder from: {args.ae_checkpoint}")
    autoencoder = AutoEncoder.load_from_checkpoint(
        args.ae_checkpoint, map_location="cpu"
    )
    projector = autoencoder.encoder.eval()
    for p in projector.parameters():
        p.requires_grad = False

    # ---- Create model ----
    feature_extractor = GINFeatureExtractor(num_layers=args.num_gnn_layers)
    model = InfiltrationModel(
        projector=projector,
        feature_extractor=feature_extractor,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- Data ----
    data_module = PanNETDataModule(
        graph_dir=args.graph_dir,
        fold_config_path=args.fold_config,
        test_fold=args.test_fold,
        batch_size=args.batch_size,
    )

    # ---- Callbacks ----
    experiment_name = f"gin_{args.num_gnn_layers}layer_fold{args.test_fold}_seed{args.seed}"
    checkpoint_dir = Path(args.output_dir) / "checkpoints" / experiment_name

    callbacks = [
        # Stop training if val_loss doesn't improve for 15 epochs
        EarlyStopping(monitor="val_loss", patience=15, min_delta=0.01, mode="min"),
        # Save only the best model (lowest val_loss)
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    # ---- Train ----
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision="bf16-mixed",
        accelerator=args.device,
        devices=1,
        log_every_n_steps=1,
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
    ips_names = {0: "IPS-A", 1: "IPS-B", 2: "IPS-C"}
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

    # Save results as JSON
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{experiment_name}.json"

    results = {
        "seed": args.seed,
        "test_fold": args.test_fold,
        "num_gnn_layers": args.num_gnn_layers,
        **metrics,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
