"""
Migrate all JSON result files to PostgreSQL.

Usage:
  uv run python scripts/migrate_json_to_db.py
  uv run python scripts/migrate_json_to_db.py --results-dir experiments/results
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.stage5_training.db import CONN_STRING, CREATE_TABLE_SQL, INSERT_SQL


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON results to PostgreSQL")
    parser.add_argument("--results-dir", default="experiments/results")
    args = parser.parse_args()

    import psycopg2

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {results_dir}")

    if not json_files:
        return

    conn = psycopg2.connect(CONN_STRING)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)

    migrated = 0
    skipped = 0

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        # Parse timestamp if it's a string, otherwise use now
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now(timezone.utc)
        elif ts is None:
            ts = datetime.now(timezone.utc)

        try:
            entry = {
                "run_id": data.get("run_id", jf.stem),
                "timestamp": ts,
                "fold": int(data.get("fold", data.get("test_fold", 0))),
                "seed": int(data.get("seed", 0)),
                "num_gnn_layers": int(data.get("num_gnn_layers", 1)),
                "hop_distance": int(data.get("hop_distance", 3)),
                "border_distance": int(data.get("border_distance", 3)),
                "lr": float(data.get("lr", 1e-4)),
                "weight_decay": float(data.get("weight_decay", 1e-3)),
                "batch_size": int(data.get("batch_size", 8)),
                "max_epochs": int(data.get("max_epochs", 200)),
                "epochs_trained": int(data["epochs_trained"]) if data.get("epochs_trained") else None,
                "val_loss": float(data["val_loss"]) if data.get("val_loss") else None,
                "f1_ips_a": float(data.get("f1_ips_a", 0)),
                "f1_ips_b": float(data.get("f1_ips_b", 0)),
                "f1_ips_c": float(data.get("f1_ips_c", 0)),
                "f1_macro": float(data.get("f1_macro", 0)),
                "f1_weighted": float(data.get("f1_weighted", 0)),
                "qwk": float(data.get("qwk", 0)),
                "num_test_patients": int(data.get("num_test_patients", data.get("num_patients", 0))),
                "model_path": data.get("model_path", ""),
            }
            cur.execute(INSERT_SQL, entry)
            migrated += 1
        except Exception as e:
            print(f"  SKIP {jf.name}: {e}")
            skipped += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Migrated {migrated} results to PostgreSQL, skipped {skipped}")


if __name__ == "__main__":
    main()
