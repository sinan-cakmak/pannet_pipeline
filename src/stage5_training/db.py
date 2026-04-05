"""
Database logging for experiment results.

Logs to a Neon PostgreSQL database (table: bipartite_experiments).
Falls back to local JSON if the database is unreachable.
"""

from __future__ import annotations

from datetime import datetime, timezone


CONN_STRING = (
    "postgresql://neondb_owner:npg_jqTonk92afDE"
    "@ep-bitter-flower-aixibz5q-pooler.c-4.us-east-1.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require&connect_timeout=10"
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bipartite_experiments (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    fold SMALLINT NOT NULL,
    seed INTEGER NOT NULL,
    num_gnn_layers SMALLINT NOT NULL,
    hop_distance SMALLINT NOT NULL,
    border_distance SMALLINT NOT NULL,
    cell_info_mode TEXT DEFAULT 'none',
    cell_info_dim SMALLINT DEFAULT 4,
    log_normalize BOOLEAN DEFAULT FALSE,
    lr REAL NOT NULL,
    weight_decay REAL NOT NULL,
    batch_size SMALLINT NOT NULL,
    max_epochs SMALLINT NOT NULL,
    epochs_trained SMALLINT,
    val_loss REAL,
    f1_ips_a REAL,
    f1_ips_b REAL,
    f1_ips_c REAL,
    f1_macro REAL,
    f1_weighted REAL,
    qwk REAL,
    num_test_patients SMALLINT,
    model_path TEXT
)
"""

INSERT_SQL = """
INSERT INTO bipartite_experiments (
    run_id, timestamp, fold, seed, num_gnn_layers,
    hop_distance, border_distance, cell_info_mode, cell_info_dim, log_normalize,
    lr, weight_decay, batch_size, max_epochs, epochs_trained, val_loss,
    f1_ips_a, f1_ips_b, f1_ips_c, f1_macro, f1_weighted,
    qwk, num_test_patients, model_path
) VALUES (
    %(run_id)s, %(timestamp)s, %(fold)s, %(seed)s, %(num_gnn_layers)s,
    %(hop_distance)s, %(border_distance)s, %(cell_info_mode)s, %(cell_info_dim)s, %(log_normalize)s,
    %(lr)s, %(weight_decay)s, %(batch_size)s, %(max_epochs)s, %(epochs_trained)s, %(val_loss)s,
    %(f1_ips_a)s, %(f1_ips_b)s, %(f1_ips_c)s, %(f1_macro)s, %(f1_weighted)s,
    %(qwk)s, %(num_test_patients)s, %(model_path)s
)
"""


def log_to_db(results: dict) -> bool:
    """
    Log experiment results to PostgreSQL. Returns True on success.

    The table 'bipartite_experiments' is created automatically if it doesn't exist.
    """
    try:
        import psycopg2

        conn = psycopg2.connect(CONN_STRING)
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        cur.execute(INSERT_SQL, results)
        conn.commit()
        cur.close()
        conn.close()
        print("[DB] Results logged to PostgreSQL (bipartite_experiments).")
        return True
    except Exception as e:
        print(f"[DB] Failed to log to PostgreSQL: {e}")
        return False


def build_log_entry(
    seed: int,
    test_fold: int,
    num_gnn_layers: int,
    hop_distance: int,
    border_distance: int,
    batch_size: int,
    max_epochs: int,
    epochs_trained: int | None,
    val_loss: float | None,
    metrics: dict,
    model_path: str,
    cell_info_mode: str = "none",
    cell_info_dim: int = 4,
    log_normalize: bool = False,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
) -> dict:
    """Build a dict matching the DB columns."""
    return {
        "run_id": f"bipartite_gin_{num_gnn_layers}L_hop{hop_distance}_{cell_info_mode}_{cell_info_dim}d_fold{test_fold}_seed{seed}",
        "timestamp": datetime.now(timezone.utc),
        "fold": test_fold,
        "seed": seed,
        "num_gnn_layers": num_gnn_layers,
        "hop_distance": hop_distance,
        "border_distance": border_distance,
        "cell_info_mode": cell_info_mode,
        "cell_info_dim": cell_info_dim,
        "log_normalize": log_normalize,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "epochs_trained": epochs_trained,
        "val_loss": val_loss,
        "f1_ips_a": metrics.get("f1_ips_a", 0.0),
        "f1_ips_b": metrics.get("f1_ips_b", 0.0),
        "f1_ips_c": metrics.get("f1_ips_c", 0.0),
        "f1_macro": metrics.get("f1_macro", 0.0),
        "f1_weighted": metrics.get("f1_weighted", 0.0),
        "qwk": metrics.get("qwk", 0.0),
        "num_test_patients": metrics.get("num_patients", 0),
        "model_path": model_path,
    }
