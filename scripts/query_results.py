"""Query experiment results from the database, aggregated across folds and seeds."""

import argparse
import psycopg2
from src.stage5_training.db import CONN_STRING

SQL = """
SELECT
    ROUND(AVG(f1_macro)::numeric, 4)    AS "Macro F1",
    ROUND(AVG(f1_weighted)::numeric, 4) AS "Weighted F1",
    ROUND(AVG(qwk)::numeric, 4)        AS "QWK",
    ROUND(AVG(f1_ips_a)::numeric, 4)   AS "F1 IPS-1",
    ROUND(AVG(f1_ips_b)::numeric, 4)   AS "F1 IPS-2",
    ROUND(AVG(f1_ips_c)::numeric, 4)   AS "F1 IPS-3",
    COUNT(*)                            AS "Runs"
FROM bipartite_experiments
WHERE cell_info_mode = %s
  AND cell_info_dim = %s
  AND log_normalize = %s
"""

def main():
    parser = argparse.ArgumentParser(description="Query aggregated experiment results")
    parser.add_argument("--cell-info-mode", required=True, choices=["none", "concat", "gate"])
    parser.add_argument("--cell-info-dim", type=int, default=4)
    parser.add_argument("--log-normalize", action="store_true")
    args = parser.parse_args()

    conn = psycopg2.connect(CONN_STRING)
    cur = conn.cursor()
    cur.execute(SQL, (args.cell_info_mode, args.cell_info_dim, args.log_normalize))
    row = cur.fetchone()
    cols = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    log_str = ", log_normalize=True" if args.log_normalize else ""
    print(f"\ncell_info_mode={args.cell_info_mode}, cell_info_dim={args.cell_info_dim}{log_str}")
    print("=" * 40)
    for name, val in zip(cols, row):
        print(f"  {name:15s}: {val}")

if __name__ == "__main__":
    main()
