from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "entity_matching"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate entity matching labels from derived_map.csv.")
    parser.add_argument("--row-sample", type=int, default=3, help="Sample size per derived table.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    derived_map_path = META_DIR / "derived_map.csv"
    if not derived_map_path.exists():
        raise FileNotFoundError(f"Missing {derived_map_path}. Run generate_datalake.py first.")

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(derived_map_path)

    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        origin = row.origin_table
        derived = row.derived_table
        row_map = json.loads(row.row_map)
        if not row_map:
            continue
        sample_size = min(args.row_sample, len(row_map))
        chosen_positions = rng.choice(len(row_map), size=sample_size, replace=False)
        for position in sorted(chosen_positions):
            records.append(
                {
                    "ltable_name": origin,
                    "l_no": int(row_map[position]),
                    "rtable_name": derived,
                    "r_no": int(position),
                    "label": 1,
                }
            )

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "entity_matching_labels.csv"
    pd.DataFrame(
        records,
        columns=["ltable_name", "l_no", "rtable_name", "r_no", "label"],
    ).to_csv(out_path, index=False)
    print(f"Wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()

