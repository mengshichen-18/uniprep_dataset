from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "joinable_table_search"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate joinable table search labels from joinable_map.csv.")
    args = parser.parse_args()
    _ = args

    joinable_map_path = META_DIR / "joinable_map.csv"
    if not joinable_map_path.exists():
        raise FileNotFoundError(f"Missing {joinable_map_path}. Run generate_datalake.py first.")

    df = pd.read_csv(joinable_map_path)
    records: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        origin = str(row.origin_table)
        joinable = str(row.joinable_table)
        raw_pairs = getattr(row, "join_pairs", None)
        if raw_pairs is None:
            continue
        try:
            pairs = json.loads(raw_pairs)
        except Exception:
            continue
        if not isinstance(pairs, list):
            continue
        for item in pairs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            left_col, right_col = item
            records.append(
                {
                    "table_name_1": origin,
                    "column_name_1": str(left_col),
                    "table_name_2": joinable,
                    "column_name_2": str(right_col),
                    "ratio": 1.0,
                    "label": 1,
                }
            )

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "joinable_table_search_labels.csv"
    out_df = pd.DataFrame(
        records,
        columns=[
            "table_name_1",
            "column_name_1",
            "table_name_2",
            "column_name_2",
            "ratio",
            "label",
        ],
    )
    if not out_df.empty:
        out_df = out_df.drop_duplicates().reset_index(drop=True)
        out_df["label"] = out_df["label"].astype(int)
        out_df["ratio"] = out_df["ratio"].astype(float)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()

