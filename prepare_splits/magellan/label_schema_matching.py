from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "schema_matching"


def iter_pairs_from_column_map(
    origin_table: str,
    other_table: str,
    column_map_raw: str | None,
) -> list[dict[str, object]]:
    if not column_map_raw or not str(column_map_raw).strip():
        return []
    try:
        mapping = json.loads(column_map_raw)
    except Exception:
        return []
    if not isinstance(mapping, dict):
        return []
    records: list[dict[str, object]] = []
    for original_col, mapped_col in mapping.items():
        records.append(
            {
                "table_name_1": origin_table,
                "renamed_column_name_1": str(original_col),
                "table_name_2": other_table,
                "renamed_column_name_2": str(mapped_col),
                "label": 1,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate schema matching labels from derived/joinable column maps.")
    args = parser.parse_args()
    _ = args

    derived_path = META_DIR / "derived_map.csv"
    joinable_path = META_DIR / "joinable_map.csv"
    if not derived_path.exists():
        raise FileNotFoundError(f"Missing {derived_path}. Run generate_datalake.py first.")

    records: list[dict[str, object]] = []

    derived_df = pd.read_csv(derived_path)
    for row in derived_df.itertuples(index=False):
        records.extend(
            iter_pairs_from_column_map(
                origin_table=str(row.origin_table),
                other_table=str(row.derived_table),
                column_map_raw=getattr(row, "column_map", None),
            )
        )

    if joinable_path.exists():
        joinable_df = pd.read_csv(joinable_path)
        for row in joinable_df.itertuples(index=False):
            records.extend(
                iter_pairs_from_column_map(
                    origin_table=str(row.origin_table),
                    other_table=str(row.joinable_table),
                    column_map_raw=getattr(row, "column_map", None),
                )
            )

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "schema_matching_labels.csv"
    out_df = pd.DataFrame(
        records,
        columns=[
            "table_name_1",
            "renamed_column_name_1",
            "table_name_2",
            "renamed_column_name_2",
            "label",
        ],
    )
    if not out_df.empty:
        out_df = out_df.drop_duplicates().reset_index(drop=True)
        out_df["label"] = out_df["label"].astype(int)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()

