from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
REPO_ROOT = Path(os.environ.get("DATASET_ROOT", str(BASE_DIR.parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "unionable_table_search"
SANTOS_ORI_DIR = REPO_ROOT / "santos_benchmark_ori"
SANTOS_GT_PATH = SANTOS_ORI_DIR / "santos_small_benchmark_groundtruth.csv"

TABLE_REGISTRY_PATH = META_DIR / "table_registry.csv"


def strip_generated_prefix(filename: str) -> str:
    """Remove the numeric prefix added by generate_datalake.py (e.g., '12_x.csv' -> 'x.csv')."""
    parts = filename.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected generated table name format: {filename}")
    return parts[1]


def build_source_lookup(registry: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map (source_kind, base_filename) -> datalake_plus original filename.

    We need this because Santos has two sources (datalake + query) and can reuse
    the same basename in both folders.
    """
    if "filename" not in registry.columns or "kind" not in registry.columns:
        raise ValueError(f"Unexpected table_registry columns: {sorted(registry.columns)}")

    originals = registry[registry["kind"].astype(str) == "original"].copy()
    if originals.empty:
        raise ValueError("No original tables found in table_registry.csv; run generate_datalake.py first.")

    if "table_id" not in originals.columns:
        raise ValueError("table_registry.csv missing table_id column")

    datalake_dir = SANTOS_ORI_DIR / "datalake"
    query_dir = SANTOS_ORI_DIR / "query"
    if not datalake_dir.exists() or not query_dir.exists():
        raise FileNotFoundError(f"Missing Santos source folders under {SANTOS_ORI_DIR}")

    datalake_files = {p.name for p in datalake_dir.glob("*.csv")}
    query_files = {p.name for p in query_dir.glob("*.csv")}

    by_base: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in originals.itertuples(index=False):
        filename = str(row.filename)
        base = strip_generated_prefix(filename)
        by_base[base].append((int(row.table_id), filename))

    lookup: dict[tuple[str, str], str] = {}
    for base, candidates in by_base.items():
        candidates = sorted(candidates, key=lambda item: item[0])
        in_datalake = base in datalake_files
        in_query = base in query_files

        if in_datalake and in_query:
            if len(candidates) < 2:
                raise ValueError(
                    f"Expected 2 originals for basename present in both sources: {base} "
                    f"(found {len(candidates)})"
                )
            lookup[("datalake", base)] = candidates[0][1]
            lookup[("query", base)] = candidates[1][1]
        elif in_datalake:
            if len(candidates) != 1:
                raise ValueError(f"Expected 1 original for datalake-only file {base} (found {len(candidates)})")
            lookup[("datalake", base)] = candidates[0][1]
        elif in_query:
            if len(candidates) != 1:
                raise ValueError(f"Expected 1 original for query-only file {base} (found {len(candidates)})")
            lookup[("query", base)] = candidates[0][1]

    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unionable table search labels from Santos ground truth.")
    args = parser.parse_args()
    _ = args

    if not TABLE_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing {TABLE_REGISTRY_PATH}. Run generate_datalake.py first.")
    if not SANTOS_GT_PATH.exists():
        raise FileNotFoundError(f"Missing Santos UTS ground truth: {SANTOS_GT_PATH}")

    registry = pd.read_csv(TABLE_REGISTRY_PATH)
    source_lookup = build_source_lookup(registry)

    gt_df = pd.read_csv(SANTOS_GT_PATH, encoding="utf-8-sig")
    required_cols = ["query_table", "data_lake_table"]
    for col in required_cols:
        if col not in gt_df.columns:
            raise ValueError(f"Missing required column in ground truth: {col}")

    records: list[dict[str, object]] = []
    skipped = 0
    missing_examples: list[str] = []
    for row in gt_df.itertuples(index=False):
        query_base = str(getattr(row, "query_table"))
        dl_base = str(getattr(row, "data_lake_table"))
        query_table = source_lookup.get(("query", query_base))
        dl_table = source_lookup.get(("datalake", dl_base))
        if query_table is None or dl_table is None:
            skipped += 1
            if len(missing_examples) < 5:
                missing_examples.append(f"query={query_base} -> {query_table}, datalake={dl_base} -> {dl_table}")
            continue
        if query_table == dl_table:
            continue
        records.append({"table_name_1": query_table, "table_name_2": dl_table, "label": 1})

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "unionable_table_search_labels.csv"
    out_df = pd.DataFrame(records, columns=["table_name_1", "table_name_2", "label"])
    if not out_df.empty:
        out_df = out_df.drop_duplicates().reset_index(drop=True)
        out_df["label"] = out_df["label"].astype(int)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    if skipped:
        detail = "; ".join(missing_examples)
        print(f"Skipped {skipped} GT rows with missing tables ({detail})")


if __name__ == "__main__":
    main()
