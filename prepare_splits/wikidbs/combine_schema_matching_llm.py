import argparse
import ast
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
RAW_DEFAULT = BASE_DIR / "label_plus" / "schema_matching" / "GPT_extracted_schema_matching_results.csv"
OUTPUT_DEFAULT = BASE_DIR / "label_plus" / "schema_matching" / "schema_matching_labels.csv"
LEGACY_BASE = Path(os.environ.get("DATASET_ROOT", str(BASE_DIR.parent))) / "wikidbs"


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def best_match(candidates: list[str], query: str) -> str:
    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        score = SequenceMatcher(None, query, cand).ratio()
        if score > best_score:
            best_score = score
            best = cand
    return best


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("```", "")
    s = re.sub(r"^\s*python\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def parse_pairs(raw: str) -> list[list[str]]:
    raw = strip_code_fences(raw)
    if not raw:
        return []
    try:
        obj = ast.literal_eval(raw)
    except Exception:
        return []
    pairs: list[list[str]] = []
    for item in obj:
        if isinstance(item, tuple):
            item = list(item)
        if isinstance(item, str):
            parts = [p.strip().strip("'\"") for p in item.split(",")]
            if len(parts) == 4:
                pairs.append(parts)
            continue
        if isinstance(item, list) and len(item) == 4:
            pairs.append([str(x).strip() for x in item])
    return pairs


def resolve_table(dataset_tables: list[dict[str, str]], raw_table: str) -> str | None:
    norm = normalize_token(raw_table)
    exact = [t for t in dataset_tables if t["norm"] == norm]
    if exact:
        return exact[0]["filename"]
    norms = [t["norm"] for t in dataset_tables]
    best_norm = best_match(norms, norm)
    for t in dataset_tables:
        if t["norm"] == best_norm:
            return t["filename"]
    return None


def resolve_column(columns: list[str], raw_col: str) -> str:
    norm = normalize_token(raw_col)
    exact = [c for c in columns if normalize_token(c) == norm]
    if exact:
        return exact[0]
    best_norm = best_match([normalize_token(c) for c in columns], norm)
    for c in columns:
        if normalize_token(c) == best_norm:
            return c
    return columns[0]


def normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Map LLM schema matching pairs to datalake tables.")
    parser.add_argument("--raw-csv", default=str(RAW_DEFAULT))
    parser.add_argument("--out-csv", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--neg-per-pos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry_path = META_DIR / "table_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}. Run generate_datalake.py first.")

    raw_path = Path(args.raw_csv)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw LLM output at {raw_path}")

    rng = random.Random(args.seed)
    registry = pd.read_csv(registry_path)
    originals = registry[registry["kind"] == "original"].copy()

    dataset_tables: dict[str, list[dict[str, str]]] = {}
    for row in originals.itertuples(index=False):
        dataset_tables.setdefault(row.dataset, []).append(
            {
                "filename": row.filename,
                "norm": normalize_token(str(row.table_stem)),
            }
        )

    raw_df = pd.read_csv(raw_path)
    columns_lookup: dict[str, dict[str, set[str]]] = {}
    columns_cache: dict[str, list[str]] = {}
    positives: list[dict[str, object]] = []
    pos_set: set[tuple[str, str, str, str]] = set()

    for row in raw_df.itertuples(index=False):
        dataset_name = str(row.dataset_name).strip()
        analysis = str(row.analysis)
        if dataset_name not in dataset_tables:
            continue
        pairs = parse_pairs(analysis)
        if not pairs:
            continue

        if dataset_name not in columns_lookup:
            info_path = LEGACY_BASE / "selected_dataset" / dataset_name / "info_full.json"
            table_cols: dict[str, set[str]] = {}
            if info_path.exists():
                payload = json.loads(info_path.read_text(encoding="utf-8"))
                tables = payload.get("TABLES", {})
                for table_name, table_info in tables.items():
                    cols = table_info.get("COLUMNS", []) or []
                    table_cols[table_name] = {normalize_column_name(c) for c in cols}
            columns_lookup[dataset_name] = table_cols

        for t1, c1, t2, c2 in pairs:
            table_cols = columns_lookup.get(dataset_name, {})
            cols1 = table_cols.get(t1)
            cols2 = table_cols.get(t2)
            if cols1 is not None and normalize_column_name(c1) not in cols1:
                continue
            if cols2 is not None and normalize_column_name(c2) not in cols2:
                continue

            file1 = resolve_table(dataset_tables[dataset_name], t1)
            file2 = resolve_table(dataset_tables[dataset_name], t2)
            if not file1 or not file2 or file1 == file2:
                continue

            if file1 not in columns_cache:
                columns_cache[file1] = pd.read_csv(
                    BASE_DIR / "datalake_plus" / file1, nrows=1
                ).columns.astype(str).tolist()
            if file2 not in columns_cache:
                columns_cache[file2] = pd.read_csv(
                    BASE_DIR / "datalake_plus" / file2, nrows=1
                ).columns.astype(str).tolist()

            col1 = resolve_column(columns_cache[file1], c1)
            col2 = resolve_column(columns_cache[file2], c2)
            key = (file1, col1, file2, col2)
            if key in pos_set:
                continue
            pos_set.add(key)
            positives.append(
                {
                    "table_name_1": file1,
                    "renamed_column_name_1": col1,
                    "table_name_2": file2,
                    "renamed_column_name_2": col2,
                    "label": 1,
                }
            )

    negatives: list[dict[str, object]] = []
    if args.neg_per_pos > 0 and positives:
        pool_by_dataset = {
            dataset: [t["filename"] for t in tables]
            for dataset, tables in dataset_tables.items()
        }
        for pos in positives:
            file1 = pos["table_name_1"]
            dataset = registry.loc[registry["filename"] == file1, "dataset"].iloc[0]
            candidates = pool_by_dataset.get(dataset, [])
            if not candidates:
                continue
            for _ in range(args.neg_per_pos):
                file2 = rng.choice(candidates)
                if file2 not in columns_cache:
                    columns_cache[file2] = pd.read_csv(
                        BASE_DIR / "datalake_plus" / file2, nrows=1
                    ).columns.astype(str).tolist()
                col1 = pos["renamed_column_name_1"]
                col2 = rng.choice(columns_cache[file2])
                key = (file1, col1, file2, col2)
                if key in pos_set:
                    continue
                pos_set.add(key)
                negatives.append(
                    {
                        "table_name_1": file1,
                        "renamed_column_name_1": col1,
                        "table_name_2": file2,
                        "renamed_column_name_2": col2,
                        "label": 0,
                    }
                )
                break

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(positives + negatives)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path} (pos={len(positives)} neg={len(negatives)})")


if __name__ == "__main__":
    main()
