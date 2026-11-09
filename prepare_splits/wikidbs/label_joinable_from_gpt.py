import argparse
import ast
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "joinable_table_search"
LEGACY_GPT_PATH = Path(os.environ.get("DATASET_ROOT", str(Path(__file__).resolve().parents[2]))) / "wikidbs" / "label" / "joinable_table_search" / "GPT_extracted_results.csv"


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
    s = s.replace("python\njoinable_relationships = ", "")
    s = s.replace("python\n", "")
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


def resolve_table(
    dataset_tables: list[dict[str, str]],
    raw_table: str,
) -> str | None:
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
    if norm in {"rowid"} and "label" in columns:
        return "label"
    exact = [c for c in columns if normalize_token(c) == norm]
    if exact:
        return exact[0]
    best_norm = best_match([normalize_token(c) for c in columns], norm)
    for c in columns:
        if normalize_token(c) == best_norm:
            return c
    return columns[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Map GPT joinable results to datalake_plus tables.")
    parser.add_argument("--gpt-results", default=str(LEGACY_GPT_PATH))
    args = parser.parse_args()

    registry_path = META_DIR / "table_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}. Run generate_datalake.py first.")

    gpt_path = Path(args.gpt_results)
    if not gpt_path.exists():
        raise FileNotFoundError(f"Missing GPT results at {gpt_path}")

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

    gpt_df = pd.read_csv(gpt_path)
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    columns_cache: dict[str, list[str]] = {}

    for row in gpt_df.itertuples(index=False):
        dataset_name = str(row.dataset_name).strip()
        analysis = str(row.analysis)
        if dataset_name not in dataset_tables:
            continue
        pairs = parse_pairs(analysis)
        if not pairs:
            continue

        for t1, c1, t2, c2 in pairs:
            file1 = resolve_table(dataset_tables[dataset_name], t1)
            file2 = resolve_table(dataset_tables[dataset_name], t2)
            if not file1 or not file2:
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
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "table_name_1": file1,
                    "column_name_1": col1,
                    "table_name_2": file2,
                    "column_name_2": col2,
                    "ratio": 1.0,
                    "label": 1,
                }
            )

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "joinable_table_search_labels.csv"
    pd.DataFrame(
        records,
        columns=[
            "table_name_1",
            "column_name_1",
            "table_name_2",
            "column_name_2",
            "ratio",
            "label",
        ],
    ).to_csv(out_path, index=False)
    print(f"Wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
