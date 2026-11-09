from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import Valentine modules
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("DATASET_ROOT", str(Path(__file__).resolve().parents[2])))
VALENTINE_DIR = Path(os.environ.get("VALENTINE_DIR", str(REPO_ROOT / "valentine")))
if str(VALENTINE_DIR) not in sys.path:
    sys.path.append(str(VALENTINE_DIR))

from dataset import Dataset  # type: ignore  # noqa: E402
import add_noise_schema  # type: ignore  # noqa: E402
import horizontal_transformations  # type: ignore  # noqa: E402
import vertical_transformations  # type: ignore  # noqa: E402
import os

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

MAGELLAN_ORI_DIR = REPO_ROOT / "magellan_ori"

OUTPUT_BASE = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
DATALAKE_DIR = OUTPUT_BASE / "datalake_plus"
META_DIR = OUTPUT_BASE / "metadata"

HORIZONTAL_OVERLAP = 0.6
DATA_NOISE_PERCENTAGE = 100
DATA_NOISE_INTENSITY = 15
COLUMN_NOISE_CHOICE = 4
DERIVED_COLUMN_DROP_RATIO = 0.3
JOINABLE_COMMON_RATIO = 0.35
FILENAME_ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789")
CHUNK_SIZE_ROW = 100


@dataclass
class TableMeta:
    dataset: str
    filename: str
    kind: str  # "original", "derived", or "joinable"
    origin_table: str
    table_stem: str
    table_id: int
    row_count: int
    column_count: int


def ensure_directories(clear: bool) -> None:
    for directory in [DATALAKE_DIR, META_DIR]:
        if clear and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def load_magellan_datasets() -> List[str]:
    if not MAGELLAN_ORI_DIR.exists():
        raise FileNotFoundError(f"Missing Magellan source directory: {MAGELLAN_ORI_DIR}")
    dataset_names: list[str] = []
    for path in MAGELLAN_ORI_DIR.iterdir():
        if not path.is_dir():
            continue
        if (path / "tableA.csv").exists() and (path / "tableB.csv").exists():
            dataset_names.append(path.name)
    return sorted(dataset_names)


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    schema = {idx: [col, str(df[col].dtype)] for idx, col in enumerate(df.columns)}
    return Dataset(schema=schema, data=df.copy())


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def clean_noisy_column_names(
    rename_map: Dict[str, str],
    dataset_name: str,
    table_name: str,
) -> Dict[str, str]:
    table_stem = Path(table_name).stem
    raw_candidates: set[str] = {dataset_name, table_stem}
    raw_candidates.update(part for part in table_stem.split("_") if part)
    raw_candidates.update(part for part in dataset_name.split("_") if part)
    normalized_candidates = {token for token in (_normalize_token(item) for item in raw_candidates) if token}

    cleaned: Dict[str, str] = {}
    used: set[str] = set()
    for original_name, noisy_name in rename_map.items():
        candidate = noisy_name
        if "_" in noisy_name:
            prefix, rest = noisy_name.split("_", 1)
            norm_prefix = _normalize_token(prefix)
            if rest and norm_prefix and normalized_candidates:
                if any(
                    norm_prefix == cand
                    or norm_prefix.startswith(cand)
                    or cand.startswith(norm_prefix)
                    for cand in normalized_candidates
                ):
                    candidate = rest
        candidate = candidate or noisy_name
        base_candidate = candidate
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base_candidate}_{suffix}"
        used.add(candidate)
        cleaned[original_name] = candidate
    return cleaned


def randomly_drop_columns(df: pd.DataFrame, drop_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    column_count = df.shape[1]
    if column_count <= 1 or drop_ratio <= 0:
        return df
    drop_count = math.ceil(column_count * drop_ratio)
    # Keep at least 2 columns to avoid degenerate 1-column tables (e.g., only an id column).
    drop_count = min(drop_count, column_count - 2)
    if drop_count <= 0:
        return df
    drop_indices = rng.choice(column_count, size=drop_count, replace=False)
    drop_columns = [df.columns[idx] for idx in drop_indices]
    return df.drop(columns=drop_columns)


def _perturb_table_stem(stem: str, rng: np.random.Generator) -> str:
    tokens = re.split(r"[-_]+", stem)
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return stem

    choice = int(rng.integers(0, 4))
    if choice == 0:
        tokens = tokens[::-1]
    elif choice == 1:
        tokens = [tok[: max(2, min(4, len(tok)))] for tok in tokens]
    elif choice == 2:
        tokens = [re.sub(r"[aeiouAEIOU]", "", tok) or tok for tok in tokens]
    else:
        rng.shuffle(tokens)

    joiner = rng.choice(["_", "-", ""])
    noisy = joiner.join(tokens).lower()
    return noisy or stem


def generate_noisy_filename(
    counter: int,
    dataset_name: str,
    table_stem: str,
    used: Set[str],
    rng: np.random.Generator,
    suffix_len: int = 6,
) -> str:
    while True:
        noisy_stem = _perturb_table_stem(table_stem, rng)
        suffix = "".join(rng.choice(FILENAME_ALPHABET, size=suffix_len))
        candidate = f"{counter}_{dataset_name}_{noisy_stem}_{suffix}.csv"
        if candidate not in used:
            used.add(candidate)
            return candidate


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size_row: int,
    chunk_size_col: int,
    rng: np.random.Generator,
) -> tuple[list[pd.DataFrame], list[int]]:
    """Chunk a dataframe by rows and (optionally) sub-sample non-id columns.

    Mirrors the logic used by `magellan/dataset_construction.py`:
    - adapt `chunk_size_row` so each source table yields ~5 chunks at most,
    - merge the last chunk if the remainder is small,
    - always keep the `id` column if present.
    """
    if df.empty:
        return [], [0]

    if chunk_size_row < int(len(df) / 5):
        chunk_size_row = int(len(df) / 5)
    if chunk_size_row <= 0:
        chunk_size_row = max(1, len(df))

    num_chunks = int(np.ceil(len(df) / chunk_size_row))
    chunks: list[pd.DataFrame] = []
    start_rows: list[int] = []

    finish = False
    for i in range(num_chunks):
        start_row = i * chunk_size_row
        end_row = min((i + 1) * chunk_size_row, len(df))
        if (len(df) - end_row) < 0.5 * chunk_size_row:
            end_row = len(df)
            finish = True

        df_chunk = df.iloc[start_row:end_row]
        start_rows.append(start_row)

        all_columns = df_chunk.columns.tolist()
        id_col = "id" if "id" in all_columns else None

        candidate_cols = [col for col in all_columns if col != id_col]
        select_k = min(chunk_size_col, len(candidate_cols))
        selected_cols: list[str] = []
        if select_k > 0:
            selected_cols = rng.choice(candidate_cols, size=select_k, replace=False).astype(str).tolist()

        if id_col is not None:
            selected_cols = [id_col] + selected_cols
        elif not selected_cols and all_columns:
            selected_cols = [str(all_columns[0])]

        chunks.append(df_chunk[selected_cols].copy())

        if finish:
            break

    start_rows.append(len(df) + 1)
    return chunks, start_rows


def _apply_column_noise(
    df: pd.DataFrame,
    dataset_name: str,
    table_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    if df.empty:
        return df.copy(), {}

    noisy_ds = dataframe_to_dataset(df)
    before_names = {idx: noisy_ds.schema[idx][0] for idx in sorted(noisy_ds.schema.keys())}
    noisy_ds = add_noise_schema.approximate_column_names(
        noisy_ds,
        Path(table_name).stem,
        COLUMN_NOISE_CHOICE,
    )
    after_names = {idx: noisy_ds.schema[idx][0] for idx in sorted(noisy_ds.schema.keys())}
    rename_map = {before_names[idx]: after_names[idx] for idx in before_names}
    rename_map = clean_noisy_column_names(rename_map, dataset_name, table_name)

    derived_df = df.copy().rename(columns=rename_map)
    derived_df = randomly_drop_columns(derived_df, DERIVED_COLUMN_DROP_RATIO, rng)
    return derived_df, rename_map


def create_valentine_variant(
    df: pd.DataFrame,
    dataset_name: str,
    table_name: str,
    rng: np.random.Generator,
) -> Optional[tuple[pd.DataFrame, List[int], Dict[str, str]]]:
    if df.empty:
        return None

    baseline_ds = dataframe_to_dataset(df)
    try:
        derived_plain, _, _ = horizontal_transformations.split(
            baseline_ds,
            overlap=HORIZONTAL_OVERLAP,
            rand=False,
            approx=False,
            prc=0,
            approx_prc=0,
        )
    except Exception as exc:
        print(f"⚠️  Horizontal split failed on {dataset_name}/{table_name}: {exc}")
        return None

    index_map = derived_plain.data.index.astype(int).tolist()

    noisy_ds = dataframe_to_dataset(df)
    derived_noisy, _, _ = horizontal_transformations.split(
        noisy_ds,
        overlap=HORIZONTAL_OVERLAP,
        rand=False,
        approx=True,
        prc=DATA_NOISE_PERCENTAGE,
        approx_prc=DATA_NOISE_INTENSITY,
    )
    if derived_noisy.data.empty:
        return None

    derived_df = derived_noisy.data.copy()
    derived_df, rename_map = _apply_column_noise(
        derived_df,
        dataset_name=dataset_name,
        table_name=table_name,
        rng=rng,
    )

    if len(index_map) != len(derived_df):
        min_len = min(len(index_map), len(derived_df))
        index_map = index_map[:min_len]
        derived_df = derived_df.iloc[:min_len].copy()

    row_perm = rng.permutation(len(derived_df))
    derived_df = derived_df.iloc[row_perm].reset_index(drop=True)
    index_map = [index_map[idx] for idx in row_perm]

    shuffled_columns = list(derived_df.columns)
    rng.shuffle(shuffled_columns)
    derived_df = derived_df[shuffled_columns]

    renamed_lookup: Dict[str, str] = {}
    for original_col in df.columns:
        candidate = rename_map.get(original_col, original_col)
        if candidate in derived_df.columns:
            renamed_lookup[original_col] = candidate

    if not renamed_lookup:
        return None

    return derived_df, index_map, renamed_lookup


def create_fallback_variant(
    df: pd.DataFrame,
    dataset_name: str,
    table_name: str,
    rng: np.random.Generator,
) -> Optional[tuple[pd.DataFrame, List[int], Dict[str, str]]]:
    if df.empty:
        return None

    derived_df, rename_map = _apply_column_noise(
        df,
        dataset_name=dataset_name,
        table_name=table_name,
        rng=rng,
    )
    if derived_df.empty:
        return None

    index_map = list(range(len(derived_df)))
    row_perm = rng.permutation(len(derived_df))
    derived_df = derived_df.iloc[row_perm].reset_index(drop=True)
    index_map = [index_map[idx] for idx in row_perm]

    shuffled_columns = list(derived_df.columns)
    rng.shuffle(shuffled_columns)
    derived_df = derived_df[shuffled_columns]

    renamed_lookup: Dict[str, str] = {}
    for original_col in df.columns:
        candidate = rename_map.get(original_col, original_col)
        if candidate in derived_df.columns:
            renamed_lookup[original_col] = candidate

    if not renamed_lookup:
        return None

    return derived_df, index_map, renamed_lookup


def choose_primary_key_index(df: pd.DataFrame) -> int:
    best_idx = 0
    best_score = -1.0
    total_rows = len(df)
    for idx, column in enumerate(df.columns):
        non_null = df[column].dropna()
        if non_null.empty or total_rows == 0:
            score = 0.0
        else:
            score = non_null.nunique(dropna=True) / max(len(non_null), 1)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def create_joinable_variant(
    df: pd.DataFrame,
    dataset_name: str,
    table_name: str,
    rng: np.random.Generator,
) -> Optional[Tuple[pd.DataFrame, Dict[str, str], List[Tuple[str, str]]]]:
    if df.empty or df.shape[1] < 2:
        return None

    pk_index = choose_primary_key_index(df)
    column_indices = list(range(df.shape[1]))
    working_df = df.copy()
    working_df.columns = column_indices
    schema = {idx: [df.columns[idx], str(df.dtypes.iloc[idx])] for idx in column_indices}
    base_ds = Dataset(schema=schema, data=working_df)

    try:
        _, joinable_ds, common_columns = vertical_transformations.split(
            base_ds,
            pk=pk_index,
            common=JOINABLE_COMMON_RATIO,
            rand=False,
            approx=True,
            prc=DATA_NOISE_PERCENTAGE,
            approx_prc=DATA_NOISE_INTENSITY,
        )
    except Exception as exc:
        print(f"⚠️  Vertical split failed on {dataset_name}/{table_name}: {exc}")
        return None

    if joinable_ds.data.empty:
        return None

    joinable_df = joinable_ds.data.copy()
    before_names = {idx: joinable_ds.schema[idx][0] for idx in sorted(joinable_ds.schema.keys())}
    column_alignment = {idx: before_names[idx] for idx in before_names if idx in joinable_df.columns}
    joinable_df = joinable_df.rename(columns=column_alignment)

    joinable_ds = add_noise_schema.approximate_column_names(
        joinable_ds,
        Path(table_name).stem,
        COLUMN_NOISE_CHOICE,
    )
    after_names = {idx: joinable_ds.schema[idx][0] for idx in sorted(joinable_ds.schema.keys())}
    rename_map = {before_names[idx]: after_names[idx] for idx in before_names}
    rename_map = clean_noisy_column_names(rename_map, dataset_name, table_name)
    joinable_df = joinable_df.rename(columns=rename_map)

    shuffled_columns = list(joinable_df.columns)
    rng.shuffle(shuffled_columns)
    joinable_df = joinable_df[shuffled_columns]

    rename_lookup: Dict[str, str] = {}
    for idx, original_name in enumerate(df.columns):
        before_name = before_names.get(idx)
        if before_name is None:
            continue
        cleaned_name = rename_map.get(before_name, before_name)
        if cleaned_name in joinable_df.columns:
            rename_lookup[original_name] = cleaned_name

    if not rename_lookup:
        return None

    join_pairs: List[Tuple[str, str]] = []
    for idx in common_columns:
        if idx >= len(df.columns):
            continue
        original_name = df.columns[idx]
        renamed = rename_lookup.get(original_name)
        if renamed:
            join_pairs.append((str(original_name), str(renamed)))

    if not join_pairs:
        return None

    return joinable_df, rename_lookup, join_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Magellan datalake_plus (original + derived + joinable tables) using Valentine-style transforms."
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing outputs.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ensure_directories(clear=not args.no_clean)

    dataset_names = load_magellan_datasets()
    if args.limit:
        dataset_names = dataset_names[: args.limit]

    table_counter = 0
    used_filenames: Set[str] = set()
    table_registry: List[TableMeta] = []
    derived_records: List[Dict[str, object]] = []
    joinable_records: List[Dict[str, object]] = []

    for dataset_name in dataset_names:
        dataset_dir = MAGELLAN_ORI_DIR / dataset_name
        for side in ["tableA", "tableB"]:
            csv_name = f"{side}.csv"
            csv_path = dataset_dir / csv_name
            if not csv_path.exists():
                print(f"⚠️  Skip {dataset_name}: missing {csv_path}")
                continue

            df_full = pd.read_csv(csv_path)
            all_columns = df_full.columns.tolist()
            id_col = "id" if "id" in all_columns else None
            candidate_cols = [col for col in all_columns if col != id_col]
            candidate_count = len(candidate_cols)

            for j in range(2):
                chunk_size_col = candidate_count - j
                if chunk_size_col <= 0:
                    continue
                chunks, _ = chunk_dataframe(
                    df_full,
                    chunk_size_row=CHUNK_SIZE_ROW,
                    chunk_size_col=chunk_size_col,
                    rng=rng,
                )
                if not chunks:
                    continue

                for chunk_idx, df_original in enumerate(chunks):
                    base_stem = f"{side}_colsize{len(df_original.columns)}_chunk{chunk_idx}"
                    original_filename = f"{table_counter}_{dataset_name}_{base_stem}.csv"
                    df_original.to_csv(DATALAKE_DIR / original_filename, index=False)
                    used_filenames.add(original_filename)
                    table_registry.append(
                        TableMeta(
                            dataset=dataset_name,
                            filename=original_filename,
                            kind="original",
                            origin_table=original_filename,
                            table_stem=base_stem,
                            table_id=table_counter,
                            row_count=len(df_original),
                            column_count=len(df_original.columns),
                        )
                    )
                    table_counter += 1

                    variant = create_valentine_variant(
                        df_original,
                        dataset_name=dataset_name,
                        table_name=f"{base_stem}.csv",
                        rng=rng,
                    )
                    if variant is None:
                        variant = create_fallback_variant(
                            df_original,
                            dataset_name=dataset_name,
                            table_name=f"{base_stem}.csv",
                            rng=rng,
                        )
                    if variant is None:
                        print(f"⚠️  Skip derived for {dataset_name}/{base_stem}: empty variant")
                        continue

                    derived_df, index_map, renamed_lookup = variant
                    derived_filename = generate_noisy_filename(
                        table_counter,
                        dataset_name,
                        base_stem,
                        used_filenames,
                        rng=rng,
                    )
                    derived_df.to_csv(DATALAKE_DIR / derived_filename, index=False)
                    table_registry.append(
                        TableMeta(
                            dataset=dataset_name,
                            filename=derived_filename,
                            kind="derived",
                            origin_table=original_filename,
                            table_stem=Path(derived_filename).stem,
                            table_id=table_counter,
                            row_count=len(derived_df),
                            column_count=len(derived_df.columns),
                        )
                    )
                    derived_records.append(
                        {
                            "dataset_name": dataset_name,
                            "origin_table": original_filename,
                            "derived_table": derived_filename,
                            "row_map": json.dumps(index_map),
                            "column_map": json.dumps(renamed_lookup),
                        }
                    )
                    table_counter += 1

                    joinable_variant = create_joinable_variant(
                        df_original,
                        dataset_name=dataset_name,
                        table_name=f"{base_stem}.csv",
                        rng=rng,
                    )
                    if joinable_variant is None:
                        continue

                    joinable_df, joinable_lookup, join_pairs = joinable_variant
                    joinable_filename = generate_noisy_filename(
                        table_counter,
                        dataset_name,
                        f"joinable_{base_stem}",
                        used_filenames,
                        rng=rng,
                    )
                    joinable_df.to_csv(DATALAKE_DIR / joinable_filename, index=False)
                    table_registry.append(
                        TableMeta(
                            dataset=dataset_name,
                            filename=joinable_filename,
                            kind="joinable",
                            origin_table=original_filename,
                            table_stem=Path(joinable_filename).stem,
                            table_id=table_counter,
                            row_count=len(joinable_df),
                            column_count=len(joinable_df.columns),
                        )
                    )
                    joinable_records.append(
                        {
                            "dataset_name": dataset_name,
                            "origin_table": original_filename,
                            "joinable_table": joinable_filename,
                            "column_map": json.dumps(joinable_lookup),
                            "join_pairs": json.dumps(join_pairs),
                        }
                    )
                    table_counter += 1

    registry_path = META_DIR / "table_registry.csv"
    pd.DataFrame([record.__dict__ for record in table_registry]).to_csv(registry_path, index=False)

    derived_path = META_DIR / "derived_map.csv"
    pd.DataFrame(derived_records).to_csv(derived_path, index=False)

    joinable_path = META_DIR / "joinable_map.csv"
    pd.DataFrame(joinable_records).to_csv(joinable_path, index=False)

    print("Generation complete.")
    print(f"  Tables written: {len(table_registry)}")
    print(f"  Derived mappings: {len(derived_records)}")
    print(f"  Joinable mappings: {len(joinable_records)}")
    print(f"  Registry: {registry_path}")
    print(f"  Derived map: {derived_path}")
    print(f"  Joinable map: {joinable_path}")


if __name__ == "__main__":
    main()
