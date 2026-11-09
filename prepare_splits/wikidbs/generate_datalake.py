from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

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
import horizontal_transformations  # type: ignore  # noqa: E402
import add_noise_schema  # type: ignore  # noqa: E402
import os

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

LEGACY_BASE = REPO_ROOT / "wikidbs"
SELECTED_DATASETS_DIR = LEGACY_BASE / "selected_dataset"
SELECTED_LIST_PATH = LEGACY_BASE / "selected_folder_wikidbs.csv"

OUTPUT_BASE = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
DATALAKE_DIR = OUTPUT_BASE / "datalake_plus"
META_DIR = OUTPUT_BASE / "metadata"

HORIZONTAL_OVERLAP = 0.6
DATA_NOISE_PERCENTAGE = 100
DATA_NOISE_INTENSITY = 15
COLUMN_NOISE_CHOICE = 4
DERIVED_COLUMN_DROP_RATIO = 0.3
FILENAME_ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789")


@dataclass
class TableMeta:
    dataset: str
    filename: str
    kind: str  # "original" or "derived"
    origin_table: str
    table_stem: str
    table_id: int
    row_count: int
    column_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_directories(clear: bool) -> None:
    for directory in [DATALAKE_DIR, META_DIR]:
        if clear and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def load_selected_datasets() -> List[str]:
    if not SELECTED_LIST_PATH.exists():
        raise FileNotFoundError(
            "Expected selected_folder_wikidbs.csv at "
            f"{SELECTED_LIST_PATH}. Run the legacy selection first."
        )
    frame = pd.read_csv(SELECTED_LIST_PATH)
    if "filename" not in frame.columns:
        raise ValueError("selected_folder_wikidbs.csv must contain a 'filename' column.")
    return frame["filename"].dropna().tolist()


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
    raw_candidates: Set[str] = {dataset_name, table_stem}
    raw_candidates.update(part for part in table_stem.split("_") if part)
    raw_candidates.update(part for part in dataset_name.split("_") if part)
    normalized_candidates = {
        token for token in (_normalize_token(item) for item in raw_candidates) if token
    }

    cleaned: Dict[str, str] = {}
    used: Set[str] = set()
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
    drop_count = min(drop_count, column_count - 1)
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

    original_columns = list(df.columns)
    renamed_lookup: Dict[str, str] = {}
    for original_col in original_columns:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate datalake_plus (original + derived tables) with Valentine-style obfuscation."
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing outputs.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ensure_directories(clear=not args.no_clean)

    dataset_names = load_selected_datasets()
    if args.limit:
        dataset_names = dataset_names[: args.limit]

    table_counter = 0
    used_filenames: Set[str] = set()
    table_registry: List[TableMeta] = []
    derived_records: List[Dict[str, object]] = []

    for dataset_name in dataset_names:
        tables_root = SELECTED_DATASETS_DIR / dataset_name / "tables"
        if not tables_root.exists():
            print(f"⚠️  Skip dataset {dataset_name}: missing {tables_root}")
            continue
        csv_paths = sorted(tables_root.glob("*.csv"))
        if not csv_paths:
            print(f"⚠️  Skip dataset {dataset_name}: no tables found")
            continue

        for csv_path in csv_paths:
            df_original = pd.read_csv(csv_path)
            original_filename = f"{table_counter}_{dataset_name}_{csv_path.name}"
            df_original.to_csv(DATALAKE_DIR / original_filename, index=False)
            used_filenames.add(original_filename)
            table_registry.append(
                TableMeta(
                    dataset=dataset_name,
                    filename=original_filename,
                    kind="original",
                    origin_table=original_filename,
                    table_stem=csv_path.stem,
                    table_id=table_counter,
                    row_count=len(df_original),
                    column_count=len(df_original.columns),
                )
            )
            table_counter += 1

            variant = create_valentine_variant(
                df_original,
                dataset_name=dataset_name,
                table_name=csv_path.name,
                rng=rng,
            )
            if variant is None:
                variant = create_fallback_variant(
                    df_original,
                    dataset_name=dataset_name,
                    table_name=csv_path.name,
                    rng=rng,
                )
            if variant is None:
                print(f"⚠️  Skip derived for {dataset_name}/{csv_path.name}: empty variant")
                continue

            derived_df, index_map, renamed_lookup = variant
            derived_filename = generate_noisy_filename(
                table_counter,
                dataset_name,
                csv_path.stem,
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

    registry_path = META_DIR / "table_registry.csv"
    registry_df = pd.DataFrame([record.__dict__ for record in table_registry])
    registry_df.to_csv(registry_path, index=False)

    derived_path = META_DIR / "derived_map.csv"
    pd.DataFrame(derived_records).to_csv(derived_path, index=False)

    print("Generation complete.")
    print(f"  Tables written: {len(table_registry)}")
    print(f"  Derived mappings: {len(derived_records)}")
    print(f"  Registry: {registry_path}")
    print(f"  Derived map: {derived_path}")


if __name__ == "__main__":
    main()

