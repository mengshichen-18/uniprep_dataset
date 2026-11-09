from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
DATALAKE_DIR = BASE_DIR / "datalake_plus"
LABEL_ROOT = BASE_DIR / "label_plus"
META_DIR = BASE_DIR / "metadata"

TRAIN_RATIO = 0.4
VALIDATE_RATIO = 0.3
TEST_RATIO = 0.3
LABEL_BALANCE_RATIO = 3
RANDOM_SEED = 42
SPLIT_MODE = "random"  # random | dataset-disjoint


def assign_groups_to_splits(
    groups: List[str],
    train_ratio: float,
    validate_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, str]:
    if abs(train_ratio + validate_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    unique_groups = sorted({str(g) for g in groups if str(g).strip()})
    if not unique_groups:
        return {}

    perm = rng.permutation(len(unique_groups))
    shuffled = [unique_groups[i] for i in perm]
    total = len(shuffled)
    test_count = round(total * test_ratio)
    val_count = round(total * validate_ratio)

    test_groups = set(shuffled[:test_count])
    val_groups = set(shuffled[test_count:test_count + val_count])

    mapping: Dict[str, str] = {}
    for group in unique_groups:
        if group in test_groups:
            mapping[group] = "test"
        elif group in val_groups:
            mapping[group] = "validate"
        else:
            mapping[group] = "train"
    return mapping


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    validate_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df)
    if total == 0:
        return df.copy(), df.copy(), df.copy()

    if abs(train_ratio + validate_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    perm = rng.permutation(total)
    shuffled = df.iloc[perm].reset_index(drop=True)
    test_count = round(total * test_ratio)
    val_count = round(total * validate_ratio)

    test_df = shuffled.iloc[:test_count].copy()
    val_df = shuffled.iloc[test_count:test_count + val_count].copy()
    train_df = shuffled.iloc[test_count + val_count:].copy()
    return train_df, val_df, test_df


def balance_labels(df: pd.DataFrame, max_ratio: int) -> pd.DataFrame:
    if "label" not in df.columns:
        return df
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    if pos.empty:
        return df
    max_neg = len(pos) * max_ratio
    if len(neg) <= max_neg:
        return df
    keep_neg = neg.sample(n=max_neg, random_state=RANDOM_SEED)
    combined = pd.concat([pos, keep_neg], ignore_index=True)
    return combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def balance_labels_by_group(df: pd.DataFrame, group_col: str, max_ratio: int) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return balance_labels(df, max_ratio)
    frames: list[pd.DataFrame] = []
    for _, sub_df in df.groupby(group_col, sort=False):
        frames.append(balance_labels(sub_df, max_ratio))
    if not frames:
        return df
    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def load_registry() -> pd.DataFrame:
    registry_path = META_DIR / "table_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}. Run generate_datalake.py first.")
    return pd.read_csv(registry_path)


def build_tables_by_dataset(registry: pd.DataFrame) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for row in registry.itertuples(index=False):
        mapping.setdefault(str(row.dataset), []).append(str(row.filename))
    return mapping


def get_table_columns(
    table_name: str,
    cache: Dict[str, List[str]],
) -> List[str]:
    if table_name in cache:
        return cache[table_name]
    table_path = DATALAKE_DIR / table_name
    df = pd.read_csv(table_path, nrows=0)
    cols = df.columns.astype(str).tolist()
    cache[table_name] = cols
    return cols


def process_entity_matching_groundtruth(
    df: pd.DataFrame,
    dataset_map: Dict[str, str],
    max_ratio: int,
) -> pd.DataFrame:
    required_cols = ["ltable_name", "l_id", "rtable_name", "r_id", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out = df[required_cols].copy()
    out["label"] = out["label"].fillna(0).astype(int)
    out["l_id"] = out["l_id"].astype(int)
    out["r_id"] = out["r_id"].astype(int)
    out.insert(0, "dataset_name", out["ltable_name"].map(lambda t: str(dataset_map.get(str(t), ""))))
    out = balance_labels_by_group(out, "dataset_name", max_ratio)
    return out.reset_index(drop=True)


def process_schema_matching(
    df: pd.DataFrame,
    dataset_map: Dict[str, str],
    tables_by_dataset: Dict[str, List[str]],
    rng: np.random.Generator,
    neg_per_pos: int,
    max_ratio: int,
) -> pd.DataFrame:
    required_cols = [
        "table_name_1",
        "renamed_column_name_1",
        "table_name_2",
        "renamed_column_name_2",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    working = df.copy()
    if "label" not in working.columns:
        working["label"] = 1
    working["label"] = working["label"].fillna(1).astype(int)
    positives = working[working["label"] == 1]

    pos_set = set(
        (row.table_name_1, row.renamed_column_name_1, row.table_name_2, row.renamed_column_name_2)
        for row in positives.itertuples(index=False)
    )
    for row in positives.itertuples(index=False):
        pos_set.add((row.table_name_2, row.renamed_column_name_2, row.table_name_1, row.renamed_column_name_1))

    columns_cache: Dict[str, List[str]] = {}
    origin_to_targets: Dict[str, List[str]] = {}
    mapped_targets: Dict[tuple[str, str, str], set[str]] = {}
    for row in positives.itertuples(index=False):
        t1 = str(row.table_name_1)
        c1 = str(row.renamed_column_name_1)
        t2 = str(row.table_name_2)
        c2 = str(row.renamed_column_name_2)
        origin_to_targets.setdefault(t1, []).append(t2)
        mapped_targets.setdefault((t1, t2, c1), set()).add(c2)
        mapped_targets.setdefault((t2, t1, c2), set()).add(c1)
    for t1, targets in list(origin_to_targets.items()):
        origin_to_targets[t1] = sorted(set(targets))

    negatives: List[Dict[str, object]] = []
    neg_set: set[tuple[str, str, str, str]] = set()
    for row in positives.itertuples(index=False):
        for _ in range(neg_per_pos):
            # Prefer "hard" negatives: keep the same (table_1, table_2) pair but mismatch columns.
            t1 = str(row.table_name_1)
            c1_pos = str(row.renamed_column_name_1)
            t2 = str(row.table_name_2)
            c2_pos = str(row.renamed_column_name_2)

            cols1 = get_table_columns(t1, columns_cache)
            cols2 = get_table_columns(t2, columns_cache)
            if not cols1 or not cols2:
                continue

            found = False

            # 1) Fix c1 (source) and sample an incorrect target column in t2.
            banned_targets = mapped_targets.get((t1, t2, c1_pos), set())
            for _ in range(30):
                c2_neg = cols2[int(rng.integers(len(cols2)))]
                if c2_neg in banned_targets:
                    continue
                key = (t1, c1_pos, t2, c2_neg)
                if key in pos_set or key in neg_set:
                    continue
                neg_set.add(key)
                negatives.append(
                    {
                        "table_name_1": t1,
                        "renamed_column_name_1": c1_pos,
                        "table_name_2": t2,
                        "renamed_column_name_2": c2_neg,
                        "label": 0,
                    }
                )
                found = True
                break

            # 2) Fix c2 (target) and sample an incorrect source column in t1.
            if not found:
                banned_sources = mapped_targets.get((t2, t1, c2_pos), set())
                for _ in range(30):
                    c1_neg = cols1[int(rng.integers(len(cols1)))]
                    if c1_neg in banned_sources:
                        continue
                    key = (t1, c1_neg, t2, c2_pos)
                    if key in pos_set or key in neg_set:
                        continue
                    neg_set.add(key)
                    negatives.append(
                        {
                            "table_name_1": t1,
                            "renamed_column_name_1": c1_neg,
                            "table_name_2": t2,
                            "renamed_column_name_2": c2_pos,
                            "label": 0,
                        }
                    )
                    found = True
                    break

            # 3) Fallback: stay within the same origin table, but choose a different target table.
            if not found:
                targets = origin_to_targets.get(t1, [])
                if len(targets) > 1:
                    for _ in range(30):
                        t2_alt = targets[int(rng.integers(len(targets)))]
                        if t2_alt == t2:
                            continue
                        cols2_alt = get_table_columns(t2_alt, columns_cache)
                        if not cols2_alt:
                            continue
                        c2_alt = cols2_alt[int(rng.integers(len(cols2_alt)))]
                        key = (t1, c1_pos, t2_alt, c2_alt)
                        if key in pos_set or key in neg_set:
                            continue
                        neg_set.add(key)
                        negatives.append(
                            {
                                "table_name_1": t1,
                                "renamed_column_name_1": c1_pos,
                                "table_name_2": t2_alt,
                                "renamed_column_name_2": c2_alt,
                                "label": 0,
                            }
                        )
                        found = True
                        break

            # 4) Last resort: dataset-local random negative.
            if not found:
                dataset = dataset_map.get(t1) or dataset_map.get(t2)
                candidates = tables_by_dataset.get(str(dataset), []) if dataset else []
                if len(candidates) < 2:
                    continue
                for _ in range(60):
                    t1_rand = candidates[int(rng.integers(len(candidates)))]
                    t2_rand = candidates[int(rng.integers(len(candidates)))]
                    if t1_rand == t2_rand:
                        continue
                    cols1_rand = get_table_columns(t1_rand, columns_cache)
                    cols2_rand = get_table_columns(t2_rand, columns_cache)
                    if not cols1_rand or not cols2_rand:
                        continue
                    c1_rand = cols1_rand[int(rng.integers(len(cols1_rand)))]
                    c2_rand = cols2_rand[int(rng.integers(len(cols2_rand)))]
                    key = (t1_rand, c1_rand, t2_rand, c2_rand)
                    if key in pos_set or key in neg_set:
                        continue
                    neg_set.add(key)
                    negatives.append(
                        {
                            "table_name_1": t1_rand,
                            "renamed_column_name_1": c1_rand,
                            "table_name_2": t2_rand,
                            "renamed_column_name_2": c2_rand,
                            "label": 0,
                        }
                    )
                    break

    combined = pd.concat([working, pd.DataFrame(negatives)], ignore_index=True)
    combined.insert(0, "dataset_name", combined["table_name_1"].map(lambda t: str(dataset_map.get(str(t), ""))))
    combined = balance_labels_by_group(combined, "dataset_name", max_ratio)
    return combined.reset_index(drop=True)


def process_unionable_table_search(
    df: pd.DataFrame,
    tables_by_dataset: Dict[str, List[str]],
    rng: np.random.Generator,
    neg_per_pos: int,
    max_ratio: int,
) -> pd.DataFrame:
    required_cols = ["table_name", "dataset_name", "unionable_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    _ = tables_by_dataset

    groups_by_dataset: Dict[str, Dict[int, List[str]]] = {}
    for row in df.itertuples(index=False):
        dataset = str(row.dataset_name)
        groups_by_dataset.setdefault(dataset, {}).setdefault(int(row.unionable_id), []).append(str(row.table_name))

    records: List[Dict[str, object]] = []
    for dataset, groups in groups_by_dataset.items():
        positives: List[Dict[str, object]] = []
        pos_set: set[tuple[str, str]] = set()
        for tables in groups.values():
            if len(tables) < 2:
                continue
            tables = sorted(set(tables))
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    pair = (tables[i], tables[j])
                    pos_set.add(pair)
                    pos_set.add((pair[1], pair[0]))
                    positives.append(
                        {"dataset_name": dataset, "table_name_1": pair[0], "table_name_2": pair[1], "label": 1}
                    )

        table_names = sorted({t for tables in groups.values() for t in tables})
        if not table_names:
            continue

        negatives: List[Dict[str, object]] = []
        for _ in range(len(positives) * neg_per_pos):
            for _ in range(60):
                t1 = table_names[int(rng.integers(len(table_names)))]
                t2 = table_names[int(rng.integers(len(table_names)))]
                if t1 == t2:
                    continue
                if (t1, t2) in pos_set:
                    continue
                pos_set.add((t1, t2))
                negatives.append({"dataset_name": dataset, "table_name_1": t1, "table_name_2": t2, "label": 0})
                break

        records.extend(positives)
        records.extend(negatives)

    combined = pd.DataFrame(records, columns=["dataset_name", "table_name_1", "table_name_2", "label"])
    combined = balance_labels_by_group(combined, "dataset_name", max_ratio)
    return combined.reset_index(drop=True)


def process_joinable_table_search(
    df: pd.DataFrame,
    dataset_map: Dict[str, str],
    tables_by_dataset: Dict[str, List[str]],
    rng: np.random.Generator,
    neg_per_pos: int,
    max_ratio: int,
) -> pd.DataFrame:
    required_cols = ["table_name_1", "column_name_1", "table_name_2", "column_name_2"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    working = df.copy()
    if "label" not in working.columns:
        working["label"] = 1
    if "ratio" not in working.columns:
        working["ratio"] = working["label"].astype(float)

    positives = working[working["label"] == 1]
    pos_set = set(
        (row.table_name_1, row.column_name_1, row.table_name_2, row.column_name_2)
        for row in positives.itertuples(index=False)
    )
    for row in positives.itertuples(index=False):
        pos_set.add((row.table_name_2, row.column_name_2, row.table_name_1, row.column_name_1))

    columns_cache: Dict[str, List[str]] = {}
    negatives: List[Dict[str, object]] = []
    for row in positives.itertuples(index=False):
        dataset = dataset_map.get(row.table_name_1)
        candidates = tables_by_dataset.get(dataset, [])
        if len(candidates) < 2:
            continue
        for _ in range(neg_per_pos):
            for _ in range(30):
                t1 = candidates[int(rng.integers(len(candidates)))]
                t2 = candidates[int(rng.integers(len(candidates)))]
                if t1 == t2:
                    continue
                cols1 = get_table_columns(t1, columns_cache)
                cols2 = get_table_columns(t2, columns_cache)
                if not cols1 or not cols2:
                    continue
                c1 = cols1[int(rng.integers(len(cols1)))]
                c2 = cols2[int(rng.integers(len(cols2)))]
                key = (t1, c1, t2, c2)
                if key in pos_set:
                    continue
                pos_set.add(key)
                negatives.append(
                    {
                        "table_name_1": t1,
                        "column_name_1": c1,
                        "table_name_2": t2,
                        "column_name_2": c2,
                        "ratio": 0.0,
                        "label": 0,
                    }
                )
                break

    combined = pd.concat([working, pd.DataFrame(negatives)], ignore_index=True)
    combined.insert(0, "dataset_name", combined["table_name_1"].map(lambda t: str(dataset_map.get(str(t), ""))))
    combined = balance_labels_by_group(combined, "dataset_name", max_ratio)
    return combined.reset_index(drop=True)


def write_dataset_splits(
    df: pd.DataFrame,
    task_dir: Path,
    dataset_to_split: Dict[str, str],
    dataset_col: str = "dataset_name",
) -> None:
    if dataset_col not in df.columns:
        raise ValueError(f"Missing {dataset_col} in dataframe for {task_dir.name}")
    task_dir.mkdir(parents=True, exist_ok=True)
    split_names = ["train", "validate", "test"]
    for split in split_names:
        mask = df[dataset_col].map(lambda d: dataset_to_split.get(str(d), "")) == split
        out_df = df[mask].copy()
        out_df = out_df.drop(columns=[dataset_col])
        out_df.to_csv(task_dir / f"{split}.csv", index=False)


def write_splits(
    df: pd.DataFrame,
    task_dir: Path,
    train_ratio: float,
    validate_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
    drop_cols: tuple[str, ...] = (),
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    clean_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    train_df, val_df, test_df = split_dataset(clean_df, train_ratio, validate_ratio, test_ratio, rng)
    train_df.to_csv(task_dir / "train.csv", index=False)
    val_df.to_csv(task_dir / "validate.csv", index=False)
    test_df.to_csv(task_dir / "test.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/validate/test splits for magellan.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default=SPLIT_MODE,
        choices=["random", "dataset-disjoint"],
        help="Split mode: random does record-level random split (tables can appear in multiple splits); "
        "dataset-disjoint keeps entire datasets isolated across splits.",
    )
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--validate-ratio", type=float, default=VALIDATE_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--neg-per-pos", type=int, default=1)
    parser.add_argument("--max-neg-ratio", type=int, default=LABEL_BALANCE_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    registry = load_registry()
    dataset_map = dict(zip(registry["filename"], registry["dataset"]))
    tables_by_dataset = build_tables_by_dataset(registry)

    dataset_to_split: Dict[str, str] = {}
    if args.split_mode == "dataset-disjoint":
        datasets = registry["dataset"].dropna().astype(str).unique().tolist()
        dataset_to_split = assign_groups_to_splits(
            datasets,
            args.train_ratio,
            args.validate_ratio,
            args.test_ratio,
            rng,
        )

    # Entity matching (uses Magellan ground truth)
    entity_labels = pd.read_csv(LABEL_ROOT / "entity_matching" / "entity_matching_labels.csv")
    entity_df = process_entity_matching_groundtruth(entity_labels, dataset_map, args.max_neg_ratio)
    if args.split_mode == "dataset-disjoint":
        write_dataset_splits(entity_df, LABEL_ROOT / "entity_matching", dataset_to_split)
    else:
        write_splits(
            entity_df,
            LABEL_ROOT / "entity_matching",
            args.train_ratio,
            args.validate_ratio,
            args.test_ratio,
            rng,
            drop_cols=("dataset_name",),
        )

    # Schema matching
    schema_labels = pd.read_csv(LABEL_ROOT / "schema_matching" / "schema_matching_labels.csv")
    schema_df = process_schema_matching(
        schema_labels,
        dataset_map,
        tables_by_dataset,
        rng,
        args.neg_per_pos,
        args.max_neg_ratio,
    )
    if args.split_mode == "dataset-disjoint":
        write_dataset_splits(schema_df, LABEL_ROOT / "schema_matching", dataset_to_split)
    else:
        write_splits(
            schema_df,
            LABEL_ROOT / "schema_matching",
            args.train_ratio,
            args.validate_ratio,
            args.test_ratio,
            rng,
            drop_cols=("dataset_name",),
        )

    # Unionable table search
    union_labels = pd.read_csv(LABEL_ROOT / "unionable_table_search" / "unionable_table_search_labels.csv")
    union_df = process_unionable_table_search(
        union_labels,
        tables_by_dataset,
        rng,
        args.neg_per_pos,
        args.max_neg_ratio,
    )
    if args.split_mode == "dataset-disjoint":
        write_dataset_splits(union_df, LABEL_ROOT / "unionable_table_search", dataset_to_split)
    else:
        write_splits(
            union_df,
            LABEL_ROOT / "unionable_table_search",
            args.train_ratio,
            args.validate_ratio,
            args.test_ratio,
            rng,
            drop_cols=("dataset_name",),
        )

    # Joinable table search
    join_labels = pd.read_csv(LABEL_ROOT / "joinable_table_search" / "joinable_table_search_labels.csv")
    join_df = process_joinable_table_search(
        join_labels,
        dataset_map,
        tables_by_dataset,
        rng,
        args.neg_per_pos,
        args.max_neg_ratio,
    )
    if args.split_mode == "dataset-disjoint":
        write_dataset_splits(join_df, LABEL_ROOT / "joinable_table_search", dataset_to_split)
    else:
        write_splits(
            join_df,
            LABEL_ROOT / "joinable_table_search",
            args.train_ratio,
            args.validate_ratio,
            args.test_ratio,
            rng,
            drop_cols=("dataset_name",),
        )

    print(f"Splits written to label_plus/<task>/train|validate|test.csv ({args.split_mode})")


if __name__ == "__main__":
    main()
