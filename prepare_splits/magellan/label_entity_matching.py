from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import os


REPO_ROOT = Path(os.environ.get("DATASET_ROOT", str(Path(__file__).resolve().parents[2])))
MAGELLAN_ORI_DIR = REPO_ROOT / "magellan_ori"

BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "entity_matching"

CHUNK_STEM_RE = re.compile(r"^(tableA|tableB)_colsize(?P<colsize>\d+)_chunk(?P<chunk>\d+)$")


def _load_registry() -> pd.DataFrame:
    registry_path = META_DIR / "table_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}. Run generate_datalake.py first.")
    return pd.read_csv(registry_path)


def load_chunk_index() -> dict[str, dict[str, dict[str, object]]]:
    """Return dataset -> side -> {files,start_rows,row_counts} for chunked originals.

    Entity-matching labels in `magellan_ori/*/{train,valid,test}.csv` reference
    `ltable_id/rtable_id` which correspond to original global row ids.
    Since Magellan datalake tables are row-chunked, we map each id to:
    - the chunk table that contains it, and
    - the chunk-local row index used in `label_plus/entity_matching/*.csv`.

    We only use the max-colsize (full-column) chunk variants, consistent with the original script
    which generates EM labels only for the j==0 colsize.
    """
    registry = _load_registry()
    originals = registry[registry["kind"] == "original"].copy()

    by_dataset: dict[str, dict[str, dict[int, list[tuple[int, str, int]]]]] = {}
    for row in originals.itertuples(index=False):
        dataset = str(row.dataset)
        stem = str(row.table_stem)
        match = CHUNK_STEM_RE.match(stem)
        if not match:
            continue
        side = match.group(1)
        colsize = int(match.group("colsize"))
        chunk_idx = int(match.group("chunk"))
        filename = str(row.filename)
        row_count = int(row.row_count)
        by_dataset.setdefault(dataset, {}).setdefault(side, {}).setdefault(colsize, []).append(
            (chunk_idx, filename, row_count)
        )

    chunk_index: dict[str, dict[str, dict[str, object]]] = {}
    for dataset, by_side in by_dataset.items():
        chunk_index[dataset] = {}
        for side, by_colsize in by_side.items():
            if not by_colsize:
                continue
            max_colsize = max(by_colsize.keys())
            entries = sorted(by_colsize[max_colsize], key=lambda item: item[0])

            files: list[str] = []
            start_rows: list[int] = []
            row_counts: list[int] = []
            offset = 0
            for chunk_idx, filename, row_count in entries:
                _ = chunk_idx
                start_rows.append(offset)
                files.append(filename)
                clean_count = max(int(row_count), 0)
                row_counts.append(clean_count)
                offset += clean_count
            start_rows.append(offset + 1)

            chunk_index[dataset][side] = {
                "files": files,
                "start_rows": start_rows,
                "row_counts": row_counts,
                "colsize": max_colsize,
            }

    return chunk_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Magellan entity matching labels from ground truth.")
    args = parser.parse_args()
    _ = args

    chunk_index = load_chunk_index()
    if not MAGELLAN_ORI_DIR.exists():
        raise FileNotFoundError(f"Missing {MAGELLAN_ORI_DIR}")

    frames: list[pd.DataFrame] = []
    for dataset_name in sorted(chunk_index.keys()):
        dataset_info = chunk_index[dataset_name]
        left_info = dataset_info.get("tableA")
        right_info = dataset_info.get("tableB")
        if not left_info or not right_info:
            continue

        left_files = list(left_info["files"])
        left_start = np.asarray(left_info["start_rows"], dtype=int)
        left_counts = np.asarray(left_info["row_counts"], dtype=int)
        right_files = list(right_info["files"])
        right_start = np.asarray(right_info["start_rows"], dtype=int)
        right_counts = np.asarray(right_info["row_counts"], dtype=int)

        dataset_dir = MAGELLAN_ORI_DIR / dataset_name
        split_frames: list[pd.DataFrame] = []
        for split_name in ["train.csv", "valid.csv", "test.csv"]:
            split_path = dataset_dir / split_name
            if not split_path.exists():
                continue
            split_frames.append(pd.read_csv(split_path))
        if not split_frames:
            continue

        df = pd.concat(split_frames, ignore_index=True)
        required = {"ltable_id", "rtable_id", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"Unexpected columns in {dataset_dir}: {df.columns.tolist()}")

        df = df.rename(columns={"ltable_id": "l_id", "rtable_id": "r_id"}).copy()
        df["l_id"] = df["l_id"].astype(int)
        df["r_id"] = df["r_id"].astype(int)
        df["label"] = df["label"].astype(int)

        left_pos = np.searchsorted(left_start, df["l_id"].to_numpy(dtype=int), side="right") - 1
        right_pos = np.searchsorted(right_start, df["r_id"].to_numpy(dtype=int), side="right") - 1
        if (left_pos < 0).any() or (left_pos >= len(left_files)).any():
            bad = df.loc[(left_pos < 0) | (left_pos >= len(left_files)), "l_id"].head(5).tolist()
            raise ValueError(f"Failed to map l_id to chunk for {dataset_name}; sample ids={bad}")
        if (right_pos < 0).any() or (right_pos >= len(right_files)).any():
            bad = df.loc[(right_pos < 0) | (right_pos >= len(right_files)), "r_id"].head(5).tolist()
            raise ValueError(f"Failed to map r_id to chunk for {dataset_name}; sample ids={bad}")

        left_start_per_row = left_start[left_pos]
        right_start_per_row = right_start[right_pos]
        local_lid = df["l_id"].to_numpy(dtype=int) - left_start_per_row
        local_rid = df["r_id"].to_numpy(dtype=int) - right_start_per_row

        if (local_lid < 0).any() or (local_lid >= left_counts[left_pos]).any():
            bad_idx = int(np.where((local_lid < 0) | (local_lid >= left_counts[left_pos]))[0][0])
            raise ValueError(
                "Computed local l_id is out of chunk bounds for "
                f"{dataset_name}; sample=({df.iloc[bad_idx]['l_id']} -> {local_lid[bad_idx]})"
            )
        if (local_rid < 0).any() or (local_rid >= right_counts[right_pos]).any():
            bad_idx = int(np.where((local_rid < 0) | (local_rid >= right_counts[right_pos]))[0][0])
            raise ValueError(
                "Computed local r_id is out of chunk bounds for "
                f"{dataset_name}; sample=({df.iloc[bad_idx]['r_id']} -> {local_rid[bad_idx]})"
            )

        df.insert(0, "ltable_name", np.asarray(left_files, dtype=object)[left_pos])
        df.insert(2, "rtable_name", np.asarray(right_files, dtype=object)[right_pos])
        df["l_id"] = local_lid.astype(int)
        df["r_id"] = local_rid.astype(int)
        df = df[["ltable_name", "l_id", "rtable_name", "r_id", "label"]]
        frames.append(df)

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ltable_name", "l_id", "rtable_name", "r_id", "label"]
    )
    out_df = out_df.drop_duplicates().reset_index(drop=True)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "entity_matching_labels.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
