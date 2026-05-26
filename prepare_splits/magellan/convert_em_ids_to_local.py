#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "entity_matching"

CHUNK_STEM_RE = re.compile(r"^(tableA|tableB)_colsize(?P<colsize>\d+)_chunk(?P<chunk>\d+)$")


@dataclass(frozen=True)
class ChunkInfo:
    start_id: int
    end_id: int
    row_count: int


def load_chunk_ranges() -> Dict[str, ChunkInfo]:
    registry_path = META_DIR / "table_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}. Run generate_datalake.py first.")

    originals: List[Tuple[str, str, int, int, str, int]] = []
    # (dataset, side, colsize, chunk, filename, row_count)
    with registry_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kind") != "original":
                continue
            stem = str(row.get("table_stem", ""))
            match = CHUNK_STEM_RE.match(stem)
            if not match:
                continue
            dataset = str(row.get("dataset", ""))
            side = match.group(1)
            colsize = int(match.group("colsize"))
            chunk = int(match.group("chunk"))
            filename = str(row.get("filename", ""))
            row_count = int(row.get("row_count", "0") or 0)
            originals.append((dataset, side, colsize, chunk, filename, row_count))

    by_dataset_side: Dict[Tuple[str, str], Dict[int, List[Tuple[int, str, int]]]] = {}
    for dataset, side, colsize, chunk, filename, row_count in originals:
        by_dataset_side.setdefault((dataset, side), {}).setdefault(colsize, []).append((chunk, filename, row_count))

    ranges: Dict[str, ChunkInfo] = {}
    for (_dataset, _side), by_colsize in by_dataset_side.items():
        max_colsize = max(by_colsize.keys())
        entries = sorted(by_colsize[max_colsize], key=lambda item: item[0])
        offset = 0
        for _chunk_idx, filename, row_count in entries:
            row_count = max(int(row_count), 0)
            ranges[filename] = ChunkInfo(
                start_id=offset,
                end_id=offset + row_count - 1,
                row_count=row_count,
            )
            offset += row_count
    return ranges


def normalize_one_id(raw_id: int, info: ChunkInfo) -> tuple[int, bool]:
    # If id is in chunk-global range, convert to local offset.
    if info.start_id <= raw_id <= info.end_id:
        local_id = raw_id - info.start_id
        return local_id, True
    # Otherwise assume it is already local.
    return raw_id, False


def convert_file(path: Path, chunk_ranges: Dict[str, ChunkInfo], dry_run: bool) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {"ltable_name", "l_id", "rtable_name", "r_id", "label"}
        if not required.issubset(set(fieldnames)):
            raise ValueError(f"Unexpected columns in {path}: {fieldnames}")

        rows: list[dict[str, str]] = []
        changed_count = 0
        total_rows = 0
        for row in reader:
            total_rows += 1
            ltable = row["ltable_name"]
            rtable = row["rtable_name"]
            if ltable not in chunk_ranges:
                raise KeyError(f"{path}: unknown ltable_name={ltable}")
            if rtable not in chunk_ranges:
                raise KeyError(f"{path}: unknown rtable_name={rtable}")

            l_raw = int(row["l_id"])
            r_raw = int(row["r_id"])
            l_local, l_converted = normalize_one_id(l_raw, chunk_ranges[ltable])
            r_local, r_converted = normalize_one_id(r_raw, chunk_ranges[rtable])

            l_info = chunk_ranges[ltable]
            r_info = chunk_ranges[rtable]
            if not (0 <= l_local < l_info.row_count):
                raise ValueError(
                    f"{path}: local l_id out of range for {ltable}: raw={l_raw}, local={l_local}, row_count={l_info.row_count}"
                )
            if not (0 <= r_local < r_info.row_count):
                raise ValueError(
                    f"{path}: local r_id out of range for {rtable}: raw={r_raw}, local={r_local}, row_count={r_info.row_count}"
                )

            if l_converted and l_local != l_raw:
                changed_count += 1
            if r_converted and r_local != r_raw:
                changed_count += 1

            row["l_id"] = str(l_local)
            row["r_id"] = str(r_local)
            rows.append(row)

    if not dry_run:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return total_rows, changed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Magellan EM ids from global ids to chunk-local row ids (idempotent for already-local files)."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "entity_matching_labels.csv",
            "train.csv",
            "validate.csv",
            "test.csv",
        ],
        help="Target files under label_plus/entity_matching/",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only; do not overwrite files.")
    args = parser.parse_args()

    chunk_ranges = load_chunk_ranges()
    for name in args.files:
        path = LABEL_DIR / name
        total_rows, changed = convert_file(path, chunk_ranges, dry_run=args.dry_run)
        mode = "CHECKED" if args.dry_run else "UPDATED"
        print(f"{mode} {path}: rows={total_rows}, changed_id_fields={changed}")


if __name__ == "__main__":
    main()
