#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
LABEL_DIR = BASE_DIR / "label_plus" / "entity_matching"
META_DIR = BASE_DIR / "metadata"

CHUNK_STEM_RE = re.compile(r"^(tableA|tableB)_colsize(?P<colsize>\d+)_chunk(?P<chunk>\d+)$")


@dataclass(frozen=True)
class ChunkInfo:
    start_id: int
    end_id: int
    row_count: int


def iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def reservoir_sample(rows: Iterable[dict[str, str]], k: int, rng: random.Random) -> List[dict[str, str]]:
    sample: List[dict[str, str]] = []
    for i, row in enumerate(rows):
        if i < k:
            sample.append(row)
            continue
        j = rng.randint(0, i)
        if j < k:
            sample[j] = row
    return sample


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
        if not by_colsize:
            continue
        max_colsize = max(by_colsize.keys())
        entries = sorted(by_colsize[max_colsize], key=lambda item: item[0])
        offset = 0
        for _chunk_idx, filename, row_count in entries:
            row_count = max(int(row_count), 0)
            start_id = offset
            end_id = offset + row_count - 1
            ranges[filename] = ChunkInfo(start_id=start_id, end_id=end_id, row_count=row_count)
            offset += row_count

    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check Magellan entity-matching id semantics.\n\n"
            "It samples rows from label_plus/entity_matching/{train,validate,test}.csv and shows:\n"
            "- whether l_id/r_id are valid chunk-local row indices\n"
            "- recovered global ids via: global_id = chunk_start_id + local_id\n"
        )
    )
    parser.add_argument("--split", choices=["train", "validate", "test"], default="test")
    parser.add_argument("--num", type=int, default=10, help="Number of sampled label rows to print.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_path = LABEL_DIR / f"{args.split}.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}. Run dataset_splits.py first.")

    ranges = load_chunk_ranges()
    rng = random.Random(args.seed)
    samples = reservoir_sample(iter_csv_rows(label_path), k=max(int(args.num), 1), rng=rng)

    print(f"Sampled {len(samples)} rows from {label_path}\n")
    for i, row in enumerate(samples, start=1):
        ltable = row["ltable_name"]
        rtable = row["rtable_name"]
        l_id = int(row["l_id"])
        r_id = int(row["r_id"])
        label = int(row["label"])

        linfo = ranges.get(ltable)
        rinfo = ranges.get(rtable)
        if linfo is None or rinfo is None:
            print(f"[{i}] label={label}  (missing chunk info: {ltable if linfo is None else ''} {rtable if rinfo is None else ''})")
            continue

        l_local_ok = 0 <= l_id < linfo.row_count
        r_local_ok = 0 <= r_id < rinfo.row_count
        l_global = linfo.start_id + l_id
        r_global = rinfo.start_id + r_id
        l_global_ok = linfo.start_id <= l_global <= linfo.end_id
        r_global_ok = rinfo.start_id <= r_global <= rinfo.end_id

        print(f"[{i}] label={label}")
        print(
            f"  L: {ltable}  row_count={linfo.row_count}  "
            f"l_id(local)={l_id}  local_ok={l_local_ok}  "
            f"global_id(recovered)={l_global}  global_range=[{linfo.start_id},{linfo.end_id}] ok={l_global_ok}"
        )
        print(
            f"  R: {rtable}  row_count={rinfo.row_count}  "
            f"r_id(local)={r_id}  local_ok={r_local_ok}  "
            f"global_id(recovered)={r_global}  global_range=[{rinfo.start_id},{rinfo.end_id}] ok={r_global_ok}"
        )
        print()

    print(
        "Notes:\n"
        "- For Magellan_1218 EM labels, l_id/r_id should be chunk-local indices.\n"
        "- To map back to original Magellan ids, use: global_id = chunk_start_id + local_id.\n"
    )


if __name__ == "__main__":
    main()
