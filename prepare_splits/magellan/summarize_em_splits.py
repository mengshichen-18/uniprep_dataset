#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Tuple
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
LABEL_DIR = BASE_DIR / "label_plus" / "entity_matching"

TABLE_RE = re.compile(r"^\d+_(?P<dataset>.+?)_table[AB]_colsize\d+_chunk\d+\.csv$")


def extract_dataset(table_filename: str) -> str:
    name = Path(table_filename).name
    match = TABLE_RE.match(name)
    if not match:
        return ""
    return str(match.group("dataset"))


def load_counts(path: Path) -> Dict[str, Tuple[int, int]]:
    counts: DefaultDict[str, list[int]] = defaultdict(lambda: [0, 0])  # pos, neg
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = extract_dataset(row.get("ltable_name", ""))
            if not dataset:
                continue
            label = int(row.get("label", "0") or 0)
            if label == 1:
                counts[dataset][0] += 1
            else:
                counts[dataset][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Magellan_1218 entity-matching split sizes by dataset.")
    parser.add_argument("--dir", type=str, default=str(LABEL_DIR), help="label_plus/entity_matching directory")
    args = parser.parse_args()

    label_dir = Path(args.dir)
    for split in ["train", "validate", "test"]:
        path = label_dir / f"{split}.csv"
        if not path.exists():
            print(f"[{split}] missing {path}")
            continue

        counts = load_counts(path)
        total_pos = sum(p for p, _ in counts.values())
        total_neg = sum(n for _, n in counts.values())
        total = total_pos + total_neg
        print(f"\n[{split}] total={total} pos={total_pos} neg={total_neg} pos_rate={total_pos/total if total else 0:.4f}")

        if not counts:
            continue

        # sort by total desc
        items = sorted(counts.items(), key=lambda kv: (kv[1][0] + kv[1][1]), reverse=True)
        for dataset, (pos, neg) in items:
            t = pos + neg
            share = t / total if total else 0
            print(f"  - {dataset}: total={t} pos={pos} neg={neg} pos_rate={pos/t if t else 0:.4f} share={share:.4%}")


if __name__ == "__main__":
    main()

