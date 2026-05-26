#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import os


ROOT = Path(os.environ.get("DATASET_ROOT", str(Path(__file__).resolve().parent)))
DATASET_MAP = {
    "wikidbs_1218": "wikidbs_433",
    "magellan_1218": "magellan_433",
    "santos_benchmark_1218": "santos_benchmark_433",
}
SKIP_NAMES = {"__pycache__", "datalake_plus", "metadata", "extracted_images", "label_plus"}


def copy_entry(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst)


def symlink_dir(src: Path, dst: Path) -> None:
    dst.symlink_to(src, target_is_directory=True)


def choose_python() -> str:
    candidates = [Path(sys.executable)]
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        probe = subprocess.run(
            [key, "-c", "import pandas, numpy; print('ok')"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return key
    raise RuntimeError("No usable Python interpreter found with pandas and numpy installed.")


def create_target(source_dir: Path, target_dir: Path, *, force: bool) -> None:
    if target_dir.exists() or target_dir.is_symlink():
        if not force:
            raise FileExistsError(f"Target already exists: {target_dir}")
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=False)

    for entry in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if entry.name in SKIP_NAMES:
            continue
        copy_entry(entry, target_dir / entry.name)

    shutil.copytree(source_dir / "label_plus", target_dir / "label_plus", symlinks=False)
    symlink_dir(source_dir / "datalake_plus", target_dir / "datalake_plus")
    symlink_dir(source_dir / "metadata", target_dir / "metadata")
    if (source_dir / "extracted_images").exists():
        symlink_dir(source_dir / "extracted_images", target_dir / "extracted_images")


def run_split_generator(
    target_dir: Path,
    *,
    python_bin: str,
    train_ratio: float,
    validate_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    cmd = [
        python_bin,
        str(target_dir / "dataset_splits.py"),
        "--train-ratio",
        str(train_ratio),
        "--validate-ratio",
        str(validate_ratio),
        "--test-ratio",
        str(test_ratio),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, cwd=target_dir, check=True)


def write_manifest(source_dir: Path, target_dir: Path, *, train_ratio: float, validate_ratio: float, test_ratio: float, seed: int) -> None:
    manifest = {
        "source_dataset": source_dir.name,
        "target_dataset": target_dir.name,
        "source_root": str(source_dir),
        "target_root": str(target_dir),
        "split_ratios": {
            "train": train_ratio,
            "validate": validate_ratio,
            "test": test_ratio,
        },
        "seed": seed,
        "bulky_dirs_reused_via_symlink": [
            name for name in ("datalake_plus", "metadata", "extracted_images") if (target_dir / name).exists()
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target_dir / "split_433_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create *_433 datasets from *_1218 while keeping datalake/metadata unchanged and regenerating only split files."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing *_433 directories.")
    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--validate-ratio", type=float, default=0.3)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.train_ratio + args.validate_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train/validate/test ratios must sum to 1.0")

    python_bin = choose_python()
    print(f"[python] {python_bin}")

    for source_name, target_name in DATASET_MAP.items():
        source_dir = ROOT / source_name
        target_dir = ROOT / target_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source dataset: {source_dir}")

        print(f"[create] {source_name} -> {target_name}")
        create_target(source_dir, target_dir, force=args.force)
        run_split_generator(
            target_dir,
            python_bin=python_bin,
            train_ratio=args.train_ratio,
            validate_ratio=args.validate_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        write_manifest(
            source_dir,
            target_dir,
            train_ratio=args.train_ratio,
            validate_ratio=args.validate_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(f"[done] {target_dir}")


if __name__ == "__main__":
    main()
