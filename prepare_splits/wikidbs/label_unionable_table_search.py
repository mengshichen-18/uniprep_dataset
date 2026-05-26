import argparse
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
META_DIR = BASE_DIR / "metadata"
LABEL_DIR = BASE_DIR / "label_plus" / "unionable_table_search"


def extract_prefix(filename: str) -> int:
    prefix = filename.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError(f"Unexpected table name format: {filename}")
    return int(prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unionable table search labels.")
    args = parser.parse_args()

    derived_map_path = META_DIR / "derived_map.csv"
    if not derived_map_path.exists():
        raise FileNotFoundError(f"Missing {derived_map_path}. Run generate_datalake.py first.")

    df = pd.read_csv(derived_map_path)
    records: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        origin = row.origin_table
        derived = row.derived_table
        dataset_name = row.dataset_name
        unionable_id = extract_prefix(origin)

        records.append(
            {
                "table_name": origin,
                "dataset_name": dataset_name,
                "unionable_id": unionable_id,
            }
        )
        records.append(
            {
                "table_name": derived,
                "dataset_name": dataset_name,
                "unionable_id": unionable_id,
            }
        )

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABEL_DIR / "unionable_table_search_labels.csv"
    pd.DataFrame(
        records,
        columns=["table_name", "dataset_name", "unionable_id"],
    ).to_csv(out_path, index=False)
    print(f"Wrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()

