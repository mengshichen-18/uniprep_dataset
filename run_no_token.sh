#!/usr/bin/env bash
# Sequentially run the no-token graph pipeline scripts from the data directory.

# /home/mengshi/table_quality/datasets_joint_discovery_integration/santos_benchmark_1218/datalake_plus
# /home/mengshi/table_quality/datasets_joint_discovery_integration/santos_benchmark_1218/label_plus
# /home/mengshi/table_quality/datasets_joint_discovery_integration/magellan_1218/label_plus
# /home/mengshi/table_quality/datasets_joint_discovery_integration/magellan_1218/datalake_plus
# /home/mengshi/table_quality/datasets_joint_discovery_integration/wikidbs_1218/label_plus
# /home/mengshi/table_quality/datasets_joint_discovery_integration/wikidbs_1218/datalake_plus


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [DATASET|--dataset DATASET ...|--all]

DATASET:
  santos_benchmark   (default)
  magellan
  wikidbs

Examples:
  bash data/run_no_token.sh
  bash data/run_no_token.sh --all
  bash data/run_no_token.sh magellan
  bash data/run_no_token.sh --dataset wikidbs
EOF
}

declare -A DATALAKE_PATHS=(
  [santos_benchmark]="/home/mengshi/table_quality/datasets_joint_discovery_integration/santos_benchmark_1218/datalake_plus"
  [magellan]="/home/mengshi/table_quality/datasets_joint_discovery_integration/magellan_1218/datalake_plus"
  [wikidbs]="/home/mengshi/table_quality/datasets_joint_discovery_integration/wikidbs_1218/datalake_plus"
)
declare -A LABEL_PATHS=(
  [santos_benchmark]="/home/mengshi/table_quality/datasets_joint_discovery_integration/santos_benchmark_1218/label_plus"
  [magellan]="/home/mengshi/table_quality/datasets_joint_discovery_integration/magellan_1218/label_plus"
  [wikidbs]="/home/mengshi/table_quality/datasets_joint_discovery_integration/wikidbs_1218/label_plus"
)
declare -A OUTPUT_DIRS=(
  [santos_benchmark]="$SCRIPT_DIR/santos_benchmark_no_token"
  [magellan]="$SCRIPT_DIR/magellan_no_token"
  [wikidbs]="$SCRIPT_DIR/wikidbs_no_token"
)

datasets=()
if [[ $# -eq 0 ]]; then
  datasets=("santos_benchmark")
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --all)
        datasets=("santos_benchmark" "magellan" "wikidbs")
        shift
        ;;
      --dataset)
        shift
        if [[ $# -eq 0 ]]; then
          echo "Error: --dataset requires a value." >&2
          usage >&2
          exit 2
        fi
        datasets+=("$1")
        shift
        ;;
      santos_benchmark|magellan|wikidbs)
        datasets+=("$1")
        shift
        ;;
      *)
        echo "Error: Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
fi

# Always run relative to the directory this script lives in.
cd "$SCRIPT_DIR"

for dataset in "${datasets[@]}"; do
  datalake_path="${DATALAKE_PATHS[$dataset]:-}"
  label_path="${LABEL_PATHS[$dataset]:-}"
  output_dir="${OUTPUT_DIRS[$dataset]:-}"

  if [[ -z "$datalake_path" || -z "$label_path" || -z "$output_dir" ]]; then
    echo "Error: Unknown dataset '$dataset'." >&2
    usage >&2
    exit 2
  fi

  echo "==> Dataset: $dataset"
  echo "    datalake_path: $datalake_path"
  echo "    label_path:    $label_path"
  echo "    output_dir:    $output_dir"

  python -u ./build_graph_no_token.py --datalake-path "$datalake_path" --label-path "$label_path" --output-dir "$output_dir"
  python -u ./embedding_edges_no_token.py --data-dir "$output_dir"
  python -u ./embedding_nodes_no_token.py --data-dir "$output_dir" --batch-size 128 --max-length 512
done

