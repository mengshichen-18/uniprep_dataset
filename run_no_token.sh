#!/usr/bin/env bash
# Run the no-token graph pipeline for one or more datasets.
# Reads input from DATASET_ROOT/<dataset>/{datalake_plus,label_plus}.
# Writes graph output to DATASET_ROOT/<dataset>_no_token/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [DATASET|--dataset DATASET ...|--all]

Required environment variables:
  DATASET_ROOT   Directory containing dataset subdirectories

Optional:
  PYTHON_BIN     Python interpreter (default: python3)
  MODEL_PATH     HuggingFace model path for embeddings

DATASET:
  santos_benchmark   (default)
  magellan
  wikidbs

Examples:
  DATASET_ROOT=/data/datasets bash run_no_token.sh
  DATASET_ROOT=/data/datasets bash run_no_token.sh --all
  DATASET_ROOT=/data/datasets bash run_no_token.sh magellan
  DATASET_ROOT=/data/datasets bash run_no_token.sh --dataset wikidbs
EOF
}

if [[ -z "${DATASET_ROOT:-}" ]]; then
  echo "[ERROR] Set DATASET_ROOT to the directory containing dataset subdirectories." >&2
  usage >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

declare -A DATALAKE_PATHS=(
  [santos_benchmark]="${DATASET_ROOT}/santos_benchmark/datalake_plus"
  [magellan]="${DATASET_ROOT}/magellan/datalake_plus"
  [wikidbs]="${DATASET_ROOT}/wikidbs/datalake_plus"
)
declare -A LABEL_PATHS=(
  [santos_benchmark]="${DATASET_ROOT}/santos_benchmark/label_plus"
  [magellan]="${DATASET_ROOT}/magellan/label_plus"
  [wikidbs]="${DATASET_ROOT}/wikidbs/label_plus"
)
declare -A OUTPUT_DIRS=(
  [santos_benchmark]="${DATASET_ROOT}/santos_benchmark_no_token"
  [magellan]="${DATASET_ROOT}/magellan_no_token"
  [wikidbs]="${DATASET_ROOT}/wikidbs_no_token"
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

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/build_graph_no_token.py" \
    --datalake-path "$datalake_path" \
    --label-path "$label_path" \
    --output-dir "$output_dir"

  embed_args=(--data-dir "$output_dir")
  [[ -n "$MODEL_PATH" ]] && embed_args+=(--model-path "$MODEL_PATH")

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/embedding_edges_no_token.py" "${embed_args[@]}"
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/embedding_nodes_no_token.py" "${embed_args[@]}" --batch-size 128 --max-length 512
done
