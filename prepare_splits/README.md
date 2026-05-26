# Prepare Splits

Scripts to build `datalake_plus/`, `label_plus/`, and `metadata/` from raw original datasets, and generate train/validate/test splits. The output feeds directly into the graph build pipeline (`run_no_token.sh`).

## Prerequisites

- Python 3.9+ with `pandas`, `numpy` installed
- The Valentine library (used by `generate_datalake.py`; local Python modules only)
- Original dataset files (downloaded from Google Drive — see root README)

Set these environment variables before running:

```bash
export DATASET_ROOT=/path/to/your/datasets_dir   # contains magellan_ori/, santos_benchmark_ori/, wikidbs/, valentine/
export DATASET_DIR=/path/to/your/datasets_dir/<dataset>   # specific output dataset directory
export VALENTINE_DIR=/path/to/valentine           # optional override for Valentine library path
```

Each script reads from `DATASET_ROOT` (for source data and Valentine) and writes to `DATASET_DIR` (the target dataset directory). If `DATASET_DIR` is not set, scripts fall back to their own directory — which works when running them from the original dataset location.

## Pipeline per Dataset

The steps are the same for all three datasets. Run sequentially within each dataset:

### Magellan

```bash
export DATASET_ROOT=/data/datasets
export DATASET_DIR=${DATASET_ROOT}/magellan

python prepare_splits/magellan/generate_datalake.py
python prepare_splits/magellan/label_entity_matching.py
python prepare_splits/magellan/label_schema_matching.py
python prepare_splits/magellan/label_unionable_table_search.py
python prepare_splits/magellan/label_joinable_table_search.py
python prepare_splits/magellan/dataset_splits.py
```

### Santos Benchmark

```bash
export DATASET_ROOT=/data/datasets
export DATASET_DIR=${DATASET_ROOT}/santos_benchmark

python prepare_splits/santos_benchmark/generate_datalake.py
python prepare_splits/santos_benchmark/label_entity_matching.py
python prepare_splits/santos_benchmark/label_schema_matching.py
python prepare_splits/santos_benchmark/label_unionable_table_search.py
python prepare_splits/santos_benchmark/label_joinable_table_search.py
python prepare_splits/santos_benchmark/dataset_splits.py
```

### WikiDBs

```bash
export DATASET_ROOT=/data/datasets
export DATASET_DIR=${DATASET_ROOT}/wikidbs

python prepare_splits/wikidbs/generate_datalake.py
python prepare_splits/wikidbs/label_entity_matching.py
python prepare_splits/wikidbs/label_unionable_table_search.py
python prepare_splits/wikidbs/label_joinable_from_gpt.py
python prepare_splits/wikidbs/extract_schema_matching_llm.py --resume
python prepare_splits/wikidbs/combine_schema_matching_llm.py --neg-per-pos 0
python prepare_splits/wikidbs/dataset_splits.py
```

Notes for WikiDBs:
- `extract_schema_matching_llm.py` calls an LLM API (DeepSeek by default). Set `DEEPSEEK_API_KEY` or place your key in `$DATASET_DIR/.deepseek_key`.
- `label_joinable_from_gpt.py` reuses pre-extracted GPT results at `$DATASET_ROOT/wikidbs/label/joinable_table_search/GPT_extracted_results.csv`.

## Optional: Re-split with custom ratios

`resplit_dataset.py` creates new dataset directories from existing ones with a different train/validate/test ratio. It copies all files, symlinks the bulky `datalake_plus/` and `metadata/` directories, and re-runs `dataset_splits.py`:

```bash
DATASET_ROOT=/data/datasets python prepare_splits/resplit_dataset.py \
    magellan_orig:magellan santos_benchmark_orig:santos_benchmark wikidbs_orig:wikidbs \
    --train-ratio 0.4 --validate-ratio 0.3 --test-ratio 0.3
```

## Connecting to the Build Graph Pipeline

After splits are ready, run from the repo root:

```bash
DATASET_ROOT=/data/datasets bash run_no_token.sh --all
```

The output directories (`<dataset>_no_token/`) are used as `GRAPH_DIR` in the Uniprep training pipeline.
