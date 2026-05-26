# Prepare Splits

Scripts to build `datalake_plus/`, `label_plus/`, and `metadata/` from raw original datasets, and generate train/validate/test splits. The output feeds directly into the graph build pipeline (`run_no_token.sh`).

## Prerequisites

- Python 3.9+ with `pandas`, `numpy` installed
- The Valentine library (used by `generate_datalake.py`; local Python modules only)
- Original dataset files (downloaded from Google Drive — see root README)

Set these environment variables before running:

```bash
export DATASET_ROOT=/path/to/your/datasets_dir   # contains magellan_ori/, santos_benchmark_ori/, wikidbs/, valentine/
export DATASET_DIR=/path/to/your/datasets_dir/<dataset>_433   # specific output dataset directory
export VALENTINE_DIR=/path/to/valentine           # optional override for Valentine library path
```

Each script reads from `DATASET_ROOT` (for source data and Valentine) and writes to `DATASET_DIR` (the target dataset directory). If `DATASET_DIR` is not set, scripts fall back to their own directory — which works when running them from the original dataset location.

## Pipeline per Dataset

The steps are the same for all three datasets. Run sequentially within each dataset:

### Magellan

```bash
export DATASET_ROOT=/data/datasets
export DATASET_DIR=${DATASET_ROOT}/magellan_433

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
export DATASET_DIR=${DATASET_ROOT}/santos_benchmark_433

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
export DATASET_DIR=${DATASET_ROOT}/wikidbs_433

python prepare_splits/wikidbs/generate_datalake.py
python prepare_splits/wikidbs/label_entity_matching.py
python prepare_splits/wikidbs/label_unionable_table_search.py
python prepare_splits/wikidbs/label_joinable_from_gpt.py
python prepare_splits/wikidbs/extract_schema_matching_llm.py --resume
python prepare_splits/wikidbs/combine_schema_matching_llm.py --neg-per-pos 0
python prepare_splits/wikidbs/dataset_splits.py
```

Notes for WikiDBs:
- `extract_schema_matching_llm.py` calls an LLM API (DeepSeek by default). Set `DEEPSEEK_API_KEY` or place your key in `wikidbs_433/.deepseek_key`.
- `label_joinable_from_gpt.py` reuses pre-extracted GPT results at `$DATASET_ROOT/wikidbs/label/joinable_table_search/GPT_extracted_results.csv`.

## Optional: Re-split with 4/3/3 ratios

`create_parallel_433_from_1218.py` creates `_433` directories from existing `_1218` datasets using 0.4/0.3/0.3 train/validate/test ratios. It copies all files and re-runs `dataset_splits.py` in place:

```bash
DATASET_ROOT=/data/datasets python prepare_splits/create_parallel_433_from_1218.py
```

## Connecting to the Build Graph Pipeline

After splits are ready, run from the repo root:

```bash
DATASET_ROOT=/data/datasets bash run_no_token.sh --all
```

The output directories (`<dataset>_433_no_token/`) are used as `GRAPH_DIR` in the Uniprep training pipeline.
