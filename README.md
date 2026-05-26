# Uniprep Dataset Scripts

Dataset construction scripts used by [Uniprep](https://github.com/mengshichen-18/uniprep).

The original dataset files are hosted on Google Drive:

https://drive.google.com/drive/folders/15TtuTNZqpwKOI-xWe-VjUYkNQbCRWBAk?usp=drive_link

Download the original data, then follow the steps below to build graph-ready inputs.

## Step 1: Prepare dataset splits

Build `datalake_plus/`, `label_plus/`, and `metadata/` from the original data, then generate train/validate/test splits. See [`prepare_splits/README.md`](prepare_splits/README.md) for full instructions.

```bash
export DATASET_ROOT=/path/to/your/datasets_dir
export DATASET_DIR=${DATASET_ROOT}/magellan_433   # repeat for each dataset

python prepare_splits/magellan/generate_datalake.py
python prepare_splits/magellan/label_entity_matching.py
# ... (see prepare_splits/README.md for complete command list)
python prepare_splits/magellan/dataset_splits.py
```

## Step 2: Build graphs

Run the no-token graph pipeline for each dataset. Outputs go to `$DATASET_ROOT/<dataset>_433_no_token/`.

```bash
DATASET_ROOT=/path/to/your/datasets_dir bash run_no_token.sh --all
```

Required environment variables:

| Variable | Description |
|---|---|
| `DATASET_ROOT` | Directory containing `<dataset>_433` subdirectories |
| `MODEL_PATH` | HuggingFace model path for embedding (e.g. `sentence-t5-base`) |

The output directories (`<dataset>_433_no_token/`) are passed as `GRAPH_DIR` when training with Uniprep.
