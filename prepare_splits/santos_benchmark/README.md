Santos Benchmark Pipeline
=========================

This folder builds a Valentine-style `datalake_plus/` and four task labels from
`santos_benchmark_ori/`.

Unlike a LLM-based pipeline, this pipeline uses deterministic Valentine-inspired
transformations:

- **Derived tables** (horizontal split + noise + column obfuscation) provide
  supervision for:
  - `entity_matching` (row-level correspondences via `derived_map.csv`)
  - `schema_matching`
- **Unionable table search** uses the Santos ground truth
  (`santos_benchmark_ori/santos_small_benchmark_groundtruth.csv`), mapped onto
  the `datalake_plus/` **original** tables (derived/joinable are excluded).
- **Joinable tables** (vertical split + noise) provide supervision for:
  - `joinable_table_search`

Outputs
-------

- `datalake_plus/`: original + derived + joinable tables
- `metadata/`:
  - `table_registry.csv` (inventory + dataset + kind)
  - `derived_map.csv` (origin↔derived row/column mapping)
  - `joinable_map.csv` (origin↔joinable column mapping + join pairs)
- `label_plus/`:
  - `entity_matching/entity_matching_labels.csv`
  - `schema_matching/schema_matching_labels.csv`
  - `unionable_table_search/unionable_table_search_labels.csv` (GT query↔datalake positive pairs)
  - `joinable_table_search/joinable_table_search_labels.csv`
  - plus `train.csv`, `validate.csv`, `test.csv` for each task after splitting

Scripts
-------

1) Build datalake + metadata
```
python generate_datalake.py
```

2) Generate base labels
```
python label_entity_matching.py
python label_schema_matching.py
python label_unionable_table_search.py
python label_joinable_table_search.py
```

3) Generate train/validate/test splits
```
python dataset_splits.py
```

Notes
-----

- `dataset_splits.py` writes **dataset-disjoint** splits (a dataset's tables never appear in multiple splits).
- Schema/entity matching negatives are generated as "hard" negatives within in-domain table pairs.
