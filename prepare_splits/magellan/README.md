Magellan_1218 Pipeline
======================

This folder builds a Valentine-style `datalake_plus/` and four task labels from
`magellan_ori/` (the classic Magellan entity matching benchmarks).

Compared with `magellan/`, this pipeline **does not use LLMs** to create schema
alternatives.  Instead, it relies on deterministic transformations inspired by
the Valentine generator:

- **Derived tables** (horizontal split + noise + column obfuscation) provide
  supervision for:
  - `schema_matching`
  - `unionable_table_search`
- **Joinable tables** (vertical split + noise) provide supervision for:
  - `joinable_table_search`
- **Entity matching** labels come from the original Magellan ground truth
  (`magellan_ori/<dataset>/{train,valid,test}.csv`), mapped to the generated
  `datalake_plus/` table names.

To increase the number of tables (and match the behavior of `magellan/`), each
source `tableA/tableB` is also chunked into multiple **original** tables:

- Row chunks (`CHUNK_SIZE_ROW=100`, automatically adjusted so each table yields
  ~5 chunks, with the last chunk merged if the remainder is small)
- Two column-size variants per chunk (full columns, and full-1 non-id columns)

Entity-matching labels are generated **only for the max-colsize (full-column)**
chunk variants (consistent with `magellan/dataset_construction.py`).

**Important:** for Magellan, `l_id/r_id` in `label_plus/entity_matching/*.csv`
are now **chunk-local row indices** (0..N-1) so they can be read directly from
`datalake_plus/<table>.csv` via row index.

If you need the original/global id from `magellan_ori`, recover it as:
`global_id = chunk_start_id + local_row`.

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
  - `unionable_table_search/unionable_table_search_labels.csv`
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

- `dataset_splits.py` now defaults to **random** split mode (tables can appear across train/validate/test).
  - Use `--split-mode dataset-disjoint` if you want the old dataset-isolated behavior.
- Schema-matching negatives are generated as "hard" negatives within in-domain table pairs.
