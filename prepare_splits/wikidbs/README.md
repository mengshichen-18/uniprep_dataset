Wikidbs_1218 Pipeline
=====================

This folder builds a new datalake + labels from `wikidbs/selected_dataset`.
It keeps the Valentine-style string obfuscation for derived tables (used by
entity matching and unionable search), while using LLM-based extraction for
schema matching and GPT-extracted results for joinable search.

Outputs
-------

- `datalake_plus/`: original + derived tables (2x original table count)
- `label_plus/`:
  - `entity_matching/entity_matching_labels.csv`
  - `unionable_table_search/unionable_table_search_labels.csv`
  - `joinable_table_search/joinable_table_search_labels.csv`
  - `schema_matching/schema_matching_labels.csv`
- `metadata/`:
  - `table_registry.csv` (table inventory + dataset)
  - `derived_map.csv` (origin/derived mapping + row map + column map)

Scripts
-------

1) Build the datalake (original + derived tables)
```
python generate_datalake.py
```

2) Entity matching labels (positives from original↔derived row mapping)
```
python label_entity_matching.py
```

3) Unionable table search labels (original + derived share unionable_id)
```
python label_unionable_table_search.py
```

4) Joinable table search labels (map existing GPT results to datalake)
```
python label_joinable_from_gpt.py
```

5) Schema matching LLM extraction (text mode by default)
```
python extract_schema_matching_llm.py --resume
python combine_schema_matching_llm.py --neg-per-pos 0
```

6) Generate train/validate/test splits (dataset-disjoint, hard negatives)
```
python dataset_splits.py
```

Notes
-----

- LLM scripts read schema from `wikidbs/selected_dataset/<db>/info_full.json` by default.
- For image mode: `python extract_schema_matching_llm.py --schema-source image` (requires PyMuPDF).
- Running `extract_schema_matching_llm.py` always overwrites `GPT_extracted_schema_matching_results.csv`.
- Put your key in `wikidbs_1218/.deepseek_key` or set `DEEPSEEK_API_KEY`; set `DEEPSEEK_BASE_URL` if needed.
- If you see a `fitz` import error, uninstall the `fitz` package and install PyMuPDF (`pip install pymupdf`).
- Joinable labels reuse `wikidbs/label/joinable_table_search/GPT_extracted_results.csv`.
- This pipeline only writes base labels; split generation can be added later if needed.
