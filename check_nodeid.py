import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
import ast

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


def load_node_mapping(dir_path: Path):
    mapping_file = dir_path / "node_id_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"node_id_mapping.json not found at: {mapping_file}")
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    return mapping


def build_id_to_layer_and_key(mapping: dict):
    """Build reverse lookup: node_id -> (layer, original_key)

    original_key is the deserialized key. For tuple-like strings, we parse via ast.literal_eval.
    """
    id_to_layer = {}
    id_to_key = {}
    all_ids = []

    total = sum(len(m) for m in mapping.values())
    batch = 0
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, desc="Reversing node mapping", unit="node", mininterval=1.0)

    for layer, m in mapping.items():
        for k_str, nid in m.items():
            # JSON dumped keys as strings; try to parse tuple/dict literals safely
            try:
                k = ast.literal_eval(k_str)
            except Exception:
                k = k_str
            if nid in id_to_layer and id_to_layer[nid] != layer:
                # Duplicate id across layers shouldn't happen; keep first and record
                pass
            id_to_layer[nid] = layer
            id_to_key[nid] = k
            all_ids.append(nid)

            batch += 1
            if pbar is not None and batch >= 100000:
                pbar.update(batch)
                batch = 0

    if pbar is not None:
        if batch:
            pbar.update(batch)
        pbar.close()

    return id_to_layer, id_to_key, all_ids


def check_node_ids(all_ids):
    problems = []
    id_set = set(all_ids)
    if not all_ids:
        return problems
    min_id, max_id = min(all_ids), max(all_ids)
    if min_id != 0:
        problems.append(f"Min node id is {min_id}, expected 0")
    if len(id_set) != len(all_ids):
        problems.append(f"Found duplicate node ids: total={len(all_ids)}, unique={len(id_set)})")
    expected_count = max_id + 1
    if len(id_set) != expected_count:
        # Show a small sample of missing ids to diagnose holes
        missing = []
        if expected_count - len(id_set) <= 10000:  # avoid expensive when gigantic
            missing = [i for i in range(0, max_id + 1) if i not in id_set][:20]
        problems.append(
            f"Node ids not contiguous: unique={len(id_set)}, expected={expected_count}; sample_missing={missing}"
        )
    return problems


def stream_files(dir_path: Path):
    files = {
        "edges": dir_path / "edge_lists.txt",
        "types": dir_path / "edge_type.txt",
        "labels": dir_path / "edge_labels.txt",
        "train": dir_path / "train_mask.txt",
        "val": dir_path / "validate_mask.txt",
        "test": dir_path / "test_mask.txt",
    }
    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    return (
        open(files["edges"], "r"),
        open(files["types"], "r"),
        open(files["labels"], "r"),
        open(files["train"], "r"),
        open(files["val"], "r"),
        open(files["test"], "r"),
    )


def layer_pair_ok(etype: str, u_layer: str, v_layer: str) -> bool:
    # Hierarchical edges
    if etype == "token_cell":
        return u_layer == "token" and v_layer == "cell"
    if etype == "cell_row":
        return u_layer == "cell" and v_layer == "row"
    if etype == "cell_column":
        return u_layer == "cell" and v_layer == "column"
    if etype == "row_table":
        return u_layer == "row" and v_layer == "table"
    if etype == "column_table":
        return u_layer == "column" and v_layer == "table"

    # Task edges
    if etype == "entity_matching":
        return u_layer == "row" and v_layer == "row"
    if etype == "schema_matching":
        return u_layer == "column" and v_layer == "column"
    if etype == "union_table_search":
        return u_layer == "table" and v_layer == "table"
    if etype == "joinable_table_search":
        return u_layer == "column" and v_layer == "column"

    return True  # Unknown type: don't block, but will be counted elsewhere


def run_checks(dir_path: Path, max_report: int = 50):
    print(f"Checking graph files under: {dir_path}")

    mapping = load_node_mapping(dir_path)
    print("Loaded node_id_mapping.json, building reverse index...")
    id_to_layer, id_to_key, all_ids = build_id_to_layer_and_key(mapping)

    # 1) Node id sanity
    nid_issues = check_node_ids(all_ids)
    if nid_issues:
        print("[NodeID] Issues detected:")
        for s in nid_issues:
            print(" -", s)
    else:
        print("[NodeID] OK: contiguous from 0..N-1 and unique across layers")

    # 2) Cell text reuse overview
    cell_map = mapping.get("cell", {})
    id_to_texts = defaultdict(list)
    for text, nid in cell_map.items():
        id_to_texts[nid].append(text)
    reused = [(nid, len(txts)) for nid, txts in id_to_texts.items() if len(txts) > 1]
    if reused:
        print(f"[Cell] {len(reused)} node_ids map to >1 distinct cell texts (show up to 5 examples):")
        for nid, cnt in reused[:5]:
            print(f" - node {nid} maps to {cnt} cell texts; sample: {id_to_texts[nid][:3]}")
    else:
        print("[Cell] No node has multiple distinct cell texts")

    # 3) Edge-level checks (streaming)
    edges_f, types_f, labels_f, train_f, val_f, test_f = stream_files(dir_path)
    total = 0
    counts = Counter()
    problems = []
    id_missing = 0
    layer_mismatch = 0
    bad_line_format = 0
    bad_joinable_format = 0
    mask_mismatch = 0
    type_mismatch = 0
    label_out_of_range = 0

    batch = 0
    pbar = None
    if tqdm is not None:
        # Use dynamic total to avoid a full pass for counting
        pbar = tqdm(desc="Scanning edges", unit="edge", mininterval=1.0)

    try:
        while True:
            e_line = edges_f.readline()
            if not e_line:
                break
            t_line = types_f.readline()
            l_line = labels_f.readline()
            tr_line = train_f.readline()
            va_line = val_f.readline()
            te_line = test_f.readline()

            if not (t_line and l_line and tr_line and va_line and te_line):
                problems.append(f"Edge metadata length mismatch at edge #{total}")
                break

            e_parts = e_line.strip().split("\t")
            t = t_line.strip()
            try:
                label = int(l_line.strip())
            except ValueError:
                label_out_of_range += 1
                label = -1
            try:
                mtrain = int(tr_line.strip())
                mval = int(va_line.strip())
                mtest = int(te_line.strip())
            except ValueError:
                mask_mismatch += 1
                mtrain = mval = mtest = 0

            if len(e_parts) not in (2, 4):
                bad_line_format += 1
                if len(problems) < max_report:
                    problems.append(f"Bad edge line format at #{total}: parts={len(e_parts)} -> {e_parts[:6]}")
                total += 1
                continue

            # Parse endpoints (and optional table ids)
            try:
                u = int(e_parts[0])
                v = int(e_parts[1])
                extra = tuple(map(int, e_parts[2:])) if len(e_parts) == 4 else ()
            except ValueError:
                bad_line_format += 1
                if len(problems) < max_report:
                    problems.append(f"Non-integer edge ids at #{total}: {e_parts}")
                total += 1
                continue

            # Basic existence check
            missing_end = []
            for nid in (u, v) + extra:
                if nid not in id_to_layer:
                    missing_end.append(nid)
            if missing_end:
                id_missing += 1
                if len(problems) < max_report:
                    problems.append(f"Unknown node id(s) at edge #{total}: {missing_end}, type={t}")

            # Type-specific validation
            u_layer = id_to_layer.get(u, "?")
            v_layer = id_to_layer.get(v, "?")

            # Extra ids are only valid for joinable_table_search
            if len(extra) == 2:
                if t != "joinable_table_search":
                    type_mismatch += 1
                    if len(problems) < max_report:
                        problems.append(
                            f"Edge #{total}: has 4 ids but type is '{t}', expected 'joinable_table_search'"
                        )
                # Check extra ids are table ids
                t1, t2 = extra
                if id_to_layer.get(t1) != "table" or id_to_layer.get(t2) != "table":
                    bad_joinable_format += 1
                    if len(problems) < max_report:
                        problems.append(
                            f"Edge #{total}: joinable extras not table ids: {extra}"
                        )
                # Optional: verify column's table matches extras
                # Recover table name from column id and compare
                col1_key = id_to_key.get(u)
                col2_key = id_to_key.get(v)
                if isinstance(col1_key, tuple) and len(col1_key) == 2:
                    table_name_1 = col1_key[0]
                    table_id_1 = mapping.get("table", {}).get(str(table_name_1))
                    if table_id_1 is not None and table_id_1 != t1 and len(problems) < max_report:
                        problems.append(
                            f"Edge #{total}: column->table mismatch: col {u} from '{table_name_1}' -> {table_id_1}, extras has {t1}"
                        )
                if isinstance(col2_key, tuple) and len(col2_key) == 2:
                    table_name_2 = col2_key[0]
                    table_id_2 = mapping.get("table", {}).get(str(table_name_2))
                    if table_id_2 is not None and table_id_2 != t2 and len(problems) < max_report:
                        problems.append(
                            f"Edge #{total}: column->table mismatch: col {v} from '{table_name_2}' -> {table_id_2}, extras has {t2}"
                        )

            elif len(extra) == 0 and t == "joinable_table_search":
                bad_joinable_format += 1
                if len(problems) < max_report:
                    problems.append(f"Edge #{total}: joinable_table_search should have 4 ids, found 2")

            # Layer compatibility for all types
            if not layer_pair_ok(t, u_layer, v_layer):
                layer_mismatch += 1
                if len(problems) < max_report:
                    problems.append(
                        f"Edge #{total}: layer mismatch for type '{t}': {u}({u_layer}) -> {v}({v_layer})"
                    )

            # Masks consistency: exactly one of train/val/test is 1
            msum = mtrain + mval + mtest
            if msum != 1:
                mask_mismatch += 1
                if len(problems) < max_report:
                    problems.append(
                        f"Edge #{total}: invalid mask triplet (train,val,test)=({mtrain},{mval},{mtest})"
                    )

            # Label basic check
            if label not in (0, 1):
                label_out_of_range += 1
                if len(problems) < max_report:
                    problems.append(f"Edge #{total}: label not in {0,1}: {label}")

            counts[t] += 1
            total += 1
            batch += 1
            if pbar is not None and batch >= 100000:
                pbar.update(batch)
                batch = 0
    finally:
        edges_f.close(); types_f.close(); labels_f.close()
        train_f.close(); val_f.close(); test_f.close()
        if pbar is not None:
            if batch:
                pbar.update(batch)
            pbar.close()

    print("[Edges] Total:", total)
    print("[Edges] By type:")
    for et, c in counts.most_common():
        print(f" - {et}: {c}")

    if problems:
        print(f"[Edges] Found {len(problems)} issues (showing up to {max_report}):")
        for msg in problems[:max_report]:
            print(" -", msg)
    else:
        print("[Edges] OK: all basic checks passed")

    print("[Summary] Counters:")
    print(f" - missing_node_ids: {id_missing}")
    print(f" - layer_mismatch: {layer_mismatch}")
    print(f" - bad_line_format: {bad_line_format}")
    print(f" - bad_joinable_format: {bad_joinable_format}")
    print(f" - mask_mismatch: {mask_mismatch}")
    print(f" - type_mismatch: {type_mismatch}")
    print(f" - label_out_of_range: {label_out_of_range}")


def main():
    parser = argparse.ArgumentParser(description="Check node IDs and edges for consistency")
    parser.add_argument(
        "--dir",
        dest="dir",
        default=None,
        help="Directory containing graph files (edge_lists.txt, node_id_mapping.json, etc.)",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=50,
        help="Max number of detailed issues to print",
    )
    args = parser.parse_args()

    # Resolve default directory if not provided
    if args.dir is None:
        # Prefer wikidbs if present; otherwise wikidbs_no_token
        here = Path(__file__).parent
        candidates = [here / "wikidbs", here / "wikidbs_no_token", Path("data/wikidbs"), Path("data/wikidbs_no_token")]
        for c in candidates:
            if c.exists():
                args.dir = str(c)
                break
    dir_path = Path(args.dir).expanduser().resolve() if args.dir else None
    if not dir_path or not dir_path.exists():
        raise FileNotFoundError(f"Graph directory not found. Provide with --dir. Tried: {args.dir}")

    run_checks(dir_path, max_report=args.max_report)


if __name__ == "__main__":
    main()
