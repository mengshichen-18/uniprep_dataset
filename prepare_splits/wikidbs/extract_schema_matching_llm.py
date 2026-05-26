import argparse
import base64
import csv
import json
import mimetypes
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(os.environ.get("DATASET_ROOT", str(Path(__file__).resolve().parents[2])))
LEGACY_BASE = REPO_ROOT / "wikidbs"
SELECTED_LIST_PATH = LEGACY_BASE / "selected_folder_wikidbs.csv"
SELECTED_DATASETS_DIR = LEGACY_BASE / "selected_dataset"

OUTPUT_BASE = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent)))
IMAGE_DIR = OUTPUT_BASE / "extracted_images"
OUTPUT_CSV = OUTPUT_BASE / "label_plus" / "schema_matching" / "GPT_extracted_schema_matching_results.csv"


def load_pymupdf():
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.23
        return fitz
    except Exception:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyMuPDF is required. Install with `pip install pymupdf`."
            ) from exc
        if not hasattr(fitz, "open") or not hasattr(fitz, "Matrix"):
            raise RuntimeError(
                "Imported 'fitz' is not PyMuPDF. Uninstall the 'fitz' package and install PyMuPDF."
            )
        return fitz


def render_pages_as_images(pdf_path: Path, output_folder: Path, zoom: float) -> list[Path]:
    fitz = load_pymupdf()
    doc = fitz.open(pdf_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    img_paths: list[Path] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_path = output_folder / f"page_{page_num + 1}.png"
        pix.save(str(img_path))
        img_paths.append(img_path)
    return img_paths


def encode_image_base64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode("utf-8")


def analyze_images_with_openai(
    client: OpenAI,
    image_paths: list[Path],
    *,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    for image_path in image_paths:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        base64_image = encode_image_base64(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": "high",
                },
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


DEFAULT_PROMPT = (
    "You are labeling a schema matching dataset.\n"
    "Given the database schema (diagram or text), extract semantically equivalent COLUMN pairs across DIFFERENT tables.\n"
    "A match means the two columns represent the same attribute/identifier concept (synonyms/abbreviations), "
    "NOT merely that they are joinable via a foreign key.\n"
    "Return ONLY a Python list of 4-tuples/lists in the exact format:\n"
    "  [[table_name_1, column_name_1, table_name_2, column_name_2], ...]\n"
    "Use table and column names exactly as shown in the schema.\n"
    "Only output columns that are explicitly listed; do NOT guess missing columns.\n"
    "Only include high-confidence matches.\n"
)


def analyze_schema_text_with_openai(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def build_schema_text_from_info_full(dataset_dir: Path) -> str | None:
    schema_path = dataset_dir / "info_full.json"
    if not schema_path.exists():
        return None
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    tables = payload.get("TABLES", {})
    if not tables:
        return None

    lines: list[str] = []
    info = payload.get("INFO", {})
    db_name = info.get("db_folder_name", dataset_dir.name)
    lines.append(f"Dataset: {db_name}")
    for table_name, table_info in tables.items():
        filepath = table_info.get("FILEPATH", "")
        cols = table_info.get("COLUMNS", []) or []
        lines.append(f"Table: {table_name} (file: {filepath})")
        if cols:
            lines.append("Columns: " + ", ".join(cols))
        fks = table_info.get("FOREIGN_KEYS", []) or []
        if fks:
            lines.append("Foreign keys:")
            for fk in fks:
                if not isinstance(fk, dict):
                    continue
                fk_col = fk.get("FOREIGN_KEY", ["", ""])[1]
                ref = fk.get("REFERENCE_TABLE", "")
                lines.append(f"- {fk_col} -> {ref}")
    return "\n".join(lines)


def build_text_prompt(schema_text: str) -> str:
    return f"{DEFAULT_PROMPT}\nSchema:\n{schema_text}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract schema matching pairs from Wikidbs diagrams using a vision-capable LLM."
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--schema-source",
        choices=["info_full", "image"],
        default="info_full",
        help="Use info_full.json (text) or schema images.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY).",
    )
    parser.add_argument(
        "--api-key-file",
        default=".deepseek_key",
        help="Path to a file containing the API key (default: .deepseek_key (in the dataset directory)).",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com/v1",
        help="DeepSeek base URL, e.g. https://api.deepseek.com/v1 (or set DEEPSEEK_BASE_URL).",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    key_file_path = Path(args.api_key_file)
    if not key_file_path.is_absolute():
        key_file_path = OUTPUT_BASE / key_file_path
    key_file_value = None
    if key_file_path.exists():
        key_file_value = key_file_path.read_text(encoding="utf-8").strip() or None

    api_key = args.api_key or key_file_value or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL")
    if not api_key:
        raise ValueError(
            "Missing API key. Provide --api-key, set DEEPSEEK_API_KEY, "
            "or put the key in .deepseek_key (in the dataset directory)."
        )
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    datasets_df = pd.read_csv(SELECTED_LIST_PATH)
    dataset_names = datasets_df["filename"].tolist()
    if args.limit:
        dataset_names = dataset_names[: args.limit]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    done: set[str] = set()
    if args.resume and OUTPUT_CSV.exists():
        try:
            done_df = pd.read_csv(OUTPUT_CSV)
            if "dataset_name" in done_df.columns:
                done = set(done_df["dataset_name"].dropna().astype(str).tolist())
        except Exception:
            done = set()

    rows: list[dict[str, str]] = []
    if OUTPUT_CSV.exists():
        try:
            rows = pd.read_csv(OUTPUT_CSV).to_dict(orient="records")
        except Exception:
            rows = []

    processed = 0
    for dataset_name in dataset_names:
        if dataset_name in done:
            continue

        dataset_dir = SELECTED_DATASETS_DIR / dataset_name
        if args.schema_source == "image":
            pdf_path = dataset_dir / "schema_diagram.pdf"
            if not pdf_path.exists():
                continue
            image_folder = IMAGE_DIR / dataset_name
            if not image_folder.exists() or not any(image_folder.glob("page_*.png")):
                render_pages_as_images(pdf_path, image_folder, zoom=args.zoom)
            image_paths = sorted(image_folder.glob("page_*.png"))
            if not image_paths:
                continue
            result = analyze_images_with_openai(
                client,
                image_paths,
                model=args.model,
                prompt=DEFAULT_PROMPT,
                temperature=args.temperature,
            )
        else:
            schema_text = build_schema_text_from_info_full(dataset_dir)
            if not schema_text:
                continue
            prompt = build_text_prompt(schema_text)
            result = analyze_schema_text_with_openai(
                client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
            )

        rows.append({"dataset_name": dataset_name, "analysis": result})
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_name", "analysis"])
            writer.writeheader()
            writer.writerows(rows)

        processed += 1
        print(f"[{processed}] wrote result for {dataset_name} -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
