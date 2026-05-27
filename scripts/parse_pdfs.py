import json
from pathlib import Path

from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "documents.jsonl"


def parse_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    doc_id = pdf_path.stem
    rows = []

    for page_number, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()
        if not content:
            continue

        rows.append(
            {
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "page": page_number,
                "type": "text",
                "content": content,
            }
        )

    return rows


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found PDF files: {len(pdf_files)}")

    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR}")
        return

    total_pages = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for pdf_path in pdf_files:
            rows = parse_pdf(pdf_path)
            total_pages += len(rows)
            print(f"Processed {pdf_path.name}: {len(rows)} non-empty pages")

            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total processed pages: {total_pages}")
    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
