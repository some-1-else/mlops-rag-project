import json
import logging
from pathlib import Path

from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "documents.jsonl"
PDF_MAGIC = b"%PDF-"
MIN_AVG_CHARS_PER_PAGE = 200

logging.getLogger("pypdf").setLevel(logging.ERROR)


def has_pdf_magic(pdf_path: Path) -> bool:
    with pdf_path.open("rb") as file:
        return file.read(len(PDF_MAGIC)) == PDF_MAGIC


def parse_pdf(pdf_path: Path) -> tuple[list[dict], dict]:
    reader = PdfReader(str(pdf_path))
    doc_id = pdf_path.stem
    rows = []
    page_count = len(reader.pages)
    page_errors = []
    total_chars = 0

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            content = (page.extract_text() or "").strip()
        except Exception as exc:
            page_errors.append({"page": page_number, "error": str(exc)})
            continue

        if not content:
            continue

        total_chars += len(content)
        rows.append(
            {
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "page": page_number,
                "type": "text",
                "content": content,
            }
        )

    avg_chars_per_page = total_chars / page_count if page_count else 0
    diagnostics = {
        "file": pdf_path.name,
        "page_count": page_count,
        "non_empty_pages": len(rows),
        "avg_chars_per_page": round(avg_chars_per_page, 1),
        "page_errors": page_errors,
        "low_text": page_count > 0 and avg_chars_per_page < MIN_AVG_CHARS_PER_PAGE,
    }
    return rows, diagnostics


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found PDF files: {len(pdf_files)}")

    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR}")
        return

    total_text_pages = 0
    failed_files = []
    zero_text_files = []
    low_text_files = []
    page_error_files = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for pdf_path in pdf_files:
            try:
                if not has_pdf_magic(pdf_path):
                    failed_files.append(
                        {
                            "file": pdf_path.name,
                            "error": "missing %PDF- magic bytes",
                        }
                    )
                    print(f"Skipped {pdf_path.name}: missing %PDF- magic bytes")
                    continue

                rows, diagnostics = parse_pdf(pdf_path)
            except Exception as exc:
                failed_files.append({"file": pdf_path.name, "error": str(exc)})
                print(f"Failed {pdf_path.name}: {exc}")
                continue

            total_text_pages += len(rows)
            print(
                "Processed "
                f"{pdf_path.name}: {len(rows)} non-empty pages "
                f"of {diagnostics['page_count']} total, "
                f"avg chars/page={diagnostics['avg_chars_per_page']}"
            )

            if diagnostics["non_empty_pages"] == 0:
                zero_text_files.append(diagnostics)
            if diagnostics["low_text"]:
                low_text_files.append(diagnostics)
                print(
                    "Warning: suspiciously little extracted text in "
                    f"{pdf_path.name}; it may be scanned, OCR-free, or an error page."
                )
            if diagnostics["page_errors"]:
                page_error_files.append(diagnostics)

            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("")
    print("Parsing summary")
    print(f"Total text pages: {total_text_pages}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Files with zero extracted text pages: {len(zero_text_files)}")
    print(f"Files with low average text: {len(low_text_files)}")
    print(f"Files with page-level extraction errors: {len(page_error_files)}")

    if failed_files:
        print("Failed/skipped files:")
        for item in failed_files:
            print(f"  - {item['file']}: {item['error']}")

    if zero_text_files:
        print("Zero-text files:")
        for item in zero_text_files:
            print(f"  - {item['file']}: pages={item['page_count']}")

    if low_text_files:
        print("Low-text files:")
        for item in low_text_files:
            print(
                f"  - {item['file']}: pages={item['page_count']}, "
                f"non_empty={item['non_empty_pages']}, "
                f"avg_chars/page={item['avg_chars_per_page']}"
            )

    if page_error_files:
        print("Files with page errors:")
        for item in page_error_files:
            pages = ", ".join(str(error["page"]) for error in item["page_errors"][:10])
            print(f"  - {item['file']}: pages {pages}")

    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
