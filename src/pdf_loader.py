from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if not text:
            continue

        pages.append(
            {
                "source": pdf_path.name,
                "path": str(pdf_path),
                "page": page_idx,
                "text": text,
            }
        )

    return pages


def load_pdfs(raw_dir: Path) -> list[dict]:
    documents = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        documents.extend(load_pdf_pages(pdf_path))
    return documents
