import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunking import chunk_documents
from src.config import DATA_RAW_DIR
from src.pdf_loader import load_pdfs
from src.vector_db import add_chunks


def main() -> None:
    pages = load_pdfs(DATA_RAW_DIR)
    if not pages:
        print(f"No PDFs with extractable text found in {DATA_RAW_DIR}")
        return

    chunks = chunk_documents(pages)
    indexed_count = add_chunks(chunks)
    print(f"Indexed {indexed_count} chunks from {len(pages)} pages.")


if __name__ == "__main__":
    main()
