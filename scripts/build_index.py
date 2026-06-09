import json
import shutil
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunking import chunk_text
from src.config import CHROMA_COLLECTION, DATA_PROCESSED_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR


DOCUMENTS_PATH = DATA_PROCESSED_DIR / "documents.jsonl"
CHROMA_DIR = VECTOR_STORE_DIR / "chroma"


def read_text_records(jsonl_path: Path) -> tuple[list[dict], int]:
    records = []
    total_records = 0

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            total_records += 1
            record = json.loads(line)
            content = (record.get("content") or "").strip()

            if record.get("type") != "text" or not content:
                continue

            records.append(
                {
                    "doc_id": record["doc_id"],
                    "source_file": record["source_file"],
                    "page": record["page"],
                    "type": record["type"],
                    "content": content,
                }
            )

    return records, total_records


def build_chunks(records: list[dict]) -> list[dict]:
    chunks = []

    for record in records:
        for chunk_id, chunk in enumerate(chunk_text(record["content"]), start=1):
            chunks.append(
                {
                    "id": f"{record['doc_id']}:p{record['page']}:c{chunk_id}",
                    "document": chunk,
                    "metadata": {
                        "doc_id": record["doc_id"],
                        "source_file": record["source_file"],
                        "page": record["page"],
                        "chunk_id": chunk_id,
                        "type": record["type"],
                    },
                }
            )

    return chunks


def rebuild_index(chunks: list[dict]) -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        return

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["document"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def main() -> None:
    if not DOCUMENTS_PATH.exists():
        print(f"Input file not found: {DOCUMENTS_PATH}")
        print("Run python scripts/parse_pdfs.py first.")
        return

    records, total_records = read_text_records(DOCUMENTS_PATH)
    chunks = build_chunks(records)
    rebuild_index(chunks)

    print(f"Read JSONL records: {total_records}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved Chroma index to: {CHROMA_DIR}")

    if chunks:
        print(f"Example metadata: {chunks[0]['metadata']}")


if __name__ == "__main__":
    main()
