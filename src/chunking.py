from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == text_length:
            break

    return chunks


def chunk_documents(pages: list[dict]) -> list[dict]:
    chunks = []

    for page in pages:
        for chunk_idx, chunk in enumerate(chunk_text(page["text"]), start=1):
            chunks.append(
                {
                    "id": f"{page['source']}:p{page['page']}:c{chunk_idx}",
                    "text": chunk,
                    "metadata": {
                        "source": page["source"],
                        "path": page["path"],
                        "page": page["page"],
                        "chunk": chunk_idx,
                    },
                }
            )

    return chunks
