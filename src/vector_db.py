import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, VECTOR_STORE_DIR


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_client() -> chromadb.PersistentClient:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception as exc:
        message = str(exc).lower()
        if "not found" not in message and "does not exist" not in message:
            raise

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    collection = reset_collection()
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return len(chunks)


def query_chunks(question: str, top_k: int) -> list[dict]:
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=top_k)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "text": text,
                "metadata": metadata,
                "score": 1 - distance,
            }
        )

    return matches
