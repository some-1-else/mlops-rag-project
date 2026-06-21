"""
test_retrieval.py — проверяет, какая embedding-модель используется и
насколько хорошо русский запрос находит английские чанки IMF.

Запуск:
    python test_retrieval.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

print(f"EMBEDDING_MODEL = {EMBEDDING_MODEL}")
print(f"CHUNK_SIZE = {CHUNK_SIZE}")
print(f"CHUNK_OVERLAP = {CHUNK_OVERLAP}")
print()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DIR = VECTOR_STORE_DIR / "chroma"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=embedding_function)

# Тестовые запросы — русский и английский варианты одного и того же вопроса
test_queries = [
    "какой был рост мировой экономики",
    "world economic growth rate",
    "прогноз роста ВВП в мире",
    "global GDP growth forecast",
]

for q in test_queries:
    print("=" * 80)
    print(f"ЗАПРОС: {q!r}")
    print("=" * 80)
    result = collection.query(query_texts=[q], n_results=10)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    # Считаем сколько чанков из каждого doc_id попало в top-10
    from collections import Counter
    doc_id_counts = Counter(m.get("doc_id", "?") for m in metas)

    print(f"Top-10 по источникам: {dict(doc_id_counts)}")
    print()
    for i, (doc, meta, dist) in enumerate(zip(docs[:5], metas[:5], dists[:5]), 1):
        score = 1 - dist
        preview = doc[:120].replace("\n", " ")
        print(f"  [{i}] score={score:.3f}  doc_id={meta.get('doc_id')}  page={meta.get('page')}")
        print(f"      {preview!r}")
    print()
