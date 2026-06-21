"""
diagnose_pipeline.py — точная диагностика этапов parse_pdfs -> build_index.

Запуск из корня проекта:
    python diagnose_pipeline.py

Показывает по каждому doc_id:
    1) сколько страниц с текстом извлёк parse_pdfs.py (из documents.jsonl)
    2) сколько символов всего
    3) сколько чанков из него реально попало в Chroma коллекцию
    4) пример первого чанка (чтобы увидеть содержательный ли там текст)

Это точно покажет, на каком именно этапе IMF/другие файлы теряются:
парсинг PDF -> JSONL, или JSONL -> Chroma.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CHROMA_COLLECTION, DATA_PROCESSED_DIR, VECTOR_STORE_DIR

DOCUMENTS_PATH = DATA_PROCESSED_DIR / "documents.jsonl"
CHROMA_DIR = VECTOR_STORE_DIR / "chroma"


def check_jsonl() -> dict:
    """Что реально лежит в documents.jsonl по каждому doc_id."""
    if not DOCUMENTS_PATH.exists():
        print(f"❌ {DOCUMENTS_PATH} не найден. Запусти scripts/parse_pdfs.py")
        return {}

    stats = defaultdict(lambda: {"pages": 0, "chars": 0, "sample": ""})
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id", "?")
            content = rec.get("content", "")
            stats[doc_id]["pages"] += 1
            stats[doc_id]["chars"] += len(content)
            if not stats[doc_id]["sample"] and content.strip():
                stats[doc_id]["sample"] = content[:150].replace("\n", " ")

    return dict(stats)


def check_chroma() -> dict:
    """Что реально лежит в Chroma коллекции по каждому doc_id / source_file."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        from src.config import EMBEDDING_MODEL
    except ImportError as e:
        print(f"❌ Не удалось импортировать chromadb: {e}")
        return {}

    if not CHROMA_DIR.exists():
        print(f"❌ {CHROMA_DIR} не найден. Запусти scripts/build_index.py")
        return {}

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=embedding_function)
    except Exception as e:
        print(f"❌ Не удалось открыть коллекцию '{CHROMA_COLLECTION}': {e}")
        return {}

    total = collection.count()
    print(f"Всего векторов в Chroma: {total}\n")

    # Достаём ВСЕ метаданные (без эмбеддингов) чтобы посчитать чанки по doc_id
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])

    stats = defaultdict(int)
    for md in metadatas:
        doc_id = md.get("doc_id", md.get("source_file", "?"))
        stats[doc_id] += 1

    return dict(stats)


def main():
    print("=" * 75)
    print("ЭТАП 1: documents.jsonl (выход parse_pdfs.py)")
    print("=" * 75)
    jsonl_stats = check_jsonl()

    if jsonl_stats:
        for doc_id in sorted(jsonl_stats.keys()):
            s = jsonl_stats[doc_id]
            flag = "  ⚠️ МАЛО ТЕКСТА" if s["chars"] < 2000 else ""
            print(f"{doc_id:<55} pages={s['pages']:>3}  chars={s['chars']:>8}{flag}")
            if s["sample"]:
                print(f"    └─ sample: {s['sample']!r}")
    print()

    print("=" * 75)
    print("ЭТАП 2: Chroma коллекция (выход build_index.py)")
    print("=" * 75)
    chroma_stats = check_chroma()

    if chroma_stats:
        for doc_id in sorted(chroma_stats.keys()):
            print(f"{doc_id:<55} chunks_in_chroma={chroma_stats[doc_id]:>5}")
    print()

    print("=" * 75)
    print("СРАВНЕНИЕ: doc_id есть в JSONL, но НЕТ (или мало) в Chroma")
    print("=" * 75)

    jsonl_ids = set(jsonl_stats.keys())
    chroma_ids = set(chroma_stats.keys())

    missing_in_chroma = jsonl_ids - chroma_ids
    if missing_in_chroma:
        print("❌ Есть в JSONL, но ПОЛНОСТЬЮ отсутствуют в Chroma:")
        for doc_id in sorted(missing_in_chroma):
            print(f"   - {doc_id}  (было {jsonl_stats[doc_id]['pages']} страниц, "
                  f"{jsonl_stats[doc_id]['chars']} символов)")
    else:
        print("✅ Все doc_id из JSONL присутствуют в Chroma (хотя бы 1 чанк)")

    print()
    print("Детальное сравнение количества чанков на файл:")
    for doc_id in sorted(jsonl_ids | chroma_ids):
        in_jsonl_chars = jsonl_stats.get(doc_id, {}).get("chars", 0)
        in_chroma = chroma_stats.get(doc_id, 0)
        status = "✅" if in_chroma > 0 else "❌"
        print(f"  {status} {doc_id:<55} jsonl_chars={in_jsonl_chars:>8}  chroma_chunks={in_chroma:>5}")


if __name__ == "__main__":
    main()
