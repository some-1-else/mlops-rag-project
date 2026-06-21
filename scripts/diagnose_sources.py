"""
diagnose_sources.py — диагностика: какие PDF реально скачались и распарсились нормально.

Запуск из корня проекта:
    python diagnose_sources.py

Проверяет:
1. Какие файлы из data/raw/ существуют и какого они размера
2. Является ли каждый файл валидным PDF (или это HTML-страница ошибки)
3. Сколько текста реально извлеклось из каждого файла
4. Какие doc_id попали в documents.jsonl и сколько чанков на каждый
"""

import json
from pathlib import Path

ROOT_DIR = Path("/home/matthew_linux/test/HSE/test_ii/mlops-rag-project")
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"

# Признаки того, что файл — это не настоящий PDF, а страница ошибки/редиректа
SUSPICIOUS_MARKERS = [
    "access denied", "403 forbidden", "404 not found", "not found",
    "error", "doctype html", "<html", "captcha", "blocked",
    "page not found", "service unavailable",
]


def check_pdf_validity(path: Path) -> dict:
    """Проверяет, является ли файл настоящим PDF с полезным текстом."""
    result = {
        "file": path.name,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "exists": path.exists(),
        "is_real_pdf": False,
        "pages": 0,
        "total_chars": 0,
        "suspicious": False,
        "error": None,
    }

    if not path.exists():
        result["error"] = "FILE NOT FOUND"
        return result

    # Проверка магических байт PDF
    with path.open("rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        result["error"] = f"NOT A PDF FILE (header: {header!r}) — likely HTML error page"
        result["suspicious"] = True
        return result

    try:
        import fitz
        doc = fitz.open(str(path))
        result["pages"] = len(doc)
        total_text = ""
        for page in doc:
            total_text += page.get_text()
        result["total_chars"] = len(total_text.strip())
        result["is_real_pdf"] = True

        lowered = total_text.lower()[:2000]
        if any(marker in lowered for marker in SUSPICIOUS_MARKERS) and result["total_chars"] < 500:
            result["suspicious"] = True
            result["error"] = "Looks like an error page disguised as PDF (short + suspicious text)"

    except Exception as e:
        result["error"] = f"PyMuPDF failed to open: {e}"
        result["suspicious"] = True

    return result


def check_documents_jsonl() -> dict:
    """Считает, сколько записей/символов попало в documents.jsonl по каждому doc_id."""
    if not DOCUMENTS_PATH.exists():
        return {"error": f"{DOCUMENTS_PATH} not found — run scripts/build_index.py or scripts/parse_pdfs.py first"}

    stats: dict[str, dict] = {}
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id", "?")
            if doc_id not in stats:
                stats[doc_id] = {"pages": 0, "total_chars": 0}
            stats[doc_id]["pages"] += 1
            stats[doc_id]["total_chars"] += len(rec.get("content", ""))

    return stats


def main():
    print("=" * 70)
    print("1. ПРОВЕРКА ФАЙЛОВ В data/raw/")
    print("=" * 70)

    if not RAW_DIR.exists():
        print(f"❌ Папка {RAW_DIR} не существует")
        return

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Найдено PDF файлов: {len(pdf_files)}\n")

    bad_files = []
    for pdf_path in pdf_files:
        info = check_pdf_validity(pdf_path)
        status = "✅" if info["is_real_pdf"] and not info["suspicious"] else "❌"
        print(f"{status} {info['file']:<60} "
              f"size={info['size_bytes']:>8}b  "
              f"pages={info['pages']:>3}  "
              f"chars={info['total_chars']:>7}")
        if info["error"]:
            print(f"     ⚠️  {info['error']}")
        if info["is_real_pdf"] and not info["suspicious"] and info["total_chars"] < 1000:
            print(f"     ⚠️  Подозрительно мало текста ({info['total_chars']} chars) — проверь вручную")
        if not info["is_real_pdf"] or info["suspicious"]:
            bad_files.append(info["file"])

    print()
    print("=" * 70)
    print("2. ПРОВЕРКА documents.jsonl (что реально попало в индекс)")
    print("=" * 70)

    stats = check_documents_jsonl()
    if "error" in stats:
        print(f"❌ {stats['error']}")
    else:
        # Сравниваем с ожидаемыми именами файлов
        expected_doc_ids = {p.stem for p in pdf_files}
        found_doc_ids = set(stats.keys())
        missing = expected_doc_ids - found_doc_ids

        for doc_id in sorted(stats.keys()):
            s = stats[doc_id]
            flag = "⚠️ МАЛО ТЕКСТА" if s["total_chars"] < 1000 else ""
            print(f"  {doc_id:<60} pages={s['pages']:>3}  chars={s['total_chars']:>7}  {flag}")

        if missing:
            print(f"\n❌ ОТСУТСТВУЮТ В documents.jsonl (есть файл, но 0 чанков):")
            for doc_id in sorted(missing):
                print(f"   - {doc_id}")

    print()
    print("=" * 70)
    print("ИТОГ")
    print("=" * 70)
    if bad_files:
        print(f"❌ {len(bad_files)} файл(ов) — это НЕ настоящие PDF (вероятно HTML-страницы ошибок):")
        for f in bad_files:
            print(f"   - {f}")
        print("\nРЕШЕНИЕ: удали эти файлы и перекачай с правильным User-Agent заголовком")
        print("(см. patched_download_sources.py)")
    else:
        print("✅ Все PDF файлы выглядят валидными.")
        print("Если RAG всё равно не находит нужную информацию — проблема в retrieval (top_k, chunk_size)")
        print("а не в данных. Смотри рекомендации по top_k / chunking ниже.")


if __name__ == "__main__":
    main()
