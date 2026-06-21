"""
parse_pdfs.py — ИСПРАВЛЕННАЯ версия.

Изменения относительно оригинала:
1. try/except вокруг parse_pdf() для каждого файла — один битый PDF больше
   не останавливает обработку всех последующих файлов.
2. Проверка магических байт (%PDF-) перед попыткой открыть файл через fitz.
3. Предупреждение, если из PDF извлеклось подозрительно мало текста на страницу
   (частый признак того, что PDF — это скан без OCR, или страница ошибки).
4. Итоговая сводка: какие файлы дали 0 страниц / мало текста — чтобы сразу
   увидеть проблемные источники, а не искать их потом через RAG-запросы.
"""

import json
from pathlib import Path

import fitz
from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "documents.jsonl"

# Если страница даёт меньше этого числа символов в среднем — подозрительно мало
# (скан без OCR, страница ошибки, пустой PDF и т.п.)
MIN_AVG_CHARS_PER_PAGE_WARNING = 200


def _looks_like_real_pdf(pdf_path: Path) -> bool:
    """Проверяет магические байты — действительно ли это PDF, а не HTML/JSON ошибка."""
    try:
        with pdf_path.open("rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except OSError:
        return False


def parse_pdf(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem
    rows = []

    for page_number in range(1, len(doc) + 1):
        page = doc[page_number - 1]
        content = page.get_text().strip()

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
    skipped_files = []      # не открылись / не прошли валидацию
    suspicious_files = []   # открылись, но текста подозрительно мало

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for pdf_path in pdf_files:
            # 1. Проверка магических байт — отсеиваем HTML-страницы ошибок,
            #    сохранённые с расширением .pdf (частый случай при заблокированных
            #    источниках вроде IMF без правильных заголовков запроса).
            if not _looks_like_real_pdf(pdf_path):
                print(f"⚠️  SKIPPED (not a real PDF, likely an error page): {pdf_path.name}")
                skipped_files.append(pdf_path.name)
                continue

            # 2. try/except вокруг каждого файла — один битый файл
            #    больше не останавливает весь батч.
            try:
                rows = parse_pdf(pdf_path)
            except Exception as exc:
                print(f"⚠️  SKIPPED (failed to parse: {exc}): {pdf_path.name}")
                skipped_files.append(pdf_path.name)
                continue

            total_pages += len(rows)
            print(f"Processed {pdf_path.name}: {len(rows)} non-empty pages")

            # 3. Предупреждение про мало текста (вероятный скан / пустышка)
            if rows:
                avg_chars = sum(len(r["content"]) for r in rows) / len(rows)
                if avg_chars < MIN_AVG_CHARS_PER_PAGE_WARNING:
                    print(f"   ⚠️  Average {avg_chars:.0f} chars/page — looks suspicious "
                          f"(scanned doc without OCR, or near-empty content)")
                    suspicious_files.append(pdf_path.name)
            else:
                print(f"   ⚠️  0 pages with text extracted")
                suspicious_files.append(pdf_path.name)

            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTotal processed pages: {total_pages}")
    print(f"Saved output to: {OUTPUT_PATH}")

    if skipped_files:
        print(f"\n❌ ПРОПУЩЕНО файлов (не PDF / не открылись): {len(skipped_files)}")
        for f in skipped_files:
            print(f"   - {f}")
        print("   → Перекачай эти файлы вручную или через исправленный download_sources.py")

    if suspicious_files:
        print(f"\n⚠️  ПОДОЗРИТЕЛЬНО МАЛО ТЕКСТА: {len(suspicious_files)}")
        for f in suspicious_files:
            print(f"   - {f}")
        print("   → Эти файлы попали в индекс, но RAG вряд ли найдёт в них полезную информацию")


if __name__ == "__main__":
    main()
