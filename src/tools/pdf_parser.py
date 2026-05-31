"""
pdf_parser.py — извлечение текста, таблиц и изображений из PDF-файлов.

Зависимости:
    pip install pymupdf camelot-py[cv] pandas pillow

Публичный интерфейс (используется агентом / Ролью 2):
    parse_pdf(path)        -> ParsedDocument
    parse_directory(dir)   -> list[ParsedDocument]
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    page: int
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class TableChunk:
    page: int
    dataframe: pd.DataFrame
    caption: str = ""
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "caption": self.caption,
            "source_file": self.source_file,
            "data": self.dataframe.to_dict(orient="records"),
            "columns": list(self.dataframe.columns),
        }

    def to_markdown(self) -> str:
        return self.dataframe.to_markdown(index=False)


@dataclass
class ImageChunk:
    page: int
    image: Image.Image
    caption: str = ""
    source_file: str = ""
    save_path: str = ""     # заполняется после save_images()

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "caption": self.caption,
            "source_file": self.source_file,
            "save_path": self.save_path,
        }


@dataclass
class ParsedDocument:
    source_path: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    tables: list[TableChunk] = field(default_factory=list)
    images: list[ImageChunk] = field(default_factory=list)

    # ---- convenience ----

    def full_text(self) -> str:
        """Весь текст документа одной строкой (для RAG-индексации)."""
        return "\n\n".join(c.text for c in self.text_chunks if c.text.strip())

    def page_text(self, page: int) -> str:
        return "\n".join(c.text for c in self.text_chunks if c.page == page)

    def summary(self) -> dict:
        return {
            "source": self.source_path,
            "pages_with_text": len({c.page for c in self.text_chunks}),
            "tables": len(self.tables),
            "images": len(self.images),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text_chunks(doc: fitz.Document, source: str) -> list[TextChunk]:
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")          # (x0,y0,x1,y1,text,block_no,block_type)
        for b in blocks:
            if b[6] == 0:                         # block_type 0 = text
                text = b[4].strip()
                if text:
                    chunks.append(TextChunk(page=page_num, text=text, bbox=b[:4]))
    return chunks


def _extract_images(doc: fitz.Document, source: str, min_px: int = 50) -> list[ImageChunk]:
    """Извлекает растровые изображения (графики) со страниц."""
    images = []
    for page_num, page in enumerate(doc, start=1):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                w, h = pil_img.size
                if w < min_px or h < min_px:
                    continue  # пропускаем иконки
                # Простейшая эвристика: ищем подпись к рисунку рядом с bbox
                caption = _find_caption(page, img_info)
                images.append(ImageChunk(
                    page=page_num,
                    image=pil_img,
                    caption=caption,
                    source_file=source,
                ))
            except Exception as e:
                log.warning("Не удалось извлечь изображение xref=%s: %s", xref, e)
    return images


def _find_caption(page: fitz.Page, img_info) -> str:
    """Ищет строку 'Рис.' / 'Fig.' / 'Chart' рядом с изображением."""
    keywords = ("рис.", "fig.", "chart", "график", "диаграмм", "источник")
    for block in page.get_text("blocks"):
        if block[6] == 0:
            txt = block[4].strip().lower()
            if any(k in txt for k in keywords):
                return block[4].strip()[:200]
    return ""


def _extract_tables_camelot(pdf_path: str, source: str) -> list[TableChunk]:
    """Извлекает таблицы через Camelot (lattice → stream fallback)."""
    try:
        import camelot
    except ImportError:
        log.warning("camelot не установлен, таблицы пропущены")
        return []

    chunks = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
            for t in tables:
                df = t.df
                # Первая строка как заголовки, если там нет цифр
                if not df.empty:
                    header = df.iloc[0]
                    if not header.str.match(r"^\d").any():
                        df.columns = header
                        df = df[1:].reset_index(drop=True)
                chunks.append(TableChunk(
                    page=t.page,
                    dataframe=df,
                    source_file=source,
                ))
        except Exception as e:
            log.debug("Camelot %s не дал результат: %s", flavor, e)
        if chunks:
            break   # нашли в первом режиме — достаточно
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(path: str | Path, extract_tables: bool = True) -> ParsedDocument:
    """
    Парсит один PDF-файл.

    Args:
        path: путь к PDF.
        extract_tables: извлекать таблицы через Camelot.

    Returns:
        ParsedDocument с text_chunks, tables, images.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    log.info("Парсим %s …", path.name)
    doc = fitz.open(str(path))
    source = str(path)

    text_chunks = _extract_text_chunks(doc, source)
    images = _extract_images(doc, source)
    tables = _extract_tables_camelot(source, source) if extract_tables else []

    doc.close()
    log.info("  → текст: %d блоков, таблиц: %d, изображений: %d",
             len(text_chunks), len(tables), len(images))
    return ParsedDocument(source_path=source,
                          text_chunks=text_chunks,
                          tables=tables,
                          images=images)


def parse_directory(directory: str | Path,
                    extract_tables: bool = True) -> list[ParsedDocument]:
    """
    Парсит все PDF в папке (рекурсивно).

    Returns:
        Список ParsedDocument, по одному на файл.
    """
    directory = Path(directory)
    pdfs = sorted(directory.rglob("*.pdf"))
    log.info("Найдено %d PDF в %s", len(pdfs), directory)
    results = []
    for pdf in pdfs:
        try:
            results.append(parse_pdf(pdf, extract_tables=extract_tables))
        except Exception as e:
            log.error("Ошибка при парсинге %s: %s", pdf.name, e)
    return results


def save_images(doc: ParsedDocument, output_dir: str | Path) -> None:
    """
    Сохраняет все изображения документа на диск.
    Заполняет поле ImageChunk.save_path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.source_path).stem
    for i, img_chunk in enumerate(doc.images):
        fname = out / f"{stem}_p{img_chunk.page}_img{i}.jpg"
        img_chunk.image.save(fname, "JPEG", quality=85)
        img_chunk.save_path = str(fname)
    log.info("Сохранено %d изображений в %s", len(doc.images), out)


def export_tables_to_csv(doc: ParsedDocument, output_dir: str | Path) -> list[str]:
    """
    Сохраняет все таблицы документа как CSV-файлы.

    Returns:
        Список путей к сохранённым файлам.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.source_path).stem
    saved = []
    for i, tbl in enumerate(doc.tables):
        fname = out / f"{stem}_p{tbl.page}_table{i}.csv"
        tbl.dataframe.to_csv(fname, index=False, encoding="utf-8-sig")
        tbl.source_file = str(fname)
        saved.append(str(fname))
    return saved
