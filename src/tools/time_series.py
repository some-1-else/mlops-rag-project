"""
time_series.py — коннекторы к открытым источникам временных рядов.

Поддерживаемые источники:
    - ЦБ РФ (cbr.ru) — через XML API
    - Росстат (rosstat.gov.ru) — через CSV-файлы / HTML-таблицы
    - Кастомный CSV / URL

Зависимости:
    pip install requests pandas lxml

Публичный интерфейс (используется агентом / Ролью 2):
    get_cbr_series(series_id, start, end)   -> pd.DataFrame
    get_rosstat_series(indicator_code)      -> pd.DataFrame
    load_csv_series(path_or_url, ...)       -> pd.DataFrame
    list_cbr_series()                       -> list[dict]
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from io import StringIO, BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from lxml import etree

log = logging.getLogger(__name__)

# Таймаут HTTP-запросов (секунды)
_TIMEOUT = 30


# ---------------------------------------------------------------------------
# ЦБ РФ — XML API
# ---------------------------------------------------------------------------

# Популярные коды рядов ЦБ РФ (для справки)
CBR_KNOWN_SERIES: dict[str, str] = {
    "USD/RUB":  "R01235",
    "EUR/RUB":  "R01239",
    "CNY/RUB":  "R01375",
    "Ключевая ставка": "cbr_key_rate",    # особый эндпоинт
    "Инфляция (ИПЦ)": "cbr_inflation",    # Росстат / ЦБ
}

_CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
_CBR_KEY_RATE_URL = "https://www.cbr.ru/hd_base/KeyRate/"


def get_cbr_currency(
    series_id: str,
    start: str | date = "2020-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """
    Загружает дневной курс валюты с cbr.ru.

    Args:
        series_id: код валюты (например, 'R01235' для USD).
        start: начало периода 'YYYY-MM-DD'.
        end: конец периода (по умолчанию — сегодня).

    Returns:
        DataFrame с колонками ['date', 'value', 'nominal', 'name'].
    """
    if end is None:
        end = date.today()
    start_str = _fmt_cbr_date(start)
    end_str = _fmt_cbr_date(end)

    params = {
        "date_req1": start_str,
        "date_req2": end_str,
        "VAL_NM_RQ": series_id,
    }
    log.info("ЦБ РФ: запрос %s [%s — %s]", series_id, start_str, end_str)
    resp = requests.get(_CBR_DAILY_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    root = etree.fromstring(resp.content)
    records = []
    name = root.attrib.get("name", series_id)
    nominal = int(root.attrib.get("nominal", 1))
    for rec in root.findall("Record"):
        dt = datetime.strptime(rec.attrib["Date"], "%d.%m.%Y").date()
        val_str = rec.findtext("Value", "").replace(",", ".")
        try:
            val = float(val_str) / nominal
        except ValueError:
            continue
        records.append({"date": dt, "value": val, "nominal": nominal, "name": name})

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    log.info("  → получено %d записей", len(df))
    return df


def get_cbr_key_rate(
    start: str | date = "2020-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """
    Загружает историю ключевой ставки ЦБ РФ с сайта cbr.ru (парсинг HTML-таблицы).

    Returns:
        DataFrame ['date', 'value', 'name'].
    """
    if end is None:
        end = date.today()
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": _fmt_cbr_date(start, sep="."),
        "UniDbQuery.To": _fmt_cbr_date(end, sep="."),
    }
    log.info("ЦБ РФ: ключевая ставка [%s — %s]", start, end)
    resp = requests.get(_CBR_KEY_RATE_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text), decimal=",", thousands=" ")
    if not tables:
        raise ValueError("Не удалось распарсить таблицу ключевой ставки")
    df = tables[0].copy()
    # Обычно колонки: 'Дата' / 'Ставка, %' или аналогичные
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "дат" in c.lower()), df.columns[0])
    rate_col = next((c for c in df.columns if "став" in c.lower() or "%" in c), df.columns[1])
    df = df.rename(columns={date_col: "date", rate_col: "value"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(",", "."), errors="coerce")
    df = df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)
    df["name"] = "Ключевая ставка ЦБ РФ, %"
    log.info("  → получено %d записей", len(df))
    return df


# ---------------------------------------------------------------------------
# Росстат
# ---------------------------------------------------------------------------

_ROSSTAT_BASE = "https://rosstat.gov.ru"

# Примеры прямых CSV-ссылок Росстата (могут меняться — обновляйте при необходимости)
ROSSTAT_KNOWN_URLS: dict[str, str] = {
    "ИПЦ_месяц": "https://rosstat.gov.ru/storage/mediabank/ipc_mes.xlsx",
    "ВВП_квартал": "https://rosstat.gov.ru/storage/mediabank/vvp_kv_tab.xlsx",
    "Безработица": "https://rosstat.gov.ru/storage/mediabank/Urov_12kv.xlsx",
}


def get_rosstat_series(
    url_or_key: str,
    sheet: int | str = 0,
    date_col: str | int = 0,
    value_col: str | int = 1,
    skiprows: int = 0,
    name: str = "",
) -> pd.DataFrame:
    """
    Загружает ряд из Excel/CSV-файла Росстата по URL или ключу из ROSSTAT_KNOWN_URLS.

    Args:
        url_or_key: прямой URL или ключ из ROSSTAT_KNOWN_URLS.
        sheet: номер или название листа Excel.
        date_col: колонка с датами (имя или индекс).
        value_col: колонка со значениями (имя или индекс).
        skiprows: сколько строк пропустить.
        name: название ряда (для колонки 'name').

    Returns:
        DataFrame ['date', 'value', 'name'].
    """
    url = ROSSTAT_KNOWN_URLS.get(url_or_key, url_or_key)
    log.info("Росстат: загружаем %s …", url)
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()

    content = resp.content
    if url.endswith(".xlsx") or url.endswith(".xls"):
        df_raw = pd.read_excel(BytesIO(content), sheet_name=sheet,
                               skiprows=skiprows, header=0)
    else:
        df_raw = pd.read_csv(StringIO(content.decode("utf-8-sig")), skiprows=skiprows)

    # Выбираем нужные колонки
    cols = list(df_raw.columns)
    dc = cols[date_col] if isinstance(date_col, int) else date_col
    vc = cols[value_col] if isinstance(value_col, int) else value_col

    df = df_raw[[dc, vc]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    df["name"] = name or url_or_key
    log.info("  → получено %d записей", len(df))
    return df


# ---------------------------------------------------------------------------
# Кастомный CSV / URL
# ---------------------------------------------------------------------------

def load_csv_series(
    path_or_url: str,
    date_col: str = "date",
    value_col: str = "value",
    date_format: str | None = None,
    sep: str = ",",
    name: str = "",
) -> pd.DataFrame:
    """
    Загружает временной ряд из локального CSV или по URL.

    Returns:
        DataFrame ['date', 'value', 'name'].
    """
    log.info("CSV: загружаем %s …", path_or_url)
    if path_or_url.startswith("http"):
        resp = requests.get(path_or_url, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep=sep)
    else:
        df = pd.read_csv(path_or_url, sep=sep, encoding="utf-8-sig")

    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"], format=date_format, dayfirst=True,
                                errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)
    df["name"] = name or path_or_url
    log.info("  → получено %d записей", len(df))
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_cbr_date(d: str | date, sep: str = "/") -> str:
    if isinstance(d, str):
        d = datetime.strptime(d[:10], "%Y-%m-%d").date()
    return d.strftime(f"%d{sep}%m{sep}%Y")


def list_cbr_series() -> list[dict]:
    """Возвращает список известных рядов ЦБ РФ."""
    return [{"code": v, "name": k} for k, v in CBR_KNOWN_SERIES.items()]


def resample_series(df: pd.DataFrame, freq: str = "ME",
                    agg: str = "mean") -> pd.DataFrame:
    """
    Ресемплирует ряд на нужную частоту.

    Args:
        df: DataFrame с колонками 'date', 'value'.
        freq: 'D', 'W', 'ME', 'QE', 'YE'.
        agg: 'mean', 'last', 'sum'.

    Returns:
        Ресемплированный DataFrame.
    """
    ts = df.set_index("date")["value"]
    resampled = getattr(ts.resample(freq), agg)()
    result = resampled.reset_index()
    result.columns = ["date", "value"]
    if "name" in df.columns:
        result["name"] = df["name"].iloc[0]
    return result
