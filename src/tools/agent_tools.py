"""
agent_tools.py — обёртки Роли 1 в виде инструментов (Tools) для агента.

Роль 2 подключает этот файл и передаёт tools в LangChain / LlamaIndex / LangGraph.

Использование (Роль 2):
    from tools.agent_tools import get_all_tools
    tools = get_all_tools()
    # → передать в AgentExecutor / ReAct агент

Зависимости (дополнительно):
    pip install langchain langchain-core pydantic
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Type

import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

from src.tools.pdf_parser import parse_pdf, parse_directory, save_images, export_tables_to_csv
from src.tools.time_series import (
    get_cbr_currency, get_cbr_key_rate, get_rosstat_series,
    load_csv_series, resample_series, list_cbr_series,
)
from src.tools.analytics import (
    compute_correlation, compute_lag_analysis, run_regression,
    compute_dynamics, plot_series, plot_correlation,
    plot_lag_analysis, plot_regression, predict_series, plot_forecast,
)

# ---------------------------------------------------------------------------
# Pydantic schemas (описывают аргументы для LLM)
# ---------------------------------------------------------------------------

class ParsePDFInput(BaseModel):
    path: str = Field(..., description="Путь к PDF-файлу")
    extract_tables: bool = Field(True, description="Извлекать таблицы (Camelot)")


class ParseDirectoryInput(BaseModel):
    directory: str = Field(..., description="Путь к папке с PDF-файлами")
    save_images_dir: Optional[str] = Field(None, description="Папка для сохранения изображений")
    save_tables_dir: Optional[str] = Field(None, description="Папка для сохранения таблиц CSV")


class GetCBRCurrencyInput(BaseModel):
    series_id: str = Field(..., description="Код валюты ЦБ РФ, например R01235 для USD/RUB")
    start: str = Field("2020-01-01", description="Начало периода YYYY-MM-DD")
    end: Optional[str] = Field(None, description="Конец периода YYYY-MM-DD (по умолчанию сегодня)")


class GetCBRKeyRateInput(BaseModel):
    start: str = Field("2020-01-01", description="Начало периода YYYY-MM-DD")
    end: Optional[str] = Field(None, description="Конец периода YYYY-MM-DD")


class GetRosstatInput(BaseModel):
    url_or_key: str = Field(..., description="URL или ключ из ROSSTAT_KNOWN_URLS")
    sheet: int = Field(0, description="Номер листа Excel")
    date_col: int = Field(0, description="Индекс колонки с датами")
    value_col: int = Field(1, description="Индекс колонки со значениями")
    skiprows: int = Field(0, description="Строк пропустить в начале файла")
    name: str = Field("", description="Название ряда")


class LoadCSVInput(BaseModel):
    path_or_url: str = Field(..., description="Путь или URL к CSV-файлу")
    date_col: str = Field("date", description="Название колонки с датами")
    value_col: str = Field("value", description="Название колонки со значениями")
    sep: str = Field(",", description="Разделитель CSV")
    name: str = Field("", description="Название ряда")


class CorrelationInput(BaseModel):
    series1_json: str = Field(..., description="JSON-строка DataFrame ряда 1 (to_json(orient='records'))")
    series2_json: str = Field(..., description="JSON-строка DataFrame ряда 2")
    resample_freq: str = Field("ME", description="Частота ресемплинга: D, W, ME, QE, YE")


class LagAnalysisInput(BaseModel):
    series1_json: str = Field(..., description="JSON-строка DataFrame ряда 1")
    series2_json: str = Field(..., description="JSON-строка DataFrame ряда 2")
    max_lag: int = Field(12, description="Максимальный лаг в периодах")
    resample_freq: str = Field("ME", description="Частота ресемплинга")


class RegressionInput(BaseModel):
    y_json: str = Field(..., description="JSON-строка зависимой переменной")
    x_json_list: list[str] = Field(..., description="Список JSON-строк независимых переменных")
    resample_freq: str = Field("ME", description="Частота ресемплинга")


class DynamicsInput(BaseModel):
    series_json: str = Field(..., description="JSON-строка временного ряда")


class ForecastInput(BaseModel):
    series_json: str = Field(..., description="JSON-строка временного ряда")
    steps: int = Field(6, description="Горизонт прогноза в периодах")
    resample_freq: str = Field("ME", description="Частота ресемплинга: D, W, ME, QE, YE")


class PlotSeriesInput(BaseModel):
    series_json_list: list[str] = Field(..., description="Список JSON-строк рядов")
    title: str = Field("Временной ряд", description="Заголовок графика")
    output_path: Optional[str] = Field(None, description="Путь для сохранения (без расширения)")


class PlotCorrelationInput(BaseModel):
    series1_json: str = Field(..., description="JSON ряда 1")
    series2_json: str = Field(..., description="JSON ряда 2")
    output_path: Optional[str] = Field(None, description="Путь для сохранения")


class PlotLagInput(BaseModel):
    series1_json: str = Field(..., description="JSON ряда 1")
    series2_json: str = Field(..., description="JSON ряда 2")
    max_lag: int = Field(12, description="Максимальный лаг")
    output_path: Optional[str] = Field(None, description="Путь для сохранения")


class PlotForecastInput(BaseModel):
    series_json: str = Field(..., description="JSON временного ряда")
    steps: int = Field(6, description="Горизонт прогноза в периодах")
    resample_freq: str = Field("ME", description="Частота ресемплинга: D, W, ME, QE, YE")
    output_path: Optional[str] = Field(None, description="Путь для сохранения")


# ---------------------------------------------------------------------------
# Tool functions (принимают строки, возвращают строки — для LLM)
# ---------------------------------------------------------------------------

def tool_parse_pdf(path: str, extract_tables: bool = True) -> str:
    """Парсит PDF и возвращает резюме + текст документа."""
    doc = parse_pdf(path, extract_tables=extract_tables)
    result = {
        "summary": doc.summary(),
        "full_text_preview": doc.full_text()[:2000],
        "tables": [t.to_dict() for t in doc.tables[:5]],
        "images_meta": [img.to_dict() for img in doc.images[:10]],
    }
    return json.dumps(result, ensure_ascii=False, default=str)


def tool_parse_directory(
    directory: str,
    save_images_dir: str | None = None,
    save_tables_dir: str | None = None,
) -> str:
    """
    Парсит все PDF в папке. Опционально сохраняет изображения и таблицы.
    Возвращает список резюме по каждому файлу.
    """
    docs = parse_directory(directory)
    summaries = []
    for doc in docs:
        if save_images_dir:
            save_images(doc, save_images_dir)
        if save_tables_dir:
            export_tables_to_csv(doc, save_tables_dir)
        summaries.append(doc.summary())
    return json.dumps(summaries, ensure_ascii=False, default=str)


def tool_get_cbr_currency(series_id: str, start: str = "2020-01-01",
                           end: str | None = None) -> str:
    """Загружает курс валюты с ЦБ РФ. Возвращает JSON временного ряда."""
    df = get_cbr_currency(series_id, start, end)
    return df.to_json(orient="records", date_format="iso", force_ascii=False)


def tool_get_cbr_key_rate(start: str = "2020-01-01", end: str | None = None) -> str:
    """Загружает историю ключевой ставки ЦБ РФ. Возвращает JSON."""
    df = get_cbr_key_rate(start, end)
    return df.to_json(orient="records", date_format="iso", force_ascii=False)


def tool_get_rosstat_series(url_or_key: str, sheet: int = 0,
                             date_col: int = 0, value_col: int = 1,
                             skiprows: int = 0, name: str = "") -> str:
    """Загружает ряд с Росстата. Возвращает JSON."""
    df = get_rosstat_series(url_or_key, sheet, date_col, value_col, skiprows, name)
    return df.to_json(orient="records", date_format="iso", force_ascii=False)


def tool_load_csv_series(path_or_url: str, date_col: str = "date",
                          value_col: str = "value", sep: str = ",",
                          name: str = "") -> str:
    """Загружает временной ряд из CSV / URL. Возвращает JSON."""
    df = load_csv_series(path_or_url, date_col, value_col, sep=sep, name=name)
    return df.to_json(orient="records", date_format="iso", force_ascii=False)


def tool_list_cbr_series() -> str:
    """Возвращает список доступных кодов ЦБ РФ."""
    return json.dumps(list_cbr_series(), ensure_ascii=False)


def tool_compute_correlation(series1_json: str, series2_json: str,
                              resample_freq: str = "ME") -> str:
    """Считает корреляцию двух рядов. Аргументы — JSON-строки."""
    s1 = _json_to_df(series1_json)
    s2 = _json_to_df(series2_json)
    result = compute_correlation(s1, s2, resample_freq=resample_freq)
    return result.summary() + "\n" + json.dumps(result.__dict__,
                                                  ensure_ascii=False, default=str)


def tool_compute_lag_analysis(series1_json: str, series2_json: str,
                               max_lag: int = 12, resample_freq: str = "ME") -> str:
    """Лаговый анализ двух рядов."""
    s1 = _json_to_df(series1_json)
    s2 = _json_to_df(series2_json)
    result = compute_lag_analysis(s1, s2, max_lag=max_lag, resample_freq=resample_freq)
    out = {
        "summary": result.summary(),
        "best_lag": result.best_lag,
        "best_r": result.best_r,
        "correlations": result.correlations,
    }
    return json.dumps(out, ensure_ascii=False, default=str)


def tool_run_regression(y_json: str, x_json_list: list[str],
                         resample_freq: str = "ME") -> str:
    """OLS-регрессия. y_json — зависимая, x_json_list — независимые."""
    y = _json_to_df(y_json)
    xs = [_json_to_df(xj) for xj in x_json_list]
    result = run_regression(y, xs, resample_freq=resample_freq)
    return result.summary() + "\n" + json.dumps({
        "formula": result.formula,
        "r_squared": result.r_squared,
        "adj_r_squared": result.adj_r_squared,
        "coefficients": result.coefficients,
        "p_values": result.p_values,
        "n_obs": result.n_obs,
    }, ensure_ascii=False)


def tool_compute_dynamics(series_json: str) -> str:
    """Анализирует динамику ряда (рост, мин/макс, год-к-году)."""
    df = _json_to_df(series_json)
    result = compute_dynamics(df)
    return result.summary() + "\n" + json.dumps({
        "start_value": result.start_value,
        "end_value": result.end_value,
        "pct_change_total": result.pct_change_total,
        "mean": result.mean_value,
        "std": result.std_value,
        "min": result.min_value,
        "max": result.max_value,
    }, ensure_ascii=False)


def tool_predict_series(series_json: str, steps: int = 6, resample_freq: str = "ME") -> str:
    """Строит простой ML/MVP-прогноз ряда через линейный тренд и holdout-валидацию."""
    df = _json_to_df(series_json)
    result = predict_series(df, steps=steps, resample_freq=resample_freq)
    return result.summary() + "\n" + json.dumps({
        "series_name": result.series_name,
        "steps": result.steps,
        "frequency": result.frequency,
        "slope_per_period": result.slope_per_period,
        "last_actual_date": result.last_actual_date,
        "last_actual_value": result.last_actual_value,
        "holdout_mae": result.holdout_mae,
        "holdout_mape": result.holdout_mape,
        "forecast": result.forecast.to_dict(orient="records"),
    }, ensure_ascii=False, default=str)


def tool_plot_series(series_json_list: list[str], title: str = "Временной ряд",
                     output_path: str | None = None) -> str:
    """Строит график временных рядов. Возвращает путь к HTML-файлу."""
    dfs = [_json_to_df(j) for j in series_json_list]
    path = plot_series(*dfs, title=title, output_path=output_path)
    return f"График сохранён: {path}"


def tool_plot_correlation(series1_json: str, series2_json: str,
                           output_path: str | None = None) -> str:
    """Строит scatter-plot корреляции. Возвращает путь к HTML."""
    s1, s2 = _json_to_df(series1_json), _json_to_df(series2_json)
    result = compute_correlation(s1, s2)
    path = plot_correlation(s1, s2, result=result, output_path=output_path)
    return f"График корреляции сохранён: {path}"


def tool_plot_lag_analysis(series1_json: str, series2_json: str,
                            max_lag: int = 12,
                            output_path: str | None = None) -> str:
    """Строит график лагового анализа."""
    s1, s2 = _json_to_df(series1_json), _json_to_df(series2_json)
    result = compute_lag_analysis(s1, s2, max_lag=max_lag)
    path = plot_lag_analysis(result, output_path=output_path)
    return f"График лагов сохранён: {path}\n{result.summary()}"


def tool_plot_forecast(
    series_json: str,
    steps: int = 6,
    resample_freq: str = "ME",
    output_path: str | None = None,
) -> str:
    """Строит график факта, тренда и прогнозных точек. Возвращает путь к HTML."""
    df = _json_to_df(series_json)
    result = predict_series(df, steps=steps, resample_freq=resample_freq)
    path = plot_forecast(df, result, output_path=output_path)
    return f"График прогноза сохранён: {path}\n{result.summary()}"


# ---------------------------------------------------------------------------
# LangChain StructuredTool integration
# ---------------------------------------------------------------------------

def get_all_tools() -> list:
    """
    Возвращает список LangChain StructuredTool для всех функций Роли 1.

    Использование (Роль 2):
        from langchain.agents import AgentExecutor, create_react_agent
        from tools.agent_tools import get_all_tools

        tools = get_all_tools()
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        try:
            from langchain.tools import StructuredTool
        except ImportError as exc:
            raise ImportError(
                "Установите langchain: pip install langchain langchain-core"
            ) from exc

    tool_specs = [
        ("parse_pdf", tool_parse_pdf, ParsePDFInput,
         "Парсит один PDF-файл: извлекает текст, таблицы и изображения. "
         "Используй, когда нужно получить содержимое документа."),

        ("parse_pdf_directory", tool_parse_directory, ParseDirectoryInput,
         "Парсит все PDF в указанной папке. Опционально сохраняет таблицы и изображения."),

        ("get_cbr_currency", tool_get_cbr_currency, GetCBRCurrencyInput,
         "Загружает дневной курс валюты с сайта ЦБ РФ. "
         "series_id: R01235=USD, R01239=EUR, R01375=CNY. "
         "Возвращает JSON временного ряда."),

        ("get_cbr_key_rate", tool_get_cbr_key_rate, GetCBRKeyRateInput,
         "Загружает историю ключевой ставки ЦБ РФ. Возвращает JSON временного ряда."),

        ("get_rosstat_series", tool_get_rosstat_series, GetRosstatInput,
         "Загружает макроэкономический ряд с Росстата (Excel/CSV). "
         "Доступные ключи: ИПЦ_месяц, ВВП_квартал, Безработица."),

        ("load_csv_series", tool_load_csv_series, LoadCSVInput,
         "Загружает временной ряд из CSV-файла или URL. "
         "Укажи названия колонок с датой и значением."),

        ("list_cbr_series", lambda: tool_list_cbr_series(), None,
         "Возвращает список доступных кодов рядов ЦБ РФ."),

        ("compute_correlation", tool_compute_correlation, CorrelationInput,
         "Считает корреляцию Pearson и Spearman между двумя временными рядами. "
         "Аргументы — JSON-строки рядов (результат get_cbr_* или load_csv_series)."),

        ("compute_lag_analysis", tool_compute_lag_analysis, LagAnalysisInput,
         "Анализирует лаговую (кросс-)корреляцию между двумя рядами. "
         "Находит, с каким сдвигом один ряд лучше объясняет другой."),

        ("run_regression", tool_run_regression, RegressionInput,
         "Запускает OLS-регрессию: y_json — зависимая переменная, "
         "x_json_list — список независимых (JSON-строки рядов)."),

        ("compute_dynamics", tool_compute_dynamics, DynamicsInput,
         "Анализирует динамику временного ряда: рост/падение, мин/макс, год-к-году."),

        ("predict_series", tool_predict_series, ForecastInput,
         "Строит MVP-прогноз временного ряда на N периодов через линейный тренд. "
         "Используй, когда пользователь просит спрогнозировать, продлить график, "
         "показать будущий тренд или оценить модельный риск прогноза."),

        ("plot_series", tool_plot_series, PlotSeriesInput,
         "Строит интерактивный график временных рядов (Plotly HTML). "
         "series_json_list — список JSON-строк рядов."),

        ("plot_correlation", tool_plot_correlation, PlotCorrelationInput,
         "Строит scatter-plot корреляции двух рядов с линией тренда."),

        ("plot_lag_analysis", tool_plot_lag_analysis, PlotLagInput,
         "Строит столбчатый график кросс-корреляции при разных лагах."),

        ("plot_forecast", tool_plot_forecast, PlotForecastInput,
         "Строит график факта, тренда и прогноза временного ряда."),
    ]

    tools = []
    for spec in tool_specs:
        name, func, schema, desc = spec
        if schema is None:
            tools.append(StructuredTool.from_function(
                func=func, name=name, description=desc,
            ))
        else:
            tools.append(StructuredTool.from_function(
                func=func, name=name, description=desc,
                args_schema=schema,
            ))
    return tools


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _json_to_df(json_str: str) -> pd.DataFrame:
    """Конвертирует JSON-строку в DataFrame."""
    import json
    records = json.loads(json_str)
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
