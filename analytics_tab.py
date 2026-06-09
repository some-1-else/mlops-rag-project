"""
analytics_tab.py — Вкладка №2: Text-to-Chart агент (LLM генерирует и запускает код).

Подключение в app.py:
    from analytics_tab import render_analytics_tab
    render_analytics_tab(generate_answer_fn=generate_answer)
"""

from __future__ import annotations

import re
import traceback
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

TIME_SERIES_DIR = Path("data/raw/time_series")

# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM_PROMPT = """You are a Python data-visualisation expert. Your ONLY job is to write executable Python code for Streamlit.

STRICT OUTPUT FORMAT — follow this exactly:
1. Output ONE and only ONE fenced code block: ```python ... ```
2. No explanations before or after the block. No markdown prose. No apologies.
3. The code block must be complete and runnable as-is.

AVAILABLE VARIABLES & FUNCTIONS (already in scope, do NOT redefine, import, or alter them):
- `file_paths` (dict[str, str]): mapping filename string -> absolute system path.
- `load_data(filename)`: FUNCTION to load any available file into a pandas DataFrame. Works seamlessly for both CSV and XML.
   Usage example: df = load_data("fred_gdp.csv")
- `pd`  : pandas module
- `px`  : plotly.express module
- `go`  : plotly.graph_objects module
- `st`  : streamlit module — use it to render outputs (e.g., st.plotly_chart).

FILES AVAILABLE (!!! CRITICAL: You can ONLY load filenames listed below !!!):
{file_paths_context}

CRITICAL CODING RULES:
1. NEVER invent filenames or use example filenames. ONLY use exact strings from the "FILES AVAILABLE" list above.
2. To read a file, always use: df = load_data("filename.ext")
3. Column Handling:
   - For FRED CSV files, columns are usually `observation_date` and a metric name (e.g., `GDP`, `UNRATE`).
   - For CBR XML files, columns are usually `Date` and `Value` (or `VunitRate`).
4. Always convert date columns to datetime and sort by date before plotting:
   date_col = "Date" if "Date" in df.columns else "observation_date"
   df["date"] = pd.to_datetime(df[date_col], errors="coerce")
   df = df.dropna(subset=["date"]).sort_values("date")
5. To combine multiple series on one chart, load them separately and plot using plotly graph_objects with multiple traces:
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=df1["date"], y=df1[val_col1], name="Series 1"))
   fig.add_trace(go.Scatter(x=df2["date"], y=df2[val_col2], name="Series 2", yaxis="y2"))
   fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), template="plotly_white")
6. Always end by rendering the chart: `st.plotly_chart(fig, use_container_width=True)`
7. Do NOT write any `import streamlit`, `import pandas`, or `import plotly`. They are already injected.

USER REQUEST:
{user_prompt}
"""

REFLECTION_SYSTEM_PROMPT = """You are a Python debugging expert. The code below raised an error. Fix it.

ORIGINAL CODE:
```python
{original_code}
ERROR MESSAGE:{error_message}AVAILABLE VARIABLES & FUNCTIONS (Do NOT import or redefine them):file_paths (dict[str, str])load_data(filename) — returns pandas DataFramepd, px, go, st — already in scope.FILES AVAILABLE CONTEXT:{file_paths_context}OUTPUT: ONE corrected python ...  block only. No prose."""

# ---------------------------------------------------------------------------
# Helpers: загрузка файлов
# ---------------------------------------------------------------------------

def _scan_data_files(directory: Path) -> list[Path]:
    """Возвращает все CSV и XML файлы в папке (отсортированные)."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".csv", ".xml"})

def _load_dataframe(file_path: Path) -> pd.DataFrame:
    """Загружает файл в DataFrame. Обрабатывает CSV и CBR XML."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(file_path, encoding="utf-8-sig")
    elif suffix == ".xml":
        return _parse_cbr_xml(file_path)
    raise ValueError(f"Неподдерживаемый формат: {suffix}")

def _parse_cbr_xml(file_path: Path) -> pd.DataFrame:
    """Парсит стандартный XML ЦБ РФ (XML_dynamic формат) в DataFrame."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    root = tree.getroot()

    name = root.attrib.get("name", file_path.stem)
    nominal_default = int(root.attrib.get("nominal", 1))

    records = []
    for rec in root.findall(".//Record"):
        row = dict(rec.attrib)
        for child in rec:
            row[child.tag] = child.text
        row.setdefault("Name", name)
        row.setdefault("Nominal", str(nominal_default))
        records.append(row)

    if not records:
        for child in root:
            row = dict(child.attrib)
            for sub in child:
                row[sub.tag] = sub.text
            records.append(row)

    df = pd.DataFrame(records)
    for col in df.columns:
        if col.lower() in ("value", "vunitrate"):
            df[col] = df[col].astype(str).str.replace(",", ".").pipe(
                pd.to_numeric, errors="coerce"
            )
    return df

# ---------------------------------------------------------------------------
# Helpers: контекст для промпта
# ---------------------------------------------------------------------------

def _build_multi_file_context(selected_files: list[Path], n_rows: int = 3) -> str:
    """Строит компактный контекст схем доступных файлов для модели."""
    blocks = []
    for fp in selected_files:
        buf = StringIO()
        buf.write(f'- Filename: "{fp.name}"\n')
        try:
            df = _load_dataframe(fp)
            buf.write(f"  Columns: {list(df.columns)}\n")
            dtype_str = ", ".join(f"{c} ({str(t)})" for c, t in df.dtypes.items())
            buf.write(f"  Types: {dtype_str}\n")
            buf.write(f"  First row sample: {df.head(1).to_dict(orient='records')}\n")
        except Exception as e:
            buf.write(f"  [Error parsing schema: {e}]\n")
        blocks.append(buf.getvalue())
    return "\n".join(blocks)

# ---------------------------------------------------------------------------
# Helpers: извлечение и выполнение кода
# ---------------------------------------------------------------------------

def _strip_bad_imports(code: str) -> str:
    """Удаляет импорты, которые модель по привычке может написать."""
    bad_patterns = [
        r"^\s*import\s+streamlit\b.*$",
        r"^\s*from\s+streamlit\b.*$",
        r"^\s*import\s+pandas\b.*$",
        r"^\s*import\s+plotly\b.*$",
    ]
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        if any(re.match(p, line) for p in bad_patterns):
            cleaned.append(f"# [removed by safety filter]: {line.strip()}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

def _extract_python_code(llm_response: str) -> str | None:
    """Вытаскивает первый блок ```python ... ``` из ответа модели."""
    match = re.search(r"```python\s*\n(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return _strip_bad_imports(match.group(1).strip())
    match2 = re.search(r"```\s*\n(.*?)```", llm_response, re.DOTALL)
    if match2:
        code = match2.group(1).strip()
        if any(kw in code for kw in ("import", "load_data", "pd.", "st.", "fig")):
            return _strip_bad_imports(code)
    return None

def _execute_code(code: str, file_paths_dict: dict[str, str]) -> str | None:
    """Выполняет код через exec(), инжектируя безопасную функцию load_data."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        px, go = None, None

    # Внутренний хелпер загрузки данных для exec-контекста
    def load_data(filename: str) -> pd.DataFrame:
        if filename not in file_paths_dict:
            available = list(file_paths_dict.keys())
            raise KeyError(
                f"Файл '{filename}' отсутствует в текущем контексте запроса. "
                f"Доступные для загрузки файлы: {available}. "
                f"Используйте СТРОГО их!"
            )
        return _load_dataframe(Path(file_paths_dict[filename]))

    exec_globals = {
        "pd": pd,
        "st": st,
        "px": px,
        "go": go,
        "file_paths": file_paths_dict,
        "load_data": load_data,
    }

    try:
        exec(compile(code, "<llm_generated>", "exec"), exec_globals)  # noqa: S102
        return None
    except Exception:
        return traceback.format_exc()

# ---------------------------------------------------------------------------
# Основная функция рендера вкладки
# ---------------------------------------------------------------------------

def render_analytics_tab(generate_answer_fn):
    """Рендерит вкладку Text-to-Chart агента."""
    st.header("📊 Text-to-Chart агент")
    st.caption("Опишите, какой график построить на основе имеющихся временных рядов. "
               "Агент автоматически сформирует код и визуализирует интерактивный чарт.")
    
    all_files = _scan_data_files(TIME_SERIES_DIR)
    if not all_files:
        st.warning(f"Файлы временных рядов не найдены в `{TIME_SERIES_DIR}`.")
        return

    file_labels = {p.name: p for p in all_files}

    col_left, col_right = st.columns([2, 1])

    with col_left:
        user_prompt = st.text_area(
            "Запрос к агенту",
            placeholder=(
                "Например: Сравни динамику курса доллара cbr_usd_rub_2025.xml "
                "и цену на нефть fred_wti_oil.csv на одном графике."
            ),
            height=130,
        )

    with col_right:
        selected_names = st.multiselect(
            "Источники данных",
            options=list(file_labels.keys()),
            placeholder="Пусто → видны ВСЕ файлы",
            help="Выберите файлы принудительно, либо оставьте пустым, чтобы модель выбирала сама.",
        )
        show_raw = st.checkbox("Показать предпросмотр таблиц", value=False)
        show_code = st.checkbox("Показать сгенерированный код", value=True)

    # Если ничего не выбрали — отдаем весь датасет в скоуп
    active_files: list[Path] = (
        [file_labels[n] for n in selected_names]
        if selected_names
        else list(all_files)
    )

    file_paths_dict: dict[str, str] = {fp.name: str(fp.resolve()) for fp in active_files}

    if show_raw:
        with st.expander(f"📄 Предпросмотр ({len(active_files)} файл(ов))", expanded=False):
            for fp in active_files:
                st.markdown(f"**{fp.name}**")
                try:
                    df_preview = _load_dataframe(fp)
                    st.dataframe(df_preview.head(3), use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка чтения {fp.name}: {e}")
                st.divider()

    run_btn = st.button("▶ Сгенерировать и запустить", type="primary", disabled=not user_prompt.strip())

    if not run_btn:
        return

    # --- Сборка контекста схем ---
    with st.spinner("Анализируем схемы таблиц…"):
        file_paths_context = _build_multi_file_context(active_files)

    # --- Генерация кода ---
    with st.spinner("Агент проектирует скрипт визуализации…"):
        system_content = CODEGEN_SYSTEM_PROMPT.format(
            file_paths_context=file_paths_context,
            user_prompt=user_prompt.strip(),
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt.strip()},
        ]
        try:
            llm_response = generate_answer_fn(messages)
        except Exception as e:
            st.error(f"Ошибка запроса к Ollama: {e}")
            return

    code = _extract_python_code(llm_response)
    if code is None:
        st.error("Модель не вернула валидный Python-блок.")
        with st.expander("Ответ модели"):
            st.text(llm_response)
        return

    if show_code:
        with st.expander("🔧 Сгенерированный код", expanded=True):
            st.code(code, language="python")

    st.divider()
    st.subheader("Результат визуализации")

    # --- Первое выполнение ---
    error_msg = _execute_code(code, file_paths_dict)
    if error_msg is None:
        return  # Успех, Plotly отрендерился

    # --- Если упало: Шаг Рефлексии ---
    st.warning("⚠️ Код упал. Запуск агента-отладчика для исправления ошибок…")
    with st.expander("Лог ошибки"):
        st.code(error_msg, language="text")

    with st.spinner("Исправление кода на основе лога ошибки…"):
        reflection_content = REFLECTION_SYSTEM_PROMPT.format(
            original_code=code,
            error_message=error_msg,
            file_paths_context=file_paths_context,
        )
        reflection_messages = [
            {"role": "system", "content": reflection_content},
            {"role": "user", "content": "Fix the execution error described above completely."},
        ]
        try:
            fixed_response = generate_answer_fn(reflection_messages)
        except Exception as e:
            st.error(f"Рефлексия завершилась неудачно: {e}")
            return

    fixed_code = _extract_python_code(fixed_response)
    if fixed_code is None:
        st.error("Не удалось извлечь исправленный код.")
        return

    if show_code:
        with st.expander("🔧 Исправленный код (Попытка 2)"):
            st.code(fixed_code, language="python")

    # --- Второе выполнение ---
    final_error = _execute_code(fixed_code, file_paths_dict)
    if final_error is not None:
        st.error("Агент не смог устранить ошибку за две итерации.")
        with st.expander("Финальный Traceback"):
            st.code(final_error, language="text")