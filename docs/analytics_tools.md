# Optional Data & Analytics Tools

## Структура файлов

```
src/tools/
├── pdf_parser.py       # парсинг PDF: текст, таблицы, изображения
├── time_series.py      # коннекторы к ЦБ РФ и Росстату
├── analytics.py        # корреляции, регрессии, графики
└── agent_tools.py      # всё выше — как optional Tools для агента
```

## Быстрый старт

```bash
pip install -r requirements.txt
```

---

## Что передаём Роли 2

**Файл:** `src/tools/agent_tools.py`  
**Функция:** `get_all_tools()` → возвращает список LangChain `StructuredTool`

```python
from src.tools.agent_tools import get_all_tools

tools = get_all_tools()
# передать в AgentExecutor / ReAct агент
```

### Список инструментов

| Tool | Описание |
|------|----------|
| `parse_pdf` | Парсит один PDF, возвращает текст + таблицы + мета изображений |
| `parse_pdf_directory` | Парсит все PDF в папке |
| `get_cbr_currency` | Курс валюты с ЦБ РФ (JSON ряда) |
| `get_cbr_key_rate` | Ключевая ставка ЦБ РФ |
| `get_rosstat_series` | Ряды Росстата (ИПЦ, ВВП, безработица) |
| `load_csv_series` | Любой CSV / URL |
| `list_cbr_series` | Список доступных кодов ЦБ РФ |
| `compute_correlation` | Pearson + Spearman корреляция двух рядов |
| `compute_lag_analysis` | Лаговая кросс-корреляция |
| `run_regression` | OLS регрессия (statsmodels) |
| `compute_dynamics` | Динамика ряда, YoY |
| `predict_series` | MVP-прогноз временного ряда через линейный тренд + holdout-валидация |
| `plot_series` | График рядов → HTML-файл |
| `plot_correlation` | Scatter-plot корреляции → HTML |
| `plot_lag_analysis` | График лагов → HTML |
| `plot_forecast` | График факта, тренда, прогноза и доверительного интервала → HTML |

---

## Формат данных между модулями

Все временные ряды передаются как JSON-строки (результат `df.to_json(orient="records")`).
Для pandas 3 используйте частоты ресемплинга `ME`, `QE`, `YE` вместо старых `M`, `Q`, `Y`.

```json
[
  {"date": "2023-01-01T00:00:00", "value": 72.5, "name": "USD/RUB"},
  ...
]
```

---

## Примеры

```python
# 1. Загрузить курс USD и ключевую ставку
from src.tools.agent_tools import tool_get_cbr_currency, tool_get_cbr_key_rate

usd_json = tool_get_cbr_currency("R01235", start="2022-01-01")
rate_json = tool_get_cbr_key_rate(start="2022-01-01")

# 2. Посчитать корреляцию
from src.tools.agent_tools import tool_compute_correlation
result = tool_compute_correlation(usd_json, rate_json)
print(result)

# 3. Построить график
from src.tools.agent_tools import tool_plot_series
path = tool_plot_series([usd_json, rate_json], title="USD/RUB и ключевая ставка")
print(path)  # output/charts/series.html

# 4. Лаговый анализ
from src.tools.agent_tools import tool_plot_lag_analysis
path = tool_plot_lag_analysis(usd_json, rate_json, max_lag=12)

# 5. Прогноз временного ряда
from src.tools.agent_tools import tool_predict_series, tool_plot_forecast

forecast = tool_predict_series(usd_json, steps=6)
path = tool_plot_forecast(usd_json, steps=6)

# 6. Парсинг PDF
from src.tools.agent_tools import tool_parse_pdf
summary = tool_parse_pdf("data/report.pdf")
```

## Что показывать на защите

- `load_csv_series`, `get_cbr_currency`, `get_cbr_key_rate` и `get_rosstat_series` закрывают гибкость источников: ряд можно брать из локального файла, URL, XML/HTML API ЦБ РФ или файлов Росстата.
- `predict_series` закрывает интеграцию прогнозной логики в контур агента: агент может не только строить исторические графики, но и вызывать прогнозный модуль.
- `predict_series` возвращает holdout MAE/MAPE, что можно описывать как MVP-валидацию модельного риска прогноза.
