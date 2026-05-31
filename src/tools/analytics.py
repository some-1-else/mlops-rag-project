"""
analytics.py — аналитический модуль: корреляции, лаги, регрессии, графики.

Зависимости:
    pip install pandas numpy scipy statsmodels plotly matplotlib

Публичный интерфейс (используется агентом / Ролью 2 как Tools):
    compute_correlation(s1, s2, ...)   -> CorrelationResult
    compute_lag_analysis(s1, s2, ...)  -> LagResult
    run_regression(y, x, ...)          -> RegressionResult
    compute_dynamics(series, ...)      -> DynamicsResult
    plot_series(...)                   -> str (путь к файлу)
    plot_correlation(...)              -> str
    plot_regression(...)               -> str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

# Директория для сохранения графиков (можно переопределить)
DEFAULT_OUTPUT_DIR = Path("output/charts")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    series1_name: str
    series2_name: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_obs: int

    def summary(self) -> str:
        sig = "значимо" if self.pearson_p < 0.05 else "незначимо"
        return (
            f"Корреляция '{self.series1_name}' и '{self.series2_name}': "
            f"Pearson r={self.pearson_r:.3f} (p={self.pearson_p:.4f}, {sig}), "
            f"Spearman r={self.spearman_r:.3f}, n={self.n_obs}"
        )


@dataclass
class LagResult:
    series1_name: str
    series2_name: str
    max_lag: int
    correlations: dict[int, float]   # lag -> pearson_r
    best_lag: int
    best_r: float

    def summary(self) -> str:
        return (
            f"Лаговый анализ '{self.series1_name}' → '{self.series2_name}': "
            f"лучший лаг = {self.best_lag} пер. (r={self.best_r:.3f})"
        )


@dataclass
class RegressionResult:
    y_name: str
    x_names: list[str]
    coefficients: dict[str, float]
    intercept: float
    r_squared: float
    adj_r_squared: float
    p_values: dict[str, float]
    n_obs: int
    formula: str

    def summary(self) -> str:
        sig_vars = [k for k, p in self.p_values.items() if p < 0.05 and k != "const"]
        return (
            f"Регрессия '{self.y_name}' ~ {' + '.join(self.x_names)}: "
            f"R²={self.r_squared:.3f}, adj. R²={self.adj_r_squared:.3f}, "
            f"n={self.n_obs}. "
            f"Значимые предикторы: {sig_vars or 'нет (p>0.05)'}"
        )


@dataclass
class DynamicsResult:
    series_name: str
    start_value: float
    end_value: float
    pct_change_total: float
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    yoy_changes: pd.DataFrame   # колонки: date, yoy_pct (год-к-году)

    def summary(self) -> str:
        direction = "выросла" if self.pct_change_total > 0 else "снизилась"
        return (
            f"'{self.series_name}': {direction} на {abs(self.pct_change_total):.1f}% "
            f"за период. Среднее={self.mean_value:.2f}, "
            f"мин={self.min_value:.2f}, макс={self.max_value:.2f}."
        )


# ---------------------------------------------------------------------------
# Core analytics functions
# ---------------------------------------------------------------------------

def compute_correlation(
    series1: pd.DataFrame,
    series2: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    resample_freq: str | None = "ME",
) -> CorrelationResult:
    """
    Считает Pearson и Spearman корреляцию между двумя временными рядами.

    Args:
        series1, series2: DataFrame с колонками 'date' и 'value'.
        resample_freq: частота ресемплинга перед объединением ('ME', 'QE', 'D', None).

    Returns:
        CorrelationResult.
    """
    df = _align_series(series1, series2, date_col, value_col, resample_freq)
    x, y = df["s1"].values, df["s2"].values

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)

    name1 = series1.get("name", pd.Series(["s1"])).iloc[0] if "name" in series1 else "Series 1"
    name2 = series2.get("name", pd.Series(["s2"])).iloc[0] if "name" in series2 else "Series 2"

    return CorrelationResult(
        series1_name=name1, series2_name=name2,
        pearson_r=round(pr, 4), pearson_p=round(pp, 6),
        spearman_r=round(sr, 4), spearman_p=round(sp, 6),
        n_obs=len(df),
    )


def compute_lag_analysis(
    series1: pd.DataFrame,
    series2: pd.DataFrame,
    max_lag: int = 12,
    date_col: str = "date",
    value_col: str = "value",
    resample_freq: str | None = "ME",
) -> LagResult:
    """
    Анализирует кросс-корреляцию при разных лагах (series1 лагируется).

    Args:
        max_lag: максимальный лаг в периодах.

    Returns:
        LagResult с таблицей корреляций и лучшим лагом.
    """
    df = _align_series(series1, series2, date_col, value_col, resample_freq)
    x, y = df["s1"].values, df["s2"].values

    corrs: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            s1_shifted = x
        elif lag > 0:
            s1_shifted = np.concatenate([np.full(lag, np.nan), x[:-lag]])
        else:
            s1_shifted = np.concatenate([x[-lag:], np.full(-lag, np.nan)])

        mask = ~np.isnan(s1_shifted)
        if mask.sum() < 5:
            corrs[lag] = np.nan
            continue
        r, _ = stats.pearsonr(s1_shifted[mask], y[mask])
        corrs[lag] = round(r, 4)

    best_lag = max(corrs, key=lambda l: abs(corrs[l]) if not np.isnan(corrs[l]) else 0)

    name1 = series1.get("name", pd.Series(["s1"])).iloc[0] if "name" in series1 else "Series 1"
    name2 = series2.get("name", pd.Series(["s2"])).iloc[0] if "name" in series2 else "Series 2"

    return LagResult(
        series1_name=name1, series2_name=name2,
        max_lag=max_lag, correlations=corrs,
        best_lag=best_lag, best_r=corrs[best_lag],
    )


def run_regression(
    y_series: pd.DataFrame,
    x_series_list: list[pd.DataFrame],
    date_col: str = "date",
    value_col: str = "value",
    resample_freq: str | None = "ME",
) -> RegressionResult:
    """
    Запускает OLS-регрессию (statsmodels).

    Args:
        y_series: зависимая переменная.
        x_series_list: список независимых переменных.

    Returns:
        RegressionResult.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("Установите statsmodels: pip install statsmodels")

    # Выравниваем все ряды
    merged = y_series.set_index(date_col)[[value_col]].rename(columns={value_col: "y"})
    x_names = []
    for i, xs in enumerate(x_series_list):
        xname = xs["name"].iloc[0] if "name" in xs.columns else f"x{i+1}"
        x_names.append(xname)
        tmp = xs.set_index(date_col)[[value_col]].rename(columns={value_col: xname})
        if resample_freq:
            tmp = tmp.resample(resample_freq).mean()
        merged = merged.join(tmp, how="inner")

    if resample_freq:
        merged = merged.resample(resample_freq).mean()
    merged = merged.dropna()

    y = merged["y"].values
    X = sm.add_constant(merged[x_names].values)
    model = sm.OLS(y, X).fit()

    coef_names = ["const"] + x_names
    coefficients = {n: round(model.params[i], 6) for i, n in enumerate(coef_names)}
    p_values = {n: round(model.pvalues[i], 6) for i, n in enumerate(coef_names)}

    y_name = y_series["name"].iloc[0] if "name" in y_series.columns else "y"
    formula = f"{y_name} = {coefficients['const']:.4f} " + " ".join(
        f"{'+ ' if v >= 0 else '- '}{abs(v):.4f}·{n}"
        for n, v in coefficients.items() if n != "const"
    )

    return RegressionResult(
        y_name=y_name, x_names=x_names,
        coefficients=coefficients, intercept=coefficients["const"],
        r_squared=round(model.rsquared, 4),
        adj_r_squared=round(model.rsquared_adj, 4),
        p_values=p_values, n_obs=int(model.nobs),
        formula=formula,
    )


def compute_dynamics(
    series: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
) -> DynamicsResult:
    """
    Считает динамику (изменение, мин/макс, год-к-году).

    Returns:
        DynamicsResult.
    """
    df = series[[date_col, value_col]].dropna().sort_values(date_col).copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])

    # YoY: группируем по году, берём среднее
    df_y = df.set_index("date")["value"].resample("YE").mean().reset_index()
    df_y.columns = ["date", "value"]
    df_y["yoy_pct"] = df_y["value"].pct_change() * 100

    name = series["name"].iloc[0] if "name" in series.columns else "Series"

    return DynamicsResult(
        series_name=name,
        start_value=round(df["value"].iloc[0], 4),
        end_value=round(df["value"].iloc[-1], 4),
        pct_change_total=round((df["value"].iloc[-1] / df["value"].iloc[0] - 1) * 100, 2),
        mean_value=round(df["value"].mean(), 4),
        std_value=round(df["value"].std(), 4),
        min_value=round(df["value"].min(), 4),
        max_value=round(df["value"].max(), 4),
        yoy_changes=df_y[["date", "yoy_pct"]].dropna(),
    )


# ---------------------------------------------------------------------------
# Plotting (Plotly → HTML / PNG)
# ---------------------------------------------------------------------------

def plot_series(
    *series_list: pd.DataFrame,
    title: str = "Временной ряд",
    y_label: str = "Значение",
    output_path: str | Path | None = None,
    as_html: bool = True,
) -> str:
    """
    Строит интерактивный график одного или нескольких временных рядов.

    Returns:
        Путь к сохранённому HTML (или PNG) файлу.
    """
    fig = go.Figure()
    for df in series_list:
        name = df["name"].iloc[0] if "name" in df.columns else "Series"
        fig.add_trace(go.Scatter(x=df["date"], y=df["value"],
                                 mode="lines", name=name))
    fig.update_layout(
        title=title, yaxis_title=y_label,
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _save_fig(fig, output_path or _auto_path("series"), as_html)


def plot_correlation(
    series1: pd.DataFrame,
    series2: pd.DataFrame,
    result: CorrelationResult | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    as_html: bool = True,
) -> str:
    """
    Scatter-plot двух рядов с линией тренда.

    Returns:
        Путь к файлу.
    """
    df = _align_series(series1, series2)
    name1 = result.series1_name if result else "X"
    name2 = result.series2_name if result else "Y"

    fig = px.scatter(df, x="s1", y="s2",
                     labels={"s1": name1, "s2": name2},
                     trendline="ols",
                     title=title or f"Корреляция: {name1} vs {name2}")
    if result:
        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"r = {result.pearson_r:.3f}  p = {result.pearson_p:.4f}",
            showarrow=False, bgcolor="white", bordercolor="gray",
        )
    fig.update_layout(template="plotly_white")
    return _save_fig(fig, output_path or _auto_path("corr"), as_html)


def plot_lag_analysis(
    result: LagResult,
    title: str | None = None,
    output_path: str | Path | None = None,
    as_html: bool = True,
) -> str:
    """
    Строит график кросс-корреляции (лаги vs r).

    Returns:
        Путь к файлу.
    """
    lags = sorted(result.correlations.keys())
    rs = [result.correlations[l] for l in lags]
    colors = ["crimson" if l == result.best_lag else "steelblue" for l in lags]

    fig = go.Figure(go.Bar(x=lags, y=rs, marker_color=colors))
    fig.update_layout(
        title=title or f"Лаговая корреляция: {result.series1_name} → {result.series2_name}",
        xaxis_title="Лаг (периоды)",
        yaxis_title="Pearson r",
        template="plotly_white",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return _save_fig(fig, output_path or _auto_path("lag"), as_html)


def plot_regression(
    y_series: pd.DataFrame,
    x_series: pd.DataFrame,
    result: RegressionResult | None = None,
    output_path: str | Path | None = None,
    as_html: bool = True,
) -> str:
    """
    График фактических vs предсказанных значений (для простой регрессии).
    """
    df = _align_series(y_series, x_series)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Факт vs Предсказание", "Остатки"])
    # Простая регрессия через scipy (для графика)
    slope, intercept, r, p, _ = stats.linregress(df["s2"], df["s1"])
    y_pred = slope * df["s2"] + intercept
    residuals = df["s1"] - y_pred

    fig.add_trace(go.Scatter(x=df["s2"], y=df["s1"], mode="markers",
                             name="Наблюдения"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["s2"], y=y_pred, mode="lines",
                             name="Линия регрессии", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(residuals))), y=residuals.values,
                             mode="markers+lines", name="Остатки"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    name_y = y_series["name"].iloc[0] if "name" in y_series.columns else "Y"
    name_x = x_series["name"].iloc[0] if "name" in x_series.columns else "X"
    fig.update_layout(title=f"Регрессия: {name_y} ~ {name_x}",
                      template="plotly_white")
    return _save_fig(fig, output_path or _auto_path("regression"), as_html)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_series(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    resample_freq: str | None = "ME",
) -> pd.DataFrame:
    """Объединяет два ряда по дате (inner join), опционально ресемплирует."""
    t1 = s1.set_index(date_col)[[value_col]].rename(columns={value_col: "s1"})
    t2 = s2.set_index(date_col)[[value_col]].rename(columns={value_col: "s2"})
    if resample_freq:
        t1 = t1.resample(resample_freq).mean()
        t2 = t2.resample(resample_freq).mean()
    merged = t1.join(t2, how="inner").dropna().reset_index()
    merged.columns = ["date", "s1", "s2"]
    return merged


def _save_fig(fig: go.Figure, path: str | Path, as_html: bool) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_html:
        path = path.with_suffix(".html")
        fig.write_html(str(path))
    else:
        path = path.with_suffix(".png")
        fig.write_image(str(path))
    log.info("График сохранён: %s", path)
    return str(path)


def _auto_path(prefix: str) -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / prefix
