import os
from pathlib import Path

from dotenv import load_dotenv
from src.tools.time_series import get_cbr_currency, get_cbr_key_rate
from src.tools.analytics import compute_correlation, compute_dynamics, plot_series

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

print("=== Loading Data ===")
df_rate = get_cbr_key_rate(start="2022-01-01", end="2024-12-31")
df_usd = get_cbr_currency(series_id="R01235", start="2022-01-01", end="2024-12-31")
print(f"Key rate: {len(df_rate)} rows, USD/RUB: {len(df_usd)} rows")

print("\n=== Computing Dynamics ===")
dyn = compute_dynamics(df_rate)
print(dyn.summary())

print("\n=== Computing Correlation ===")
corr = compute_correlation(df_rate, df_usd)
print(corr.summary())

print("\n=== Plotting ===")
plot_path = plot_series(df_rate, df_usd, title="Key Rate vs USD/RUB")
print(f"Chart saved: {plot_path}")

print("\n=== Success! ===")