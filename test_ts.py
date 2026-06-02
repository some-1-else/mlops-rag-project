import pandas as pd
from src.tools.time_series import get_cbr_currency, get_cbr_key_rate

print("=== Testing Time Series Loading ===")

print("\n1. Loading CBR Key Rate...")
df_rate = get_cbr_key_rate(start="2022-01-01", end="2024-12-31")
print(f"Key rate rows: {len(df_rate)}")
print(df_rate.head())

print("\n2. Loading USD/RUB...")
df_usd = get_cbr_currency(series_id="R01235", start="2022-01-01", end="2024-12-31")
print(f"USD/RUB rows: {len(df_usd)}")
print(df_usd.head())

print("\n3. Loading EUR/RUB...")
df_eur = get_cbr_currency(series_id="R01239", start="2022-01-01", end="2024-12-31")
print(f"EUR/RUB rows: {len(df_eur)}")
print(df_eur.head())

print("\n=== Success! ===")