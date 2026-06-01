# Sources Registry

Last updated: 2026-06-01

The source list was extended from `add_to_project/ссылки_на_файлы_для_скачивания.docx`.
Use `python scripts/download_sources.py` to download the configured files.

## PDF Sources Used By RAG

Files are saved to `data/raw/` and are parsed by `scripts/parse_pdfs.py`.

| # | Local file | Source | Original URL | Notes |
|---|---|---|---|---|
| 1 | `data/raw/bank_of_russia_monetary_policy_guidelines_2025_2027.pdf` | Bank of Russia — Monetary Policy Guidelines for 2025-2027 | https://www.cbr.ru/content/document/file/165597/on_eng_2025%282026-2027%29.pdf | Original MVP source. |
| 2 | `data/raw/bank_of_russia_monetary_policy_report_2023_04.pdf` | Bank of Russia — Monetary Policy Report No. 4 (44), October 2023 | https://www.cbr.ru/Collection/Collection/File/46554/2023_04_ddcp_e.pdf | Original MVP source. |
| 3 | `data/raw/bank_of_russia_medium_term_forecast_2024_02_16.pdf` | Bank of Russia — Medium-term forecast, 16 February 2024 | https://www.cbr.ru/Collection/Collection/File/48892/forecast_240216_e.pdf | Original MVP source. |
| 4 | `data/raw/rosstat_social_economic_position_russia_2026_01_03.pdf` | Rosstat — Социально-экономическое положение России, январь-март 2026 года | https://rosstat.gov.ru/storage/mediabank/osn-03-2026.pdf | Original MVP source. |
| 5 | `data/raw/mineconomy_forecast_social_economic_development_rf_2026_2028.pdf` | Минэкономразвития России — Прогноз социально-экономического развития РФ на 2026 год и на плановый период 2027 и 2028 годов | https://www.economy.gov.ru/material/file/download/bc142016f6ab3772370bb0b4541fc778/prognoz_socialno_ekonomicheskogo_razvitiya_rf_2026-2028.pdf | Original MVP source. |
| 6 | `data/raw/cbr_regional_economy_2026_04.pdf` | Bank of Russia — Региональная экономика: комментарии ГУ, № 43, апрель 2026 | https://www.cbr.ru/Collection/Collection/File/60838/0426.pdf | Added from colleague source links. |
| 7 | `data/raw/cbr_financial_stability_review_2025q4_2026q1.pdf` | Bank of Russia — Обзор финансовой стабильности, IV квартал 2025 - I квартал 2026 | https://www.cbr.ru/Collection/Collection/File/60978/4q_2025_1q_2026.pdf | Added from colleague source links. |
| 8 | `data/raw/cbr_inflation_expectations_2026_05.pdf` | Bank of Russia — Инфляционные ожидания и потребительские настроения, май 2026 | https://www.cbr.ru/Collection/Collection/File/61003/Infl_exp_26-05.pdf | Added from colleague source links. |
| 9 | `data/raw/cbr_inflation_russia_2026_04.pdf` | Bank of Russia — Инфляция в России, № 4 (124), апрель 2026 | https://www.cbr.ru/Collection/Collection/File/60988/CPD_2026-4.pdf | Added from colleague source links. |
| 10 | `data/raw/cbr_financial_market_risk_review_2026_04.pdf` | Bank of Russia — Обзор рисков финансовых рынков, апрель 2026 | https://www.cbr.ru/Collection/Collection/File/60961/ORFR_2026-4.pdf | Added from colleague source links. |
| 11 | `data/raw/imf_world_economic_outlook_2026_04.pdf` | IMF — World Economic Outlook, April 2026 | https://www.imf.org/-/media/files/publications/weo/2026/april/english/text.pdf | Added from colleague source links. |
| 12 | `data/raw/imf_global_financial_stability_report_2026_04.pdf` | IMF — Global Financial Stability Report, April 2026 | https://www.imf.org/-/media/files/publications/gfsr/2026/april/english/text.pdf | Added from colleague source links. |
| 13 | `data/raw/imf_fiscal_monitor_2026_04.pdf` | IMF — Fiscal Monitor, April 2026 | https://www.imf.org/-/media/files/publications/fiscal-monitor/2026/april/english/text.pdf | Added from colleague source links. |

## Source Pages

- Bank of Russia — Monetary Policy Guidelines for 2025-2027: https://www.cbr.ru/eng/about_br/publ/ondkp/on_2025_2027/
- Bank of Russia — Monetary Policy Report archive: https://www.cbr.ru/eng/about_br/publ/ddkp/
- Bank of Russia — Региональная экономика: комментарии ГУ: https://www.cbr.ru/analytics/dkp/reg_review/
- Bank of Russia — Обзор финансовой стабильности: https://www.cbr.ru/analytics/finstab/ofs/
- Bank of Russia — Инфляционные ожидания: https://www.cbr.ru/analytics/dkp/inflationary_expectations/
- Bank of Russia — Инфляция в России: https://www.cbr.ru/analytics/dkp/dinamic/
- Bank of Russia — Обзор рисков финансовых рынков: https://www.cbr.ru/analytics/finstab/orfr/
- Bank of Russia — Денежная масса: https://www.cbr.ru/statistics/ms/
- Rosstat — Доклад "Социально-экономическое положение России": https://www.rosstat.gov.ru/compendium/document/50801
- Минэкономразвития России — прогноз социально-экономического развития: https://www.economy.gov.ru/material/directions/makroec/prognozy_socialno_ekonomicheskogo_razvitiya/
- IMF — World Economic Outlook: https://www.imf.org/en/publications/weo
- IMF — Global Financial Stability Report: https://www.imf.org/en/publications/gfsr
- IMF — Fiscal Monitor: https://www.imf.org/en/publications/fm
- T-Investments research: https://www.tbank.ru/invest/research/
- BCS Express analytics: https://bcs-express.ru/category/analitika
- Cbonds Review: https://review.cbonds.info/

## Time-Series Sources

Direct CSV/XML downloads are saved to `data/raw/time_series/`. They are not included in the text-only RAG index yet, but can be used by optional modules in `src/tools/time_series.py`.

| # | Local file | Source | Original URL |
|---|---|---|---|
| 1 | `data/raw/time_series/fred_gdp.csv` | FRED — GDP | https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP |
| 2 | `data/raw/time_series/fred_cpi_aucs_l.csv` | FRED — CPIAUCSL | https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL |
| 3 | `data/raw/time_series/fred_unrate.csv` | FRED — UNRATE | https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE |
| 4 | `data/raw/time_series/fred_fedfunds.csv` | FRED — FEDFUNDS | https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS |
| 5 | `data/raw/time_series/fred_wti_oil.csv` | FRED — DCOILWTICO | https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO |
| 6 | `data/raw/time_series/fred_regular_gas_price.csv` | FRED — GASREGW | https://fred.stlouisfed.org/graph/fredgraph.csv?id=GASREGW |
| 7 | `data/raw/time_series/fred_m2sl.csv` | FRED — M2SL | https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL |
| 8 | `data/raw/time_series/cbr_exchange_rates_2026_05_31.xml` | Bank of Russia — daily exchange rates | https://www.cbr.ru/scripts/XML_daily.asp?date_req=31/05/2026 |
| 9 | `data/raw/time_series/cbr_usd_rub_2025.xml` | Bank of Russia — USD/RUB dynamics for 2025 | https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2025&date_req2=31/12/2025&VAL_NM_RQ=R01235 |
| 10 | `data/raw/time_series/cbr_miacr_2025.xml` | Bank of Russia — MIACR for 2025 | https://www.cbr.ru/scripts/xml_depo.asp?date_req1=01/01/2025&date_req2=31/12/2025 |
| 11 | `data/raw/time_series/cbr_bank_account_balances_2025.xml` | Bank of Russia — bank account balances for 2025 | https://www.cbr.ru/scripts/XML_ostat.asp?date_req1=01/01/2025&date_req2=31/12/2025 |

Additional time-series catalogs from the source document:

- FRED home: https://fred.stlouisfed.org/
- FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
- Bank of Russia XML interfaces: https://www.cbr.ru/development/SXML/
- Bank of Russia key-rate API description: https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx?op=KeyRateXML
- Fedstat: https://www.fedstat.ru/
- Fedstat indicators: https://www.fedstat.ru/indicator/31074, https://www.fedstat.ru/indicator/31348, https://www.fedstat.ru/indicator/43044, https://www.fedstat.ru/indicator/43060, https://www.fedstat.ru/indicator/43045, https://www.fedstat.ru/indicator/31557
