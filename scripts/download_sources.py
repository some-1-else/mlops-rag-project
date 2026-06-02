from __future__ import annotations

import argparse
from pathlib import Path

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
TIME_SERIES_DIR = RAW_DIR / "time_series"
TIMEOUT = 60


PDF_SOURCES = [
    {
        "filename": "bank_of_russia_monetary_policy_guidelines_2025_2027.pdf",
        "url": "https://www.cbr.ru/content/document/file/165597/on_eng_2025%282026-2027%29.pdf",
    },
    {
        "filename": "bank_of_russia_monetary_policy_report_2023_04.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/46554/2023_04_ddcp_e.pdf",
    },
    {
        "filename": "bank_of_russia_medium_term_forecast_2024_02_16.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/48892/forecast_240216_e.pdf",
    },
    {
        "filename": "rosstat_social_economic_position_russia_2026_01_03.pdf",
        "url": "https://rosstat.gov.ru/storage/mediabank/osn-03-2026.pdf",
    },
    {
        "filename": "mineconomy_forecast_social_economic_development_rf_2026_2028.pdf",
        "url": "https://www.economy.gov.ru/material/file/download/bc142016f6ab3772370bb0b4541fc778/prognoz_socialno_ekonomicheskogo_razvitiya_rf_2026-2028.pdf",
    },
    {
        "filename": "cbr_regional_economy_2026_04.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/60838/0426.pdf",
    },
    {
        "filename": "cbr_financial_stability_review_2025q4_2026q1.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/60978/4q_2025_1q_2026.pdf",
    },
    {
        "filename": "cbr_inflation_expectations_2026_05.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/61003/Infl_exp_26-05.pdf",
    },
    {
        "filename": "cbr_inflation_russia_2026_04.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/60988/CPD_2026-4.pdf",
    },
    {
        "filename": "cbr_financial_market_risk_review_2026_04.pdf",
        "url": "https://www.cbr.ru/Collection/Collection/File/60961/ORFR_2026-4.pdf",
    },
    {
        "filename": "imf_world_economic_outlook_2026_04.pdf",
        "url": "https://www.imf.org/-/media/files/publications/weo/2026/april/english/text.pdf",
    },
    {
        "filename": "imf_global_financial_stability_report_2026_04.pdf",
        "url": "https://www.imf.org/-/media/files/publications/gfsr/2026/april/english/text.pdf",
    },
    {
        "filename": "imf_fiscal_monitor_2026_04.pdf",
        "url": "https://www.imf.org/-/media/files/publications/fiscal-monitor/2026/april/english/text.pdf",
    },
]


TIME_SERIES_SOURCES = [
    {
        "filename": "fred_gdp.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP",
    },
    {
        "filename": "fred_cpi_aucs_l.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
    },
    {
        "filename": "fred_unrate.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
    },
    {
        "filename": "fred_fedfunds.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
    },
    {
        "filename": "fred_wti_oil.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO",
    },
    {
        "filename": "fred_regular_gas_price.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GASREGW",
    },
    {
        "filename": "fred_m2sl.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL",
    },
    {
        "filename": "cbr_exchange_rates_2026_05_31.xml",
        "url": "https://www.cbr.ru/scripts/XML_daily.asp?date_req=31/05/2026",
    },
    {
        "filename": "cbr_usd_rub_2025.xml",
        "url": "https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2025&date_req2=31/12/2025&VAL_NM_RQ=R01235",
    },
    {
        "filename": "cbr_miacr_2025.xml",
        "url": "https://www.cbr.ru/scripts/xml_depo.asp?date_req1=01/01/2025&date_req2=31/12/2025",
    },
    {
        "filename": "cbr_bank_account_balances_2025.xml",
        "url": "https://www.cbr.ru/scripts/XML_ostat.asp?date_req1=01/01/2025&date_req2=31/12/2025",
    },
]


def download_file(url: str, output_path: Path, force: bool) -> bool:
    if output_path.exists() and not force:
        print(f"Skip existing: {output_path}")
        return False

    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Downloaded: {output_path}")
    return True


def try_download_file(url: str, output_path: Path, force: bool) -> tuple[bool, str | None]:
    try:
        downloaded = download_file(url, output_path, force)
    except requests.RequestException as exc:
        message = f"{output_path.name}: {exc}"
        print(f"Failed: {message}")
        return False, message

    return downloaded, None


def download_sources(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TIME_SERIES_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failures = []

    print(f"PDF sources: {len(PDF_SOURCES)}")
    for source in PDF_SOURCES[:5]:
        output_path = RAW_DIR / source["filename"]
        was_downloaded, failure = try_download_file(source["url"], output_path, force)
        downloaded += int(was_downloaded)
        if failure:
            failures.append(failure)

    print(f"Time-series sources: {len(TIME_SERIES_SOURCES)}")
    for source in TIME_SERIES_SOURCES:
        output_path = TIME_SERIES_DIR / source["filename"]
        was_downloaded, failure = try_download_file(source["url"], output_path, force)
        downloaded += int(was_downloaded)
        if failure:
            failures.append(failure)

    print(f"Downloaded files: {downloaded}")
    print(f"PDF output directory: {RAW_DIR}")
    print(f"Time-series output directory: {TIME_SERIES_DIR}")

    if failures:
        print("Failed downloads:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project source files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()
    download_sources(force=args.force)


if __name__ == "__main__":
    main()
