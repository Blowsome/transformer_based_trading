"""Step 1: download the US common-stock lists and daily prices from Yahoo Finance.

Two ticker lists from nasdaqtrader.com: Nasdaq-listed, and everything else (NYSE, NYSE American, ...).
Prices are saved chunk by chunk, so rerunning after an interruption resumes where it stopped.

    python -m src.download                 # everything
    python -m src.download --max-chunks 1  # first 100 tickers of each list only (smoke test)
"""
import argparse
import io
import time

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

import config

PRICES_FILE = config.RAW_DIR / "prices.parquet"
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
SYMBOL_COLUMN = {"nasdaq": "Symbol", "other": "ACT Symbol"}


def get_tickers(source: str) -> list[str]:
    """Common stocks of one list (no ETFs, test issues, warrants, units, rights, preferreds, funds)."""
    path = config.RAW_DIR / f"tickers_{source}.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)["ticker"].tolist()

    text = requests.get(config.TICKER_SOURCES[source], timeout=30).text
    # keep_default_na=False: real tickers such as "NA" must not become NaN
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    sym = SYMBOL_COLUMN[source]
    df = df[~df[sym].str.startswith("File Creation Time")]
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    df = df[~df["Security Name"].str.contains(config.EXCLUDE_NAME_PATTERN, case=False, regex=True)]
    df = df[~df[sym].str.contains(config.EXCLUDE_SYMBOL_PATTERN, regex=True)]
    tickers = sorted(df[sym].str.replace(".", "-", regex=False).unique())  # Yahoo writes BRK.B as BRK-B

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    return tickers


def download_chunk(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers, start=config.START, end=None, auto_adjust=True, group_by="ticker",
        threads=True, progress=False,
    )
    if raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker comes back flat
        raw.columns = pd.MultiIndex.from_product([tickers, raw.columns])
    long = raw.stack(level=0, future_stack=True)[FIELDS].dropna(subset=["Close"])
    long.index.names = ["date", "ticker"]
    long.columns = [c.lower() for c in long.columns]
    return long.reset_index().astype({c.lower(): "float32" for c in FIELDS})


def main(max_chunks: int | None) -> None:
    jobs = []
    for source in config.TICKER_SOURCES:
        tickers = get_tickers(source)
        chunks = [tickers[i:i + config.CHUNK_SIZE] for i in range(0, len(tickers), config.CHUNK_SIZE)]
        if max_chunks:
            chunks = chunks[:max_chunks]
        print(f"{source}: {len(tickers)} tickers, {len(chunks)} chunks")
        jobs += [(config.RAW_DIR / f"chunk_{source}_{i:03d}.parquet", chunk) for i, chunk in enumerate(chunks)]

    for path, chunk in tqdm(jobs, desc="chunks"):
        if path.exists():
            continue
        for attempt in range(2):
            try:
                df = download_chunk(chunk)
                if not df.empty:
                    df.to_parquet(path, index=False)
                break
            except Exception as e:  # network / rate limit: wait and retry once
                print(f"{path.name} attempt {attempt + 1} failed: {e}")
                time.sleep(10)
        time.sleep(config.SLEEP_BETWEEN_CHUNKS)

    parts = sorted(config.RAW_DIR.glob("chunk_*.parquet"))
    prices = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    prices = prices.drop_duplicates(["date", "ticker"])  # a few tickers could appear in both lists
    prices.to_parquet(PRICES_FILE, index=False)
    print(f"saved {PRICES_FILE}: {len(prices):,} rows, {prices['ticker'].nunique()} tickers, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=None)
    main(parser.parse_args().max_chunks)
