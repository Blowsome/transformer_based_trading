"""Step 4: baseline predictions on exactly the same test rows as the transformer.

- arima:  ARIMA(1,0,1) per ticker, fit on that ticker's training returns only, then rolled forward
          through 2026 with the parameters frozen (each forecast only uses past returns).
- random: uniform guess between the 1st and 99th percentile of training returns.
- zero:   always predict 0 (reference point; not a model).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process

import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.data import WindowData


def arima_one(args) -> np.ndarray:
    """One-step-ahead forecasts for every return of one ticker."""
    series, n_train = args
    fallback = np.full(len(series), series[:n_train].mean(), dtype=np.float32)
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(series[:n_train], order=config.ARIMA_ORDER, trend="c").fit(disp=False)
            rolled = fit.apply(series)  # same parameters, filtered over the full series
            pred = np.asarray(rolled.predict(), dtype=np.float32)
        return pred if np.isfinite(pred).all() else fallback
    except Exception:
        return fallback


def arima_predictions(data: WindowData) -> np.ndarray:
    n_dates = len(data.dates)
    train_cut = int(np.searchsorted(pd.to_datetime(data.dates), pd.Timestamp(config.TRAIN_END), side="right"))
    jobs, spans = [], []
    for k in range(len(data.tickers)):
        a, b = data.offsets[k], data.offsets[k + 1]
        has_target = data.tdate[a:b] < n_dates
        b = a + int(has_target.sum())  # drop the final row (its next day is in the future)
        series = data.targets[a:b].astype(np.float64)
        n_train = int((data.tdate[a:b] < train_cut).sum())
        jobs.append((series, n_train))
        spans.append((a, b))

    pred_all = np.full(len(data.targets), np.nan, dtype=np.float32)
    with Pool(config.ARIMA_WORKERS) as pool:
        for (a, b), pred in zip(spans, tqdm(pool.imap(arima_one, jobs, chunksize=8),
                                            total=len(jobs), desc="ARIMA")):
            pred_all[a:b] = pred
    return pred_all


def main() -> None:
    data = WindowData()
    ends = data.splits["test"]
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    arima = arima_predictions(data)[ends]
    data.frame(ends, arima).to_parquet(config.OUT_DIR / "preds_arima.parquet", index=False)

    lo, hi = data.meta["random_range"]
    rng = np.random.default_rng(config.SEED)
    data.frame(ends, rng.uniform(lo, hi, size=len(ends))).to_parquet(
        config.OUT_DIR / "preds_random.parquet", index=False)

    data.frame(ends, np.zeros(len(ends))).to_parquet(config.OUT_DIR / "preds_zero.parquet", index=False)
    print(f"saved arima / random (range {lo:.4f}..{hi:.4f}) / zero predictions for {len(ends):,} test rows")


if __name__ == "__main__":
    main()
