"""Step 2: keep surviving, liquid stocks, compute features, and save compact arrays.

Output (data/processed/):
    features.npy   float32 [N_rows, N_features]  normalized features, all tickers stacked
    targets.npy    float32 [N_rows]              next-day log return (unclipped)
    tdate.npy      int32   [N_rows]              calendar index of the day being predicted
    offsets.npy    int64   [N_tickers + 1]       row range of each ticker
    dates.npy      str     [N_days]              trading calendar
    tickers.json, meta.json
"""
import json

import numpy as np
import pandas as pd

import config
from src.download import PRICES_FILE


def load_wide() -> dict[str, pd.DataFrame]:
    prices = pd.read_parquet(PRICES_FILE)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()

    # trading calendar: days on which at least half of the tickers traded
    counts = prices.groupby("date")["ticker"].nunique()
    calendar = counts[counts >= 0.5 * counts.max()].index.sort_values()
    prices = prices[prices["date"].isin(calendar)]

    wide = {f: prices.pivot(index="date", columns="ticker", values=f).reindex(calendar)
            for f in ["open", "high", "low", "close", "volume"]}
    return wide


def select_universe(w: dict[str, pd.DataFrame]) -> list[str]:
    close = w["close"]
    first = close.apply(pd.Series.first_valid_index)
    last = close.apply(pd.Series.last_valid_index)
    survived = (first <= pd.Timestamp(config.MAX_FIRST_DATE)) & (last == close.index[-1])

    # missing days between each ticker's first bar and the end of the calendar
    present = close.notna()
    after_first = close.index.values[:, None] >= first.values[None, :]
    missing_frac = ((~present) & after_first).sum() / after_first.sum()
    complete = missing_frac <= config.MAX_MISSING_FRAC

    train = close.index <= pd.Timestamp(config.TRAIN_END)
    mean_close = close[train].mean()
    median_dollar_vol = (close[train] * w["volume"][train]).median()
    liquid = (mean_close >= config.MIN_MEAN_CLOSE) & (median_dollar_vol >= config.MIN_MEDIAN_DOLLAR_VOL)

    keep = survived & complete & liquid
    print(f"tickers: {len(close.columns)} downloaded | survived {survived.sum()} | "
          f"complete {(survived & complete).sum()} | liquid -> kept {keep.sum()}")
    return sorted(keep[keep].index)


def compute_features(w: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    c = w["close"].ffill()
    # a missing day = no trade: open/high/low equal the carried-forward close, zero volume
    o = w["open"].fillna(c)
    h = w["high"].fillna(c)
    l = w["low"].fillna(c)
    v = w["volume"].fillna(0)

    ret = np.log(c / c.shift(1))
    feats = {
        "ret": ret,
        "hl_range": np.log(h / l),
        "co_ret": np.log(c / o),
        "vol_ratio": np.log((v + 1) / (v.rolling(20).mean() + 1)),
        "ma20_gap": c / c.rolling(20).mean() - 1,
        "ret5": np.log(c / c.shift(5)),
        "vol20": ret.rolling(20).std(),
        "gap": np.log(o / c.shift(1)),
    }
    target = ret.shift(-1)  # next-day log return
    return feats, target


def main() -> None:
    wide = load_wide()
    tickers = select_universe(wide)
    wide = {k: df[tickers] for k, df in wide.items()}
    feats, target = compute_features(wide)
    dates = wide["close"].index

    rows_f, rows_y, rows_t, offsets = [], [], [], [0]
    for tk in tickers:
        start = wide["close"][tk].first_valid_index()
        s = dates.get_loc(start) + 1  # +1: the first day has no previous close
        f = np.stack([feats[name][tk].values[s:] for name in config.FEATURES], axis=1)
        y = target[tk].values[s:]
        t = np.arange(s, len(dates)) + 1  # index of the day each row's target belongs to
        rows_f.append(f)
        rows_y.append(y)
        rows_t.append(t)
        offsets.append(offsets[-1] + len(y))

    features = np.concatenate(rows_f).astype(np.float32)
    targets = np.concatenate(rows_y).astype(np.float32)
    tdate = np.concatenate(rows_t).astype(np.int32)
    offsets = np.array(offsets, dtype=np.int64)

    # rows usable by any window: past the warm-up of their ticker (rolling features are defined)
    local = np.arange(len(tdate)) - np.repeat(offsets[:-1], np.diff(offsets))
    usable = local >= config.WARMUP
    assert not np.isnan(features[usable]).any(), "NaN in features after warm-up"

    # normalization and target statistics use training dates only
    n_dates = len(dates)
    train_cut = int(np.searchsorted(dates, pd.Timestamp(config.TRAIN_END), side="right"))
    train_rows = usable & (tdate < train_cut)
    mu = features[train_rows].mean(axis=0)
    sd = features[train_rows].std(axis=0) + 1e-8
    features = np.clip((features - mu) / sd, -config.FEATURE_CLIP, config.FEATURE_CLIP)
    features[~usable] = 0.0  # warm-up rows are never inside a window; zero them to drop NaNs

    y_train = targets[train_rows & (tdate < n_dates)]
    meta = {
        "n_tickers": len(tickers),
        "n_rows": int(len(targets)),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "feature_names": config.FEATURES,
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "target_mean": float(np.mean(y_train)),
        "target_std": float(np.std(y_train)),
        "target_clip": [float(x) for x in np.percentile(y_train, config.TARGET_CLIP_PCT)],
        "random_range": [float(x) for x in np.percentile(y_train, config.RANDOM_RANGE_PCT)],
    }

    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.PROC_DIR / "features.npy", features)
    np.save(config.PROC_DIR / "targets.npy", targets)
    np.save(config.PROC_DIR / "tdate.npy", tdate)
    np.save(config.PROC_DIR / "offsets.npy", offsets)
    np.save(config.PROC_DIR / "dates.npy", np.array([str(d.date()) for d in dates]))
    (config.PROC_DIR / "tickers.json").write_text(json.dumps(tickers))
    (config.PROC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(tickers)} tickers, {len(targets):,} rows, "
          f"{meta['first_date']} -> {meta['last_date']}; features {features.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
