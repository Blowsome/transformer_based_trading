"""Step 2: keep listed, complete, clean stocks; compute features and targets; save dense day x stock arrays.

Output (data/processed/), T = trading days, N = stocks, F = features:
    features.npy   float32 [T, N, F]  stock features ranked across stocks each day (-0.5..0.5),
                                      then market features (z-scored); known at the close of day t
    ret1.npy       float32 [T, N]     log return from close t to close t+1 (the next day held)
    ret5.npy       float32 [T, N]     log return from close t to close t+HORIZON
    label.npy      float32 [T, N]     ret5 ranked among the stocks tradable on day t (-0.5..0.5), else NaN
    tradable.npy   bool    [T, N]     price, liquidity and history filter on day t
    dates.npy      str     [T]        trading calendar
    tickers.json, meta.json

features.npy is written to disk one feature at a time, so peak memory stays a few GB for 10+ years of data.
"""
import json
from datetime import date

import numpy as np
import pandas as pd

import config
from src.download import PRICES_FILE


def load_wide() -> dict[str, pd.DataFrame]:
    prices = pd.read_parquet(PRICES_FILE)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    # only completed trading days: during market hours Yahoo includes today's unfinished bar
    prices = prices[prices["date"] < pd.Timestamp(date.today())]

    # trading calendar: days on which at least half as many tickers traded as on nearby days
    # (compared with nearby days, because far fewer of today's stocks existed in 2015)
    counts = prices.groupby("date")["ticker"].nunique().sort_index()
    typical = counts.rolling(21, center=True, min_periods=1).median()
    calendar = counts[counts >= 0.5 * typical].index
    prices = prices[prices["date"].isin(calendar)]

    return {f: prices.pivot(index="date", columns="ticker", values=f).reindex(calendar)
            for f in ["open", "high", "low", "close", "volume"]}


def tradable_mask(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    dollar_vol = (close * volume).rolling(20, min_periods=15).median()
    age = close.notna().cumsum()  # trading days of history
    return ((close >= config.MIN_PRICE) & (dollar_vol >= config.MIN_DOLLAR_VOL)
            & (age >= config.WARMUP + config.WINDOW))


def select_universe(w: dict[str, pd.DataFrame]) -> list[str]:
    close = w["close"]
    first = close.apply(pd.Series.first_valid_index)
    last = close.apply(pd.Series.last_valid_index)
    listed = last == close.index[-1]

    # missing days between each ticker's first bar and the end of the calendar
    present = close.notna()
    after_first = close.index.values[:, None] >= first.values[None, :]
    missing_frac = ((~present) & after_first).sum() / after_first.sum()
    complete = missing_frac <= config.MAX_MISSING_FRAC

    # bad Yahoo bars: a big jump that reverses the next day (e.g. $0.33 -> $17 -> $0.33), or a >10x day
    r = np.log(close.ffill() / close.ffill().shift(1))
    nxt = r.shift(-1)
    spike = (r.abs() > config.SPIKE_LOG_RET) & (nxt.abs() > config.SPIKE_LOG_RET) & (np.sign(r) != np.sign(nxt))
    clean = ~(spike.any() | (r.abs() > config.MAX_LOG_RET).any())

    # stocks that are never tradable never enter a label or a backtest; drop them to save memory
    ever_tradable = tradable_mask(close, w["volume"]).any()

    keep = listed & complete & clean & ever_tradable
    print(f"tickers: {len(close.columns)} downloaded | still listed {listed.sum()} | "
          f"complete {(listed & complete).sum()} | clean data {(listed & complete & clean).sum()} | "
          f"ever tradable -> kept {keep.sum()}", flush=True)
    return sorted(keep[keep].index)


def stock_features(w: dict[str, pd.DataFrame]) -> dict:
    """name -> function computing that feature as a [T, N] DataFrame (one at a time, to save memory)."""
    c = w["close"].ffill()
    # a missing day = no trade: open/high/low equal the carried-forward close, zero volume
    o = w["open"].fillna(c)
    h = w["high"].fillna(c)
    l = w["low"].fillna(c)
    v = w["volume"].fillna(0)
    r = np.log(c / c.shift(1))
    logret = lambda k: np.log(c / c.shift(k))
    vol_mean = lambda k: v.rolling(k).mean() + 1
    return {
        "ret1": lambda: r, "ret5": lambda: logret(5), "ret10": lambda: logret(10),
        "ret20": lambda: logret(20), "ret60": lambda: logret(60),
        "vol5": lambda: r.rolling(5).std(), "vol20": lambda: r.rolling(20).std(),
        "vol60": lambda: r.rolling(60).std(),
        "hl_range": lambda: np.log(h / l), "co_ret": lambda: np.log(c / o), "gap": lambda: np.log(o / c.shift(1)),
        "vol_ratio5": lambda: np.log(vol_mean(5) / vol_mean(60)),
        "vol_ratio20": lambda: np.log(vol_mean(20) / vol_mean(60)),
        "ma5_gap": lambda: c / c.rolling(5).mean() - 1, "ma20_gap": lambda: c / c.rolling(20).mean() - 1,
        "ma60_gap": lambda: c / c.rolling(60).mean() - 1,
        "high20": lambda: np.log(c / h.rolling(20).max()), "low20": lambda: np.log(c / l.rolling(20).min()),
        "high60": lambda: np.log(c / h.rolling(60).max()), "low60": lambda: np.log(c / l.rolling(60).min()),
        "max_ret20": lambda: r.rolling(20).max(), "min_ret20": lambda: r.rolling(20).min(),
        "dollar_vol": lambda: np.log((c * v).rolling(20).median() + 1),
    }


def market_features(r: pd.DataFrame) -> dict[str, pd.Series]:
    mkt = r.mean(axis=1)  # equal-weight market of the listed stocks
    return {
        "mkt_ret1": mkt, "mkt_ret5": mkt.rolling(5).sum(), "mkt_ret20": mkt.rolling(20).sum(),
        "mkt_vol20": mkt.rolling(20).std(), "mkt_dispersion": r.std(axis=1).rolling(20).mean(),
    }


def cs_rank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Rank across stocks each day, scaled to -0.5..0.5 (NaN where the value is missing or the mask is off)."""
    return df.where(mask).rank(axis=1, pct=True) - 0.5


def main() -> None:
    wide = load_wide()
    tickers = select_universe(wide)
    wide = {k: df[tickers] for k, df in wide.items()}
    dates = wide["close"].index
    T, N = len(dates), len(tickers)
    train = dates <= pd.Timestamp(config.TRAIN_END)

    c = wide["close"].ffill()
    listed = c.notna()  # from the first trading day on
    ret1 = np.log(c.shift(-1) / c)
    ret5 = np.log(c.shift(-config.HORIZON) / c)
    tradable = tradable_mask(wide["close"], wide["volume"])
    label = cs_rank(ret5, tradable)

    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    F = len(config.STOCK_FEATURES) + len(config.MARKET_FEATURES)
    features = np.lib.format.open_memmap(config.PROC_DIR / "features.npy", mode="w+", dtype=np.float32,
                                         shape=(T, N, F))
    make = stock_features(wide)
    for j, name in enumerate(config.STOCK_FEATURES):
        features[:, :, j] = cs_rank(make.pop(name)(), listed).fillna(0).to_numpy(np.float32)
        print(f"  feature {j + 1}/{F}: {name}", flush=True)
    mkt = market_features(np.log(c / c.shift(1)))
    mkt_stats = {}
    for j, name in enumerate(config.MARKET_FEATURES, start=len(config.STOCK_FEATURES)):
        s = mkt[name]
        mu, sd = s[train].mean(), s[train].std()
        features[:, :, j] = ((s - mu) / sd).clip(-config.FEATURE_CLIP, config.FEATURE_CLIP).fillna(0) \
            .to_numpy(np.float32)[:, None]
        mkt_stats[name] = [float(mu), float(sd)]
    features.flush()
    del features

    y_train = ret1.where(tradable)[train].to_numpy().ravel()  # NaN-aware percentiles below
    meta = {
        "n_tickers": N, "n_days": T,
        "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
        "feature_names": config.STOCK_FEATURES + config.MARKET_FEATURES,
        "market_feature_stats": mkt_stats,
        "tradable_per_day": float(tradable.sum(axis=1).mean()),
        "tradable_per_day_by_year": {str(y): float(v) for y, v in
                                     tradable.sum(axis=1).groupby(dates.year).mean().items()},
        "random_range": [float(x) for x in np.nanpercentile(y_train, (1, 99))],
    }
    for name, df in [("ret1", ret1), ("ret5", ret5), ("label", label)]:
        np.save(config.PROC_DIR / f"{name}.npy", df.to_numpy(np.float32))
    np.save(config.PROC_DIR / "tradable.npy", tradable.to_numpy(bool))
    np.save(config.PROC_DIR / "dates.npy", np.array([str(d.date()) for d in dates]))
    (config.PROC_DIR / "tickers.json").write_text(json.dumps(tickers))
    (config.PROC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {N} stocks x {T} days x {F} features ({T * N * F * 4 / 1e9:.1f} GB), "
          f"{meta['first_date']} -> {meta['last_date']}; {meta['tradable_per_day']:.0f} tradable per day", flush=True)
    print("tradable per day by year: " + ", ".join(f"{y} {v:.0f}" for y, v in
                                                   meta["tradable_per_day_by_year"].items()), flush=True)


if __name__ == "__main__":
    main()
