"""Step 4: baseline predictions for exactly the same test days and stocks as the transformer.

- lightgbm: gradient-boosted trees on the decision day's features, same 5-day rank target and early stopping
            on validation IC. A strong, standard baseline the transformer has to beat.
- arima:    ARIMA(1,0,1) per stock on its own daily returns, fit on the training period only, then rolled
            forward with the parameters frozen (each forecast only uses past returns).
- random:   a random number per stock, i.e. random picks.

    python -m src.baselines                 # all three
    python -m src.baselines --only lightgbm
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per ARIMA worker process

import argparse
import warnings
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

import config
from src.data import DayData


def lightgbm_predictions(data: DayData) -> dict[str, pd.DataFrame]:
    X_tr, y_tr, _ = data.last_day_features(data.splits["train"])
    X_va, y_va, va_keys = data.last_day_features(data.splits["val"])
    day_sizes = [len(idx) for _, idx in va_keys]
    print(f"LightGBM: {len(y_tr):,} training rows, {len(y_va):,} validation rows", flush=True)

    def val_ic(pred, _):  # mean daily rank IC on the validation days, the same rule as the transformer
        ics, start = [], 0
        for n in day_sizes:
            p, y = pred[start:start + n], y_va[start:start + n]
            ics.append(np.corrcoef(rankdata(p), y)[0, 1])
            start += n
        return "val_ic", float(np.mean(ics)), True

    names = data.feature_names
    booster = lgb.train(
        dict(config.LGBM_PARAMS, metric="None"),  # early stopping on validation IC only
        lgb.Dataset(X_tr, y_tr, feature_name=names),
        num_boost_round=config.LGBM_ROUNDS,
        valid_sets=[lgb.Dataset(X_va, y_va, feature_name=names)], feval=val_ic,
        callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOP, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"LightGBM: best round {booster.best_iteration}, val IC {booster.best_score['valid_0']['val_ic']:+.4f}")
    importance = pd.Series(booster.feature_importance("gain"), index=names).sort_values(ascending=False)
    print("top features by gain: " + ", ".join(importance.index[:8]))

    out = {}
    for split in ["val", "test"]:
        X, _, keys = data.last_day_features(data.splits[split], rows="all")
        pred = booster.predict(X, num_iteration=booster.best_iteration)
        frames, start = [], 0
        for t, idx in keys:
            frames.append(data.frame(t, idx, pred[start:start + len(idx)]))
            start += len(idx)
        out[split] = pd.concat(frames, ignore_index=True)
    return out


def arima_one(args) -> np.ndarray:
    """One-step-ahead forecasts for every return of one stock."""
    series, n_train = args
    # too little training history (e.g. listed after the training period): no view, predict 0
    fallback = np.full(len(series), series[:n_train].mean() if n_train >= 60 else 0.0, dtype=np.float32)
    if n_train < 60:
        return fallback
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


def arima_predictions(data: DayData) -> np.ndarray:
    """[T, N] forecast made at the close of day t for the return of day t+1."""
    # daily return of day d is ret1[d - 1]; forecasting ret1[t] from ret1[:t] uses returns up to day t only
    T, N = data.ret1.shape
    held = pd.to_datetime(data.dates[np.minimum(np.arange(T) + 1, T - 1)])
    jobs, spans = [], []
    for k in range(N):
        valid = np.flatnonzero(~np.isnan(data.ret1[:, k]))
        a, b = valid[0], valid[-1] + 1
        series = data.ret1[a:b, k].astype(np.float64)
        jobs.append((series, int((held[a:b] <= pd.Timestamp(config.TRAIN_END)).sum())))
        spans.append((a, b))

    pred = np.full((T, N), np.nan, dtype=np.float32)
    with Pool(config.ARIMA_WORKERS) as pool:
        for k, ((a, b), p) in enumerate(zip(spans, tqdm(pool.imap(arima_one, jobs, chunksize=8),
                                                        total=len(jobs), desc="ARIMA", mininterval=5))):
            pred[a:b, k] = p
    return pred


def frames_from_matrix(data: DayData, pred: np.ndarray, split: str) -> pd.DataFrame:
    frames = []
    for t in data.splits[split]:
        idx = np.flatnonzero(~np.isnan(data.ret1[t]))
        frames.append(data.frame(t, idx, pred[t, idx]))
    return pd.concat(frames, ignore_index=True)


def main(only: str | None) -> None:
    data = DayData()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    save = lambda df, name, split: df.to_parquet(
        config.OUT_DIR / f"preds_{name}{'' if split == 'test' else '_val'}.parquet", index=False)

    if only in (None, "lightgbm"):
        for split, df in lightgbm_predictions(data).items():
            save(df, "lightgbm", split)
    if only in (None, "arima"):
        pred = arima_predictions(data)
        for split in ["val", "test"]:
            save(frames_from_matrix(data, pred, split), "arima", split)
    if only in (None, "random"):
        rng = np.random.default_rng(config.SEED)
        lo, hi = data.meta["random_range"]
        pred = rng.uniform(lo, hi, size=data.ret1.shape).astype(np.float32)
        for split in ["val", "test"]:
            save(frames_from_matrix(data, pred, split), "random", split)
    print("saved baseline predictions to", config.OUT_DIR, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["lightgbm", "arima", "random"], default=None)
    main(parser.parse_args().only)
