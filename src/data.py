"""Day-by-day access to the prepared arrays.

A sample is one decision day t: every stock's last WINDOW days of features (up to the close of t).
Windows are never stored; a day's batch is sliced from features[t - WINDOW + 1 : t + 1] when needed.
"""
import json

import numpy as np
import pandas as pd

import config


class DayData:
    def __init__(self):
        p = config.PROC_DIR
        self.features = np.load(p / "features.npy", mmap_mode="r")  # [T, N, F], read from disk on demand
        self.ret1 = np.load(p / "ret1.npy")
        self.ret5 = np.load(p / "ret5.npy")
        self.label = np.load(p / "label.npy")
        self.tradable = np.load(p / "tradable.npy")
        self.dates = np.load(p / "dates.npy")
        self.tickers = np.array(json.loads((p / "tickers.json").read_text()))
        self.meta = json.loads((p / "meta.json").read_text())
        self.feature_names = config.STOCK_FEATURES + (config.MARKET_FEATURES if config.USE_MARKET_FEATURES else [])
        self.n_features = len(self.feature_names)  # market features are the last columns of features.npy

        # decision days t, split by the days whose returns are predicted (t+1 ... t+HORIZON)
        T, H = len(self.dates), config.HORIZON
        d = pd.to_datetime(self.dates)
        t = np.arange(config.WARMUP + config.WINDOW - 1, T - 1)   # windows start after the warm-up
        first = d[t + 1]                                          # first day held
        last = d[np.minimum(t + H, T - 1)]                        # last day of the label (if it exists)
        full = t + H < T
        self.splits = {
            "train": t[full & (last <= pd.Timestamp(config.TRAIN_END))],
            "val": t[full & (first >= pd.Timestamp(config.VAL_START)) & (last <= pd.Timestamp(config.VAL_END))],
            "test": t[(first >= pd.Timestamp(config.TEST_START)) & (first <= pd.Timestamp(config.TEST_END))],
        }

    def day(self, t: int, rows: str = "label") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> inputs [n, WINDOW, F], labels [n], stock indices [n] for decision day t.

        rows="label": tradable stocks with a label (training / validation)
        rows="all":   every stock with a next-day return (predictions for the backtest)
        """
        if rows == "label":
            idx = np.flatnonzero(self.tradable[t] & ~np.isnan(self.label[t]))
        else:
            idx = np.flatnonzero(~np.isnan(self.ret1[t]))
        x = np.ascontiguousarray(
            self.features[t - config.WINDOW + 1:t + 1, idx, :self.n_features].transpose(1, 0, 2))
        return x, self.label[t, idx], idx

    def last_day_features(self, days: np.ndarray, rows: str = "label") -> tuple[np.ndarray, np.ndarray, list]:
        """Features of the decision day only (for LightGBM): X [rows, F], labels, [(t, idx), ...]."""
        xs, ys, keys = [], [], []
        for t in days:
            x, y, idx = self.day(t, rows)
            xs.append(x[:, -1])
            ys.append(y)
            keys.append((t, idx))
        return np.concatenate(xs), np.concatenate(ys), keys

    def frame(self, t: int, idx: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Prediction table shared by every model; `date` is the first day the stock is held (t+1)."""
        return pd.DataFrame({
            "date": self.dates[t + 1],
            "ticker": self.tickers[idx],
            "y_true": self.ret1[t, idx],        # next-day log return
            "y_true_5d": self.ret5[t, idx],     # next HORIZON-day log return (NaN near the end of the data)
            "tradable": self.tradable[t, idx],
            "y_pred": np.asarray(y_pred, dtype=np.float32),
        })
