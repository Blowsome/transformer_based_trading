"""Memory-light access to 60-day windows.

Windows are never stored. Each sample is just the row index of its last input day ("end");
a batch is gathered on the fly with one vectorized numpy index: features[end - 59 ... end].
"""
import json

import numpy as np
import pandas as pd

import config


class WindowData:
    def __init__(self):
        p = config.PROC_DIR
        self.features = np.load(p / "features.npy")
        self.targets = np.load(p / "targets.npy")
        self.tdate = np.load(p / "tdate.npy")
        self.offsets = np.load(p / "offsets.npy")
        self.dates = np.load(p / "dates.npy")
        self.tickers = json.loads((p / "tickers.json").read_text())
        self.meta = json.loads((p / "meta.json").read_text())
        self.row_ticker = np.repeat(np.arange(len(self.tickers)), np.diff(self.offsets))
        self._offsets_in_window = np.arange(-config.WINDOW + 1, 1)

        # valid window ends: the whole window lies after the ticker's warm-up, and the target exists
        local = np.arange(len(self.targets)) - self.offsets[self.row_ticker]
        valid = (local >= config.WARMUP + config.WINDOW - 1) & (self.tdate < len(self.dates))
        ends = np.flatnonzero(valid)

        target_day = pd.to_datetime(self.dates[self.tdate[ends]])
        self.splits = {
            "train": ends[target_day <= pd.Timestamp(config.TRAIN_END)],
            "val": ends[(target_day >= pd.Timestamp(config.VAL_START))
                        & (target_day <= pd.Timestamp(config.VAL_END))],
            "test": ends[(target_day >= pd.Timestamp(config.TEST_START))
                         & (target_day <= pd.Timestamp(config.TEST_END))],
        }

    def windows(self, ends: np.ndarray) -> np.ndarray:
        """[B] window ends -> [B, WINDOW, N_features] inputs."""
        return self.features[ends[:, None] + self._offsets_in_window]

    def batches(self, split: str, batch_size: int, shuffle: bool = False,
                max_samples: int | None = None, rng: np.random.Generator | None = None):
        ends = self.splits[split]
        if shuffle:
            ends = rng.permutation(ends)
        if max_samples:
            ends = ends[:max_samples]
        for i in range(0, len(ends), batch_size):
            e = ends[i:i + batch_size]
            yield self.windows(e), self.targets[e], e

    def frame(self, ends: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Prediction table shared by every model: date, ticker, y_true, y_pred."""
        return pd.DataFrame({
            "date": self.dates[self.tdate[ends]],
            "ticker": np.array(self.tickers)[self.row_ticker[ends]],
            "y_true": self.targets[ends],
            "y_pred": np.asarray(y_pred, dtype=np.float32),
        })
