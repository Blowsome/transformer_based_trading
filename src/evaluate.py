"""Step 5: compare all models on the validation and test periods (see config.py), tradable stocks only.

The models output scores for ranking stocks (not return forecasts), so every metric is about ranking:
- IC 1d / IC 5d:   each day, rank correlation between scores and the next-day / next-5-day returns,
                   averaged over days (t-stat: is the average reliably above 0?)
- top-15:          each day buy the 15 highest-scored stocks, sell at the next close, zero trading costs
- top-10% excess:  next-day return of the top-scored 10% minus the average tradable stock

The full backtest with plots and statistical tests is in backtest.ipynb.
"""
import numpy as np
import pandas as pd

import config

MODELS = ["transformer", "lightgbm", "arima", "random"]
TOP_N = 15


def daily_ic(df: pd.DataFrame, target: str) -> pd.Series:
    d = df.dropna(subset=[target])
    return d.groupby("date")[["y_pred", target]].apply(lambda g: g["y_pred"].rank().corr(g[target].rank()))


def metrics(df: pd.DataFrame) -> dict:
    df = df[df["tradable"]].assign(ret=lambda d: np.expm1(d["y_true"]))
    ic1, ic5 = daily_ic(df, "y_true"), daily_ic(df, "y_true_5d")
    top = df.sort_values("y_pred", ascending=False).groupby("date").head(TOP_N).groupby("date")["ret"].mean()
    rank = df.groupby("date")["y_pred"].rank(pct=True)
    excess = df[rank > 0.9].groupby("date")["ret"].mean() - df.groupby("date")["ret"].mean()
    tstat = lambda s: s.mean() / s.std() * np.sqrt(len(s))
    return {
        "IC_1d": ic1.mean(), "IC_1d_tstat": tstat(ic1), "IC_5d": ic5.mean(), "IC_5d_tstat": tstat(ic5),
        "top15_avg_day_%": 100 * top.mean(), "top15_total_%": 100 * ((1 + top).prod() - 1),
        "top15_sharpe": top.mean() / top.std() * np.sqrt(252),
        "top10pct_excess_day_%": 100 * excess.mean(), "top10pct_excess_tstat": tstat(excess),
    }


def main() -> None:
    lines = []
    for split, suffix in [(f"validation ({config.VAL_START} -> {config.VAL_END})", "_val"),
                          (f"test ({config.TEST_START} -> {config.TEST_END})", "")]:
        rows = {m: metrics(pd.read_parquet(config.OUT_DIR / f"preds_{m}{suffix}.parquet"))
                for m in MODELS if (config.OUT_DIR / f"preds_{m}{suffix}.parquet").exists()}
        table = pd.DataFrame(rows).T
        with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200,
                               "display.max_columns", None):
            lines.append(f"\n=== {split} ===\n{table}")
        table.to_csv(config.OUT_DIR / f"metrics{suffix}.csv")
    print("\n".join(lines))
    print(f"\nsaved metrics.csv and metrics_val.csv to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
