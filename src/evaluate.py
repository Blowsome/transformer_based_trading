"""Step 5: compare all models on the 2026 test set.

Metrics
- RMSE / MAE:        size of the error in predicting the next-day log return
- direction acc:     how often sign(prediction) == sign(actual), days with 0 return excluded
- IC (mean, t-stat): each day, Spearman rank correlation between predictions and actual returns
                     across stocks; averaged over days. Measures "can it rank stocks?"
- long-short:        each day buy the top 10% predicted stocks and short the bottom 10%,
                     equal weight, no transaction costs. Annualized return, Sharpe, cumulative return.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

MODELS = ["transformer", "arima", "random", "zero"]


def daily_ic(df: pd.DataFrame) -> pd.Series:
    def ic(g):
        if g["y_pred"].nunique() < 2:  # constant prediction (zero baseline): no ranking
            return np.nan
        return g["y_pred"].rank().corr(g["y_true"].rank())
    return df.groupby("date")[["y_pred", "y_true"]].apply(ic)


def long_short(df: pd.DataFrame, q: float = 0.1) -> pd.Series:
    def day(g):
        if g["y_pred"].nunique() < 2:
            return 0.0
        r = np.expm1(g["y_true"])  # simple returns for portfolio math
        lo, hi = g["y_pred"].quantile([q, 1 - q])
        return r[g["y_pred"] >= hi].mean() - r[g["y_pred"] <= lo].mean()
    return df.groupby("date")[["y_pred", "y_true"]].apply(day)


def metrics(df: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    err = df["y_pred"] - df["y_true"]
    moved = df["y_true"] != 0
    ic = daily_ic(df)
    ls = long_short(df)
    n_ic = ic.notna().sum()
    return {
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAE": err.abs().mean(),
        "direction_acc": (np.sign(df.loc[moved, "y_pred"]) == np.sign(df.loc[moved, "y_true"])).mean(),
        "IC_mean": ic.mean(),
        "IC_tstat": ic.mean() / ic.std() * np.sqrt(n_ic) if n_ic > 1 else np.nan,
        "LS_ann_return": ls.mean() * 252,
        "LS_sharpe": ls.mean() / ls.std() * np.sqrt(252) if ls.std() > 0 else np.nan,
        "LS_cum_return": (1 + ls).prod() - 1,
    }, ic, ls


def main() -> None:
    preds = {m: pd.read_parquet(config.OUT_DIR / f"preds_{m}.parquet")
             for m in MODELS if (config.OUT_DIR / f"preds_{m}.parquet").exists()}

    # evaluate every model on exactly the same (date, ticker) rows
    keys = None
    for df in preds.values():
        k = set(zip(df["date"], df["ticker"]))
        keys = k if keys is None else keys & k
    rows, ics, lss = {}, {}, {}
    for m, df in preds.items():
        df = df[[k in keys for k in zip(df["date"], df["ticker"])]].sort_values(["date", "ticker"])
        rows[m], ics[m], lss[m] = metrics(df)

    first = next(iter(preds.values()))
    print(f"Test set: {len(keys):,} predictions, {first['ticker'].nunique()} stocks, "
          f"{first['date'].min()} -> {first['date'].max()}\n")
    table = pd.DataFrame(rows).T
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200,
                           "display.max_columns", None):
        print(table)
    table.to_csv(config.OUT_DIR / "metrics.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    for m, ls in lss.items():
        if m != "zero":
            ax.plot(pd.to_datetime(ls.index), (1 + ls).cumprod(), label=m)
    ax.axhline(1, color="gray", lw=0.8)
    ax.set_title("Long top-10% / short bottom-10% (no costs), 2026 test period")
    ax.set_ylabel("growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUT_DIR / "equity_curves.png", dpi=120)

    fig, ax = plt.subplots(figsize=(10, 5))
    for m, ic in ics.items():
        if m != "zero":
            ax.plot(pd.to_datetime(ic.index), ic.rolling(20).mean(), label=f"{m} (mean {ic.mean():.4f})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Daily rank IC, 20-day rolling mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUT_DIR / "daily_ic.png", dpi=120)
    print(f"\nsaved metrics.csv, equity_curves.png, daily_ic.png to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
