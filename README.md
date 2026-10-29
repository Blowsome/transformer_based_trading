# transformer_based_trading

A small transformer that ranks US stocks by how well they are expected to do over the next 5 trading days,
compared with LightGBM, ARIMA, random picks and ensembles in buy-the-top-stocks backtests. Trained on
2015–2024, validated on 2025, tested on 2026. Everything runs on an Apple M1 laptop with 8 GB of RAM.

## Setup (Apple silicon)

The model trains on the M1 GPU (PyTorch `mps`), which requires a **native arm64** Python.
An Intel (x86_64) Miniconda running under Rosetta cannot use the GPU.

```bash
CONDA_SUBDIR=osx-arm64 conda create -n stockformer python=3.11 -y
conda activate stockformer
conda config --env --set subdir osx-arm64
pip install -r requirements.txt
python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"  # arm64 True
```

## Run

Everything, in the background (several hours, mostly training; run it overnight):

```bash
conda activate stockformer
nohup caffeinate -i scripts/run_pipeline.sh > outputs/logs/pipeline.out 2>&1 &
python -m src.status       # step list, training model/epoch/step + ETA, validation IC per epoch, guard
```

The pipeline: download → prepare → two one-seed regularization experiments (judged on 2025 validation IC)
→ final 3-seed ensemble with the better settings → baselines → evaluate → notebook.

**Safety.** A guard ([src/guard.py](src/guard.py)) runs alongside and stops the pipeline cleanly if macOS memory
pressure stays critical for 45 s or free disk drops below 3 GB (reason in `outputs/logs/guard.out`).
`caffeinate` keeps the Mac awake, `nohup` survives closing the terminal, and training checkpoints every epoch.
After a stop, a crash or a reboot, continue with
`RESUME=1 nohup caffeinate -i scripts/run_pipeline.sh > outputs/logs/pipeline.out 2>&1 &`:
finished steps are skipped and training continues from its last finished epoch.

Or step by step:

```bash
python -m src.download    # ticker lists + daily prices from Yahoo Finance (resumable)
python -m src.prepare     # universe filters, bad-data filter, features, 5-day rank target
python -m src.train --experiment reg_current   # one seed per regularization option, validation only
python -m src.train --experiment reg_strong
python -m src.train --settings best             # ensemble with the better option; validation + test predictions
python -m src.baselines   # LightGBM, ARIMA, random predictions for the same days and stocks
python -m src.evaluate    # ranking metrics for validation and test -> outputs/metrics*.csv
jupyter nbconvert --to notebook --execute --inplace backtest.ipynb   # backtest, plots, statistical tests
```

An interrupted training run continues from its last finished epoch with `python -m src.train --resume`.
Do not run `src.prepare` while training is running: training reads the files it rewrites.
Smoke test: `python -m src.train --seeds 1 --epochs 1 --max-days 40`.
All settings are in [config.py](config.py).

## Design

| Piece | Choice |
|---|---|
| Universe | US common stocks listed on Nasdaq, NYSE, NYSE American, NYSE Arca, Cboe BZX and IEX (no ETFs, funds, warrants, units, rights, preferreds) that are still listed today, ≤2% missing days after their first day. A stock can list at any time and is used once it has 80 trading days of history. Stocks with obviously bad Yahoo bars (a ~2x jump that reverses the next day, or a >10x day) are dropped |
| Tradable on a day | close ≥ $5, 20-day median dollar volume ≥ $1M, and ≥ 80 days of history. Only tradable stocks are used for the training labels and in the backtest |
| Target | each day, the tradable stocks ranked by their next-5-day return, scaled to −0.5…0.5. Removes the market's daily up/down moves and extreme outliers; the model learns which stocks beat the others |
| Splits (by the days whose returns are predicted) | train 2015 → 2024, validation 2025 (early stopping, model choices), test 2026-01 → today. A training day is kept only if its whole 5-day label ends by 2024-12-31 |
| Input | 20 days × 23 stock features (returns over 1/5/10/20/60 days, volatility over 5/20/60 days, high/low range, intraday and overnight moves, volume trends, distance from moving averages and from 20/60-day highs and lows, largest daily moves, dollar volume), each ranked across stocks every day. 5 market features (equal-weight market return, volatility, dispersion) are prepared but off by default: they let models memorize individual days |
| Model | Linear(23→64) + positional embedding → 2 pre-norm encoder layers along time (4 heads, FF 128, dropout 0.2) → last day → 1 attention layer across all stocks of the day → score |
| Training | one batch = a random 1,500 of the tradable stocks of one day (a new subset every epoch); loss = −(daily correlation of scores with the target) + 0.1 × MSE; AdamW, one-cycle cosine LR; early stopping on validation rank IC; dropout / weight decay picked from two options on validation; ensemble of 3 seeds (scores z-scored per day, then averaged) |
| Memory | features are one dense day × stock × feature array (~1.5–2 GB) written and read on disk; windows are sliced per day, never stored |

### Baselines
- **LightGBM**: gradient-boosted trees on the decision day's 23 features, same target, early stopping on validation IC.
- **ARIMA(1,0,1)** per stock on its own daily returns, fit on the training period; forecasts roll forward with frozen parameters, so every forecast uses only past data.
- **Random**: a random score per stock, i.e. random picks.

### Evaluation
- [src/evaluate.py](src/evaluate.py): daily rank IC against next-day and next-5-day returns, top-15 daily return, and top-10% excess return, for validation and test.
- [backtest.ipynb](backtest.ipynb): $10,000 backtests of the daily top 15 and of top 15 held 5 days (trading cost set by `COST_BPS`, default 0), daily-return histograms, comparison with 1,000 random portfolios, paired t-tests and Wilcoxon tests, and return by score decile.
- Earlier results are kept in `outputs/v1_nasdaq/` (Nasdaq only, next-day return target) and `outputs/v2_2023/` (all exchanges, 2023–2025 training).

## Caveats
- **Survivorship bias**: only stocks that survived through 2026 are used, so the data looks better than the real investable universe did.
- **2026 has been looked at before** (earlier versions were evaluated on it), so it is not a perfectly clean test. Paper trading after the data ends is the cleanest check.
- Yahoo Finance data is free and can contain errors; prices are split/dividend adjusted.
- A test result that looks much better than the baselines (e.g. IC > 0.1) usually means a bug or leakage, not a money machine.
