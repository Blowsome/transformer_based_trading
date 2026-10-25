# transformer_based_trading

A small transformer that reads the last 60 trading days of a Nasdaq stock and predicts its next-day
return. It is trained on 2023–2025, tested on 2026, and compared against an ARIMA baseline, a random guess,
and an "always predict 0" reference. Everything runs on an Apple M1 laptop with 8 GB of RAM.

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

```bash
python -m src.download    # ticker list + daily prices from Yahoo Finance (resumable)
python -m src.prepare     # survivor/liquidity filter, features, compact arrays
python -m src.train       # train the transformer, write 2026 test predictions
python -m src.baselines   # ARIMA, random, zero predictions on the same test rows
python -m src.evaluate    # metrics table + plots in outputs/
```

Long runs can go in the background, with progress checked by `src.status`; an interrupted training run
continues from its last finished epoch with `--resume`:

```bash
nohup caffeinate -i python -u -m src.train > outputs/train.out 2>&1 &
nohup python -u -m src.baselines > outputs/baselines.out 2>&1 &
python -m src.status       # running/stopped, finished epochs with losses, current epoch progress
```

Quick smoke test: `python -m src.download --max-chunks 1` and `python -m src.train --epochs 1 --max-samples 50000`.
All settings (dates, filters, model size, training) are in [config.py](config.py).

## Design

| Piece | Choice |
|---|---|
| Universe | Nasdaq common stocks (no ETFs, warrants, units, rights, preferreds) that traded from the first week of 2023 to the latest day, ≤2% missing days, mean price ≥ $1, median dollar volume ≥ $250k/day |
| Splits (by the day being predicted) | train 2023-01 → 2025-09, validation 2025-10 → 2025-12 (early stopping only), test 2026-01 → today |
| Input | 60 days × 8 features: log return, high/low range, close/open, volume vs 20-day avg, close vs 20-day MA, 5-day return, 20-day volatility, overnight gap. Normalized with training-period stats |
| Target | next-day log return (clipped to the 0.5–99.5 percentile for the training loss only) |
| Model | Linear(8→64) + positional embedding → 3 pre-norm encoder layers (4 heads, FF 128) → last day → Linear(64→1); ~110k parameters |
| Training | AdamW, one-cycle cosine LR, Huber loss, batch 1024, 1M random windows per epoch, early stopping on validation loss |
| Memory | windows are never stored: each batch is gathered from one ~25 MB feature array with a single numpy index |

### Baselines
- **ARIMA(1,0,1)** per stock, fit on that stock's training returns; forecasts for 2026 roll forward with frozen parameters, so every forecast uses only past data.
- **Random**: uniform draw between the 1st and 99th percentile of training daily returns.
- **Zero**: always predict 0. Not a model, but it is the bar that matters for error metrics: daily returns are mostly noise, so predicting 0 is hard to beat on RMSE.

### Metrics (outputs/metrics.csv)
- **RMSE / MAE**: error size.
- **Direction accuracy**: how often the predicted sign is right.
- **IC**: each day, rank correlation between predictions and actual returns across stocks, averaged over days. 0.01–0.03 is already meaningful for daily data.
- **Long-short**: buy the top 10% predicted, short the bottom 10%, daily, equal weight, **no transaction costs** (real costs would reduce this a lot).

## Caveats
- **Survivorship bias**: only stocks that survived through 2026 are used, so the data looks better than the real investable universe did.
- Yahoo Finance data is free and can contain errors; prices are split/dividend adjusted.
- A test result that looks much better than the baselines (e.g. IC > 0.1) usually means a bug or leakage, not a money machine.
