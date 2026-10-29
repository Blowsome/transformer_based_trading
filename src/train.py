"""Step 3: train an ensemble of transformers (one per seed), then predict the validation and test days.

    python -m src.train                                       # N_SEEDS models from scratch, config settings
    python -m src.train --experiment reg_strong               # 1 model with one REGULARIZATION_OPTIONS entry,
                                                              # validation only -> outputs/experiments/reg_strong/
    python -m src.train --settings best                       # ensemble with the experiment that had the best
                                                              # validation IC
    python -m src.train --resume                              # continue an interrupted run (same other flags)
    python -m src.train --seeds 1 --epochs 1 --max-days 40    # smoke test

Progress is written to train_progress.json in the run's folder; check it with `python -m src.status`.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

import config
from src.data import DayData
from src.model import StockTransformer

EXPERIMENT_DIR = config.OUT_DIR / "experiments"


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Progress:
    """Small JSON file that src.status reads to show where training is."""

    def __init__(self, path: Path, name: str, n_seeds: int, epochs: int, steps: int):
        self.path = path
        self.state = {"pid": os.getpid(), "name": name, "started": time.time(), "phase": "training",
                      "n_seeds": n_seeds, "epochs": epochs, "steps": steps, "history": {}}

    def update(self, **kw) -> None:
        self.state.update(kw, updated=time.time())
        self.path.write_text(json.dumps(self.state))


def loss_fn(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Loss for one day of stocks; y is the 5-day return rank (-0.5..0.5)."""
    if config.LOSS == "mse":
        return F.mse_loss(pred, y)
    if config.LOSS == "listnet":  # cross-entropy between the true and the predicted "top-heaviness"
        return -(torch.softmax(y / config.LISTNET_TEMP, 0) * torch.log_softmax(pred, 0)).sum()
    p, q = pred - pred.mean(), y - y.mean()
    corr = (p * q).sum() / (p.norm() * q.norm() + 1e-8)
    return -corr + config.MSE_WEIGHT * F.mse_loss(pred, y)


@torch.no_grad()
def predict_days(model, data: DayData, days, device, rows: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    model.eval()
    out = []
    for t in days:
        x, _, idx = data.day(t, rows)
        out.append((t, idx, model(torch.from_numpy(x).to(device)).cpu().numpy()))
    return out


def daily_ic(data: DayData, preds) -> float:
    """Mean over days of the rank correlation between predictions and the 5-day return rank (tradable stocks)."""
    ics = []
    for t, idx, p in preds:
        y = data.label[t, idx]
        ok = ~np.isnan(y)
        ics.append(np.corrcoef(rankdata(p[ok]), y[ok])[0, 1])
    return float(np.mean(ics))


def train_seed(seed: int, k: int, data: DayData, device, epochs: int, max_days: int | None,
               resume: bool, progress: Progress, model_dir: Path) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_days, val_days = data.splits["train"], data.splits["val"]
    if max_days:
        train_days, val_days = train_days[-max_days:], val_days[:max_days]

    model = StockTransformer(data.n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.LR, total_steps=epochs * len(train_days),
                                                pct_start=0.05, anneal_strategy="cos")
    ckpt_path, best_path = model_dir / f"checkpoint_seed{seed}.pt", model_dir / f"seed{seed}.pt"
    state = {"best_ic": -math.inf, "bad_epochs": 0, "log": [], "epoch": 0, "done": False}
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        state = ckpt["state"]
        print(f"seed {seed}: resumed after epoch {state['epoch']} (best val IC {state['best_ic']:.4f})")

    for epoch in range(state["epoch"] + 1, epochs + 1):
        if state["done"]:
            break
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(train_days)
        for step, t in enumerate(order, 1):
            x, y, _ = data.day(t)
            if config.TRAIN_STOCKS_PER_DAY and len(y) > config.TRAIN_STOCKS_PER_DAY:
                keep = rng.choice(len(y), config.TRAIN_STOCKS_PER_DAY, replace=False)
                x, y = x[keep], y[keep]
            loss = loss_fn(model(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            sched.step()
            total += loss.item()
            if step % config.LOG_EVERY == 0 or step == len(order):
                progress.update(seed=seed, seed_number=k, epoch=epoch, step=step, steps=len(order),
                                train_loss=total / step, epoch_started=t0)

        progress.update(phase="validating")
        val_ic = daily_ic(data, predict_days(model, data, val_days, device, "label"))
        if val_ic > state["best_ic"]:
            state["best_ic"], state["bad_epochs"] = val_ic, 0
            torch.save(model.state_dict(), best_path)
        else:
            state["bad_epochs"] += 1
        state["log"].append({"seed": seed, "epoch": epoch, "train_loss": total / len(order), "val_ic": val_ic,
                             "seconds": time.time() - t0})
        state["epoch"] = epoch
        state["done"] = state["bad_epochs"] >= config.PATIENCE or epoch == epochs
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "state": state},
                   ckpt_path)
        print(f"seed {seed} epoch {epoch:2d} | train loss {total / len(order):+.4f} | val IC {val_ic:+.4f}"
              f"{' *best' if state['bad_epochs'] == 0 else ''} | {time.time() - t0:.0f}s", flush=True)
        progress.state["history"][str(seed)] = state["log"]
        progress.update(phase="training")
        if state["bad_epochs"] >= config.PATIENCE:
            print(f"seed {seed}: early stopping, best val IC {state['best_ic']:+.4f}", flush=True)
    state["done"] = True
    return state


def zscore(p: np.ndarray) -> np.ndarray:
    return (p - p.mean()) / (p.std() + 1e-8)


def best_experiment() -> str | None:
    """Name of the finished experiment with the highest validation IC (None if there is none)."""
    results = {p.parent.name: json.loads(p.read_text())["val_ic"] for p in EXPERIMENT_DIR.glob("*/summary.json")}
    if results:
        print("experiments (val IC): " + ", ".join(f"{k} {v:+.4f}" for k, v in results.items()), flush=True)
    return max(results, key=results.get) if results else None


def main(n_seeds: int, epochs: int, max_days: int | None, resume: bool,
         experiment: str | None, settings: str | None) -> None:
    # which settings, and where this run's files go
    name = experiment or (best_experiment() if settings == "best" else settings)
    if name:
        opts = config.REGULARIZATION_OPTIONS[name]
        config.DROPOUT, config.WEIGHT_DECAY = opts["dropout"], opts["weight_decay"]
    run_dir = EXPERIMENT_DIR / experiment if experiment else config.OUT_DIR
    model_dir = run_dir / "models"
    splits = ["val"] if experiment else ["val", "test"]  # experiments never look at the test period

    device = get_device()
    data = DayData()
    print(f"device={device} | {len(data.tickers)} stocks, {data.n_features} features | days: "
          + ", ".join(f"{k} {len(v)}" for k, v in data.splits.items())
          + f" | settings: {name or 'config'} (dropout {config.DROPOUT}, weight decay {config.WEIGHT_DECAY})",
          flush=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for f in model_dir.glob("*seed*.pt"):
            f.unlink()

    steps = min(max_days or len(data.splits["train"]), len(data.splits["train"]))
    progress = Progress(run_dir / "train_progress.json", experiment or "final ensemble", n_seeds, epochs, steps)
    progress.update(seed=config.SEED, seed_number=1, epoch=1, step=0, steps=steps, epoch_started=time.time())
    seeds = [config.SEED + i for i in range(n_seeds)]
    for k, seed in enumerate(seeds, 1):
        state = train_seed(seed, k, data, device, epochs, max_days, resume, progress, model_dir)
        progress.state["history"][str(seed)] = state["log"]

    # ensemble: z-score each model's predictions within the day, then average
    progress.update(phase="predicting")
    model = StockTransformer(data.n_features).to(device)
    for split in splits:
        days = data.splits[split][:max_days] if max_days else data.splits[split]
        per_seed = []
        for seed in seeds:
            model.load_state_dict(torch.load(model_dir / f"seed{seed}.pt", map_location=device))
            per_seed.append(predict_days(model, data, days, device, "all"))
        ensemble = [(t, idx, np.mean([zscore(s[i][2]) for s in per_seed], axis=0))
                    for i, (t, idx, _) in enumerate(per_seed[0])]
        out = pd.concat([data.frame(t, idx, p) for t, idx, p in ensemble], ignore_index=True)
        suffix = "" if split == "test" else "_val"
        out.to_parquet(run_dir / f"preds_transformer{suffix}.parquet", index=False)
        if split == "val":
            ics = [daily_ic(data, s) for s in per_seed]
            val_ic = daily_ic(data, ensemble)
            print("val IC per model: " + ", ".join(f"{ic:+.4f}" for ic in ics) + f" | ensemble: {val_ic:+.4f}",
                  flush=True)
            (run_dir / "summary.json").write_text(json.dumps({
                "settings": name or "config", "dropout": config.DROPOUT, "weight_decay": config.WEIGHT_DECAY,
                "val_ic": val_ic, "val_ic_per_model": ics}, indent=2))
        if split == "test":
            print(f"saved {len(out):,} test predictions ({out['date'].min()} -> {out['date'].max()})", flush=True)
    pd.DataFrame([row for log in progress.state["history"].values() for row in log]) \
        .to_csv(run_dir / "train_log.csv", index=False)
    progress.update(phase="done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=None, help=f"default {config.N_SEEDS}, 1 for --experiment")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from the run's checkpoint_seed*.pt")
    parser.add_argument("--experiment", choices=list(config.REGULARIZATION_OPTIONS), default=None)
    parser.add_argument("--settings", choices=["best", *config.REGULARIZATION_OPTIONS], default=None)
    args = parser.parse_args()
    seeds = args.seeds or (1 if args.experiment else config.N_SEEDS)
    main(seeds, args.epochs, args.max_days, args.resume, args.experiment, args.settings)
