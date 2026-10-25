"""Step 3: train the transformer, then predict the 2026 test set.

    python -m src.train
    python -m src.train --epochs 1 --max-samples 50000   # smoke test
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch

import config
from src.data import WindowData
from src.model import StockTransformer


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model, data: WindowData, split: str, device, max_samples=None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (window ends, predicted next-day log return)."""
    model.eval()
    mean, std = data.meta["target_mean"], data.meta["target_std"]
    ends, preds = [], []
    for x, _, e in data.batches(split, 4096, max_samples=max_samples):
        out = model(torch.from_numpy(x).to(device)).cpu().numpy()
        preds.append(out * std + mean)
        ends.append(e)
    return np.concatenate(ends), np.concatenate(preds)


def evaluate_loss(model, data, split, device, loss_fn, max_samples=None) -> float:
    model.eval()
    mean, std = data.meta["target_mean"], data.meta["target_std"]
    lo, hi = data.meta["target_clip"]
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y, _ in data.batches(split, 4096, max_samples=max_samples):
            y = (np.clip(y, lo, hi) - mean) / std
            out = model(torch.from_numpy(x).to(device))
            total += loss_fn(out, torch.from_numpy(y).to(device)).item() * len(y)
            n += len(y)
    return total / n


def main(epochs: int, max_samples: int | None, resume: bool = False) -> None:
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    device = get_device()
    data = WindowData()
    mean, std = data.meta["target_mean"], data.meta["target_std"]
    lo, hi = data.meta["target_clip"]
    print(f"device={device} | windows: " + ", ".join(f"{k} {len(v):,}" for k, v in data.splits.items()))

    model = StockTransformer(n_features=data.features.shape[1]).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    per_epoch = min(max_samples or config.SAMPLES_PER_EPOCH, len(data.splits["train"]))
    steps_total = epochs * math.ceil(per_epoch / config.BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.LR, total_steps=steps_total,
                                                pct_start=0.05, anneal_strategy="cos")
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    val_cap = max_samples  # in smoke tests, keep validation small too

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.OUT_DIR / "checkpoint.pt"
    best_val, bad_epochs, log, start = float("inf"), 0, [], 1
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        best_val, bad_epochs, log = ckpt["best_val"], ckpt["bad_epochs"], ckpt["log"]
        start = ckpt["epoch"] + 1
        print(f"resumed after epoch {ckpt['epoch']} (best val {best_val:.5f})")

    for epoch in range(start, epochs + 1):
        if bad_epochs >= config.PATIENCE:
            break
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, _ in data.batches("train", config.BATCH_SIZE, shuffle=True,
                                    max_samples=per_epoch, rng=rng):
            y = (np.clip(y, lo, hi) - mean) / std
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)

        val = evaluate_loss(model, data, "val", device, loss_fn, max_samples=val_cap)
        log.append({"epoch": epoch, "train_loss": total / n, "val_loss": val, "seconds": time.time() - t0})
        print(f"epoch {epoch:2d} | train {total / n:.5f} | val {val:.5f} | {time.time() - t0:.0f}s")
        pd.DataFrame(log).to_csv(config.OUT_DIR / "train_log.csv", index=False)

        if val < best_val:
            best_val, bad_epochs = val, 0
            torch.save(model.state_dict(), config.OUT_DIR / "model.pt")
        else:
            bad_epochs += 1
        # full training state, so an interrupted run can continue with --resume
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "epoch": epoch, "best_val": best_val, "bad_epochs": bad_epochs, "log": log},
                   ckpt_path)
        if bad_epochs >= config.PATIENCE:
            print("early stopping")
            break

    model.load_state_dict(torch.load(config.OUT_DIR / "model.pt", map_location=device))
    ends, preds = predict(model, data, "test", device)
    out = data.frame(ends, preds)
    out.to_parquet(config.OUT_DIR / "preds_transformer.parquet", index=False)
    print(f"saved {len(out):,} test predictions ({out['date'].min()} -> {out['date'].max()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from outputs/checkpoint.pt")
    args = parser.parse_args()
    main(args.epochs, args.max_samples, args.resume)
