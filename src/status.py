"""Progress of background runs, read from their logs in outputs/.

    python -m src.status
    watch -n 30 python -m src.status   # refresh every 30 s (brew install watch)
"""
import re
import subprocess
import time

import config


def running(module: str) -> bool:
    return subprocess.run(["pgrep", "-f", f"python.* -m {module}"], capture_output=True).returncode == 0


def ago(seconds: float) -> str:
    return f"{seconds / 60:.0f} min ago" if seconds >= 60 else f"{seconds:.0f}s ago"


def transformer() -> None:
    path = config.OUT_DIR / "train.out"
    if not path.exists():
        print("Transformer: not started (no outputs/train.out)")
        return
    text = path.read_text()
    epochs = re.findall(r"epoch\s+(\d+) \| train ([\d.]+) \| val ([\d.]+) \| (\d+)s", text)
    is_running = running("src.train")
    print(f"Transformer: {'RUNNING' if is_running else 'STOPPED'}")
    for e, tr, va, s in epochs:
        print(f"  epoch {int(e):2d}/{config.EPOCHS} | train {tr} | val {va} | {s}s")

    if "saved" in text and "test predictions" in text:
        print("  finished: " + text.strip().splitlines()[-1])
    elif "early stopping" in text:
        print("  early stopping triggered, writing test predictions ...")
    elif is_running:
        current = int(epochs[-1][0]) + 1 if epochs else 1
        since = time.time() - path.stat().st_mtime  # log is written at the end of each epoch
        if epochs:
            avg = sum(int(s) for *_, s in epochs) / len(epochs)
            print(f"  epoch {current}/{config.EPOCHS} in progress, ~{min(since / avg, 0.99):.0%} done "
                  f"(last epoch ended {ago(since)}, ~{avg / 60:.1f} min per epoch)")
        else:
            print(f"  epoch 1/{config.EPOCHS} in progress (started {ago(since)})")
    else:
        print("  stopped before finishing; continue with: python -m src.train --resume")
        if "Error" in text:
            print("  last lines:\n    " + "\n    ".join(text.strip().splitlines()[-3:]))


def baselines() -> None:
    path = config.OUT_DIR / "baselines.out"
    if not path.exists():
        print("Baselines (ARIMA / random / zero): not started (no outputs/baselines.out)")
        return
    text = path.read_text()
    is_running = running("src.baselines")
    print(f"Baselines (ARIMA / random / zero): {'RUNNING' if is_running else 'STOPPED'}")
    if "saved arima" in text:
        print("  finished: " + text.strip().splitlines()[-1])
    else:
        last = text.replace("\r", "\n").strip().splitlines()  # tqdm redraws its bar with \r
        print("  " + (last[-1] if last else "starting ...")
              + ("" if is_running else "\n  stopped before finishing; rerun: python -m src.baselines"))


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    transformer()
    print()
    baselines()
