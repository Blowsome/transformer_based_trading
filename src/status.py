"""Where is the pipeline? Reads the progress files and logs in outputs/.

    python -m src.status
    while true; do clear; python -m src.status; sleep 30; done     # refresh every 30 s (Ctrl+C to stop)
"""
import json
import subprocess
import time

import config

STEPS = ["download", "prepare", "train_reg_current", "train_reg_strong", "train_final", "baselines", "evaluate",
         "notebook"]
LOG = config.LOG_DIR


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def minutes(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{seconds / 60:.0f} min"


def last_line(path) -> str:
    lines = path.read_text(errors="ignore").replace("\r", "\n").strip().splitlines() if path.exists() else []
    noise = ("Warning", "warn", "http", "inplace", "df.", "data.")  # library deprecation chatter
    lines = [l for l in lines if l.strip() and not l.startswith(" ") and not any(n in l for n in noise)]
    return lines[-1][:160] if lines else ""


def pipeline() -> str | None:
    """Prints the step list; returns the step that is running now."""
    active = running("run_pipeline.sh")
    current = None
    print(f"Pipeline: {'RUNNING' if active else 'not running'}")
    for step in STEPS:
        if (LOG / f"{step}.done").exists():
            mark = "done"
        elif active and (LOG / f"{step}.out").exists() and current is None:
            mark, current = "RUNNING", step
        elif (LOG / f"{step}.out").exists():
            mark = "STOPPED before finishing"
        else:
            mark = "waiting"
        detail = last_line(LOG / f"{step}.out") if mark in ("RUNNING", "STOPPED before finishing") else ""
        print(f"  {step:18s} {mark:25s} {detail}")
    stop = LOG / "STOPPED_BY_GUARD"
    if stop.exists():
        print("  !! " + stop.read_text().strip().replace("\n", "\n  !! "))
    elif not active and any((LOG / f"{s}.out").exists() and not (LOG / f"{s}.done").exists() for s in STEPS):
        print("  -> continue with: RESUME=1 nohup caffeinate -i scripts/run_pipeline.sh "
              "> outputs/logs/pipeline.out 2>&1 &")
    return current


def training() -> None:
    files = sorted([config.OUT_DIR / "train_progress.json", *config.OUT_DIR.glob("experiments/*/train_progress.json")],
                   key=lambda p: p.stat().st_mtime if p.exists() else 0)
    files = [p for p in files if p.exists()]
    for summary in sorted(config.OUT_DIR.glob("experiments/*/summary.json")):
        s = json.loads(summary.read_text())
        print(f"Experiment {summary.parent.name}: val IC {s['val_ic']:+.4f} "
              f"(dropout {s['dropout']}, weight decay {s['weight_decay']})")
    if not files:
        return
    s = json.loads(files[-1].read_text())  # the most recent training run
    is_running = running("python.* -m src.train")
    print(f"Training '{s['name']}': {s['phase'] if is_running or s['phase'] == 'done' else 'STOPPED'}"
          f" (started {minutes(time.time() - s['started'])} ago)")
    if is_running and s["phase"] == "training" and "step" in s:
        elapsed = time.time() - s["epoch_started"]
        per_step = elapsed / max(s["step"], 1)
        print(f"  model {s['seed_number']}/{s['n_seeds']} (seed {s['seed']}) | epoch {s['epoch']}/{s['epochs']} | "
              f"step {s['step']}/{s['steps']} ({s['step'] / s['steps']:.0%}) | train loss {s['train_loss']:+.4f}")
        if s["step"]:
            print(f"  this epoch ends in ~{minutes(per_step * (s['steps'] - s['step']))} "
                  f"(~{minutes(per_step * s['steps'])} per epoch); last update {time.time() - s['updated']:.0f}s ago")
    for seed, log in s["history"].items():
        if log:
            best = max(log, key=lambda r: r["val_ic"])
            print(f"  seed {seed}: {len(log)} epochs, best val IC {best['val_ic']:+.4f} (epoch {best['epoch']}) | "
                  "by epoch: " + " ".join(f"{r['val_ic']:+.3f}" for r in log))
    print(f"  (val IC = daily rank correlation between predictions and 5-day returns, "
          f"{config.VAL_START} -> {config.VAL_END}; above ~0.02 is useful)")


def guard() -> None:
    path = LOG / "guard.out"
    if path.exists():
        print(f"Guard: {'watching' if running('python.* -m src.guard') else 'not running'} | {last_line(path)}")


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    pipeline()
    print()
    training()
    guard()
