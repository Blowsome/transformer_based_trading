"""Safety guard for long runs: stops the pipeline cleanly before the Mac runs out of memory or disk.

Started by scripts/run_pipeline.sh. Every CHECK_SECONDS it checks macOS's memory pressure level and free disk.
If memory pressure is critical (or available memory is nearly gone) for STRIKES checks in a row, or disk falls
below MIN_FREE_DISK_GB, it stops the running pipeline step (SIGTERM) and writes the reason to outputs/logs/guard.out.
Training checkpoints every epoch, so `RESUME=1 scripts/run_pipeline.sh` continues where it stopped.

    python -m src.guard --pid <pipeline pid>
"""
import argparse
import os
import signal
import subprocess
import time

import psutil

import config

CHECK_SECONDS = 15
CRITICAL_PRESSURE = 4        # kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical
MIN_FREE_MEMORY_GB = 0.25    # backup check; macOS keeps "available" low on purpose, so this is only a last resort
MIN_FREE_DISK_GB = 3.0
STRIKES = 3                  # consecutive low-memory checks (45 s) before stopping, so short spikes are ignored
STOP_FILE = config.LOG_DIR / "STOPPED_BY_GUARD"


def pressure_level() -> int:
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True)
    return int(out.stdout.strip() or 1)


def pipeline_steps(pipeline_pid: int) -> list[psutil.Process]:
    try:
        return psutil.Process(pipeline_pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def stop(pipeline_pid: int, reason: str) -> None:
    msg = f"{time.strftime('%F %T')} STOPPING the pipeline: {reason}"
    print(msg, flush=True)
    STOP_FILE.write_text(msg + "\nContinue later with: RESUME=1 scripts/run_pipeline.sh\n")
    for p in pipeline_steps(pipeline_pid):
        if p.pid != os.getpid():
            p.send_signal(signal.SIGTERM)
    os.kill(pipeline_pid, signal.SIGTERM)


def main(pipeline_pid: int) -> None:
    STOP_FILE.unlink(missing_ok=True)
    print(f"{time.strftime('%F %T')} guard started for pid {pipeline_pid}: stop if memory pressure is critical "
          f"for {STRIKES * CHECK_SECONDS}s or free disk < {MIN_FREE_DISK_GB} GB", flush=True)
    strikes, last_report = 0, 0.0
    while psutil.pid_exists(pipeline_pid):
        level = pressure_level()
        available = psutil.virtual_memory().available / 1e9
        disk = psutil.disk_usage(str(config.ROOT)).free / 1e9
        swap = psutil.swap_memory().used / 1e9
        low = level >= CRITICAL_PRESSURE or available < MIN_FREE_MEMORY_GB
        strikes = strikes + 1 if low else 0
        if time.time() - last_report > 600 or (low and strikes == 1):  # a line every 10 minutes, for the record
            state = {1: "normal", 2: "warning", 4: "critical"}.get(level, str(level))
            print(f"{time.strftime('%F %T')} memory pressure {state}, available {available:.1f} GB, "
                  f"swap used {swap:.1f} GB, free disk {disk:.0f} GB", flush=True)
            last_report = time.time()
        if strikes >= STRIKES:
            stop(pipeline_pid, f"memory pressure level {level}, available {available:.2f} GB, swap {swap:.1f} GB")
            return
        if disk < MIN_FREE_DISK_GB:
            stop(pipeline_pid, f"free disk {disk:.1f} GB")
            return
        time.sleep(CHECK_SECONDS)
    print(f"{time.strftime('%F %T')} pipeline finished, guard exits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    main(parser.parse_args().pid)
