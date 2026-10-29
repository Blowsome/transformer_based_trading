#!/usr/bin/env bash
# Run every step in order; each step writes its log to outputs/logs/<step>.out.
# Check progress at any time with: python -m src.status
#
#   nohup caffeinate -i scripts/run_pipeline.sh > outputs/logs/pipeline.out 2>&1 &            # from scratch
#   RESUME=1 nohup caffeinate -i scripts/run_pipeline.sh > outputs/logs/pipeline.out 2>&1 &   # continue
#
# RESUME=1 skips finished steps and continues training from the last finished epoch.
# A memory/disk guard (src/guard.py) runs alongside and stops the pipeline cleanly before the Mac runs out.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/logs
mkdir -p "$LOG"
[[ -z "${RESUME:-}" ]] && rm -f "$LOG"/*.done

python -u -m src.guard --pid $$ >> "$LOG/guard.out" 2>&1 &
GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

run() {  # run <step name> <python -m arguments...>
  local name=$1; shift
  if [[ -n "${RESUME:-}" && -f "$LOG/$name.done" ]]; then
    echo "$(date '+%F %T') skip  $name (already done)"
    return
  fi
  echo "$(date '+%F %T') start $name"
  if ! python -u -m "$@" > "$LOG/$name.out" 2>&1; then
    echo "$(date '+%F %T') FAILED $name -> see $LOG/$name.out; continue later with RESUME=1"
    exit 1
  fi
  touch "$LOG/$name.done"
  echo "$(date '+%F %T') done  $name"
}

R=${RESUME:+--resume}
run download          src.download
run prepare           src.prepare
run train_reg_current src.train --experiment reg_current $R
run train_reg_strong  src.train --experiment reg_strong $R
run train_final       src.train --settings best $R
run baselines         src.baselines
run evaluate          src.evaluate
run notebook          jupyter nbconvert --to notebook --execute --inplace backtest.ipynb
echo "$(date '+%F %T') all done"
