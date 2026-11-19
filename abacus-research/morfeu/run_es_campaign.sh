#!/bin/bash
# Early-termination campaign on Morfeu (paired design: every seed runs BOTH 0s baseline and es).
# Seed changes validation AND test sets, so compare es vs 0s only within the same seed.
#   seed 0: 0s already exists (morfeu-biodex-full-timeout-0s) -> only es is run
#   seed 1, 2: 0s and es
# Every run saves: -metrics.json -profiling.json -records.json -output.csv -trajectory.jsonl
#                  -timeouts.csv  (opt-profiling-data/)  and the console log (morfeu/logs/)
# Usage (from abacus-research/):  bash morfeu/run_es_campaign.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "[campaign] smoke test of early termination"
TIMEOUTS="es" bash morfeu/run_biodex_morfeu.sh smoke
grep -q "early-stop" morfeu/logs/morfeu-biodex-smoke-timeout-es.log \
    && echo "[campaign] smoke: early-stop code path exercised" \
    || echo "[campaign] smoke: no early-stop event (budget too small to converge) - code ran without errors"
for S in ${SEEDS:-0 1 2}; do
    if [ "$S" = "0" ]; then RUNS="es"; else RUNS="0 es"; fi
    echo "[campaign] seed=$S runs=$RUNS  $(date)"
    SEED=$S TIMEOUTS="$RUNS" bash morfeu/run_biodex_morfeu.sh full
done
echo "[campaign] finished $(date)"
python morfeu/compare_experiments.py
