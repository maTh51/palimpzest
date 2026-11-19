#!/bin/bash
# BioDex timeout experiments on Morfeu (GPU 1). Assumes start_vllm_morfeu.sh is running
# and the nomic index was built.
#
# Usage (from abacus-research/):
#   bash morfeu/run_biodex_morfeu.sh smoke        # tiny run to validate the pipeline (~minutes)
#   bash morfeu/run_biodex_morfeu.sh full         # timeouts 0 30 60 120 dyn
#   TIMEOUTS="30 dyn" bash morfeu/run_biodex_morfeu.sh full
#   TIMEOUTS="dyn" bash morfeu/run_biodex_morfeu.sh smoke
# Timeout values: 0 = none, N = fixed N seconds, dyn = per-operator dynamic timeout
# (tune with DYN_TIMEOUT_QUANTILE / DYN_TIMEOUT_MULT / DYN_TIMEOUT_MIN / DYN_TIMEOUT_MAX / DYN_TIMEOUT_WARMUP).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_morfeu.sh"
cd "$SCRIPT_DIR/.."                       # biodex-demo.py uses relative paths
mkdir -p opt-profiling-data "$SCRIPT_DIR/logs"

MODE="${1:-smoke}"
for P in "$EMBEDDING_PORT" "$MODEL1_PORT" "$MODEL2_PORT"; do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:$P/health" | grep -q 200 \
        || { echo "[ERR] nothing healthy on port $P -> run: bash morfeu/start_vllm_morfeu.sh"; exit 1; }
done

if [ "$MODE" = "smoke" ]; then
    TIMEOUTS="${TIMEOUTS:-30}"; BUDGET="${BUDGET:-20}"; VAL="${VAL:-5}"; K=2; J=2
    export BIODEX_TEST_SAMPLES="${BIODEX_TEST_SAMPLES:-10}"   # final plan on 10 docs, not 250
else
    TIMEOUTS="${TIMEOUTS:-0 30 60 120 dyn}"; BUDGET="${BUDGET:-300}"; VAL="${VAL:-20}"; K=5; J=4
    export BIODEX_TEST_SAMPLES="${BIODEX_TEST_SAMPLES:-250}"
fi
# Dual-GPU setup: workers increased to 8 across GPU 0 & GPU 1
export BIODEX_MAX_WORKERS="${BIODEX_MAX_WORKERS:-8}"
export DISABLE_MID_EXECUTION_PRUNING=1

for T in $TIMEOUTS; do
    # T = 0 (no timeout) | <seconds> (fixed) | dyn (per-operator dynamic timeout)
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP
    if [ "$T" = "es" ]; then
        # No per-sample timeout (same as 0s) + efficacy/futility early termination; only one variable changes.
        # Calibrated offline in morfeu/analyze_early_stop_futility.py (z=1.645, >=3 updates, >=4 samples).
        export TIMEOUT_MODE=none EARLY_STOP=1
        export ES_Z="${ES_Z:-1.645}" ES_MIN_UPDATES="${ES_MIN_UPDATES:-3}" ES_N_MIN="${ES_N_MIN:-4}"
        TAG="es"
    elif [ "$T" = "qprune" ]; then
        # No per-sample timeout (same as 0s) + relative-quality pruning; only one variable changes.
        # rho=0.5 justified by morfeu/analyze_rho_margin.py (Pareto ops never < 0.85x best peer).
        export TIMEOUT_MODE=none
        export QUALITY_PRUNE_RHO="${QUALITY_PRUNE_RHO_VALUE:-0.5}"
        export QUALITY_PRUNE_MIN_SAMPLES="${QUALITY_PRUNE_MIN_SAMPLES:-4}"
        TAG="qprune-rho${QUALITY_PRUNE_RHO}"
    elif [ "$T" = "dyn" ]; then
        export TIMEOUT_MODE=dynamic
        export SAMPLE_TIMEOUT="${DYN_FALLBACK:-120}"   # used only until an operator has DYN_TIMEOUT_WARMUP finished samples
        TAG="dyn"
    elif [ "$T" -eq 0 ]; then
        export TIMEOUT_MODE=none; TAG="0s"
    else
        export TIMEOUT_MODE=fixed SAMPLE_TIMEOUT="$T"; TAG="${T}s"
    fi
    # Same optimization policy for every run, so the timeout is the only thing that changes.
    # (The old Axia scripts also switched to maxqualityatfixedtime --time-budget T, mixing two variables.)
    POLICY_ARGS=(--policy "${POLICY:-maxquality}")
    [ -n "$TIME_BUDGET" ] && POLICY_ARGS+=(--time-budget "$TIME_BUDGET")

    # seed 0 keeps the historical names; other seeds get a -s<N> suffix (never overwrite old profiles)
    SEED="${SEED:-0}"
    [ "$SEED" != "0" ] && TAG="${TAG}-s${SEED}"
    EXP="morfeu-biodex-${MODE}-timeout-${TAG}"
    if [ "$MODE" = "full" ] && [ -f "opt-profiling-data/${EXP}-metrics.json" ] && [ -z "$FORCE" ]; then
        echo "[skip] $EXP already has metrics (set FORCE=1 to overwrite)"; continue
    fi
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${EXP}-trajectory.jsonl"
    export TIMEOUT_LOG_CSV="opt-profiling-data/${EXP}-timeouts.csv"
    rm -f "$MAB_TRAJECTORY_LOG" "$TIMEOUT_LOG_CSV"
    echo "=== $EXP  (timeout_mode=$TIMEOUT_MODE early_stop=${EARLY_STOP:-0} seed=$SEED budget=$BUDGET k=$K j=$J val=$VAL test=$BIODEX_TEST_SAMPLES) ==="
    python biodex-demo.py --k "$K" --j "$J" --sample-budget "$BUDGET" --seed "$SEED" \
        --val-examples "$VAL" "${POLICY_ARGS[@]}" --exp-name "$EXP" \
        --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" --progress \
        2>&1 | tee "$SCRIPT_DIR/logs/${EXP}.log"
    echo "=== done: $EXP -> $MAB_TRAJECTORY_LOG, $TIMEOUT_LOG_CSV ==="
done
