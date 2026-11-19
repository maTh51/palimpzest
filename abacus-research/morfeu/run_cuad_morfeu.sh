#!/bin/bash
# CUAD timeout & Early-Stop experiments on Morfeu (Dual RTX 4090s).
# Assumes start_vllm_morfeu.sh is running.
#
# Usage (from abacus-research/):
#   bash morfeu/run_cuad_morfeu.sh smoke          # quick validation (~minutes)
#   TIMEOUTS="0 es" bash morfeu/run_cuad_morfeu.sh full
#   SEED=1 TIMEOUTS="0 es" bash morfeu/run_cuad_morfeu.sh full
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_morfeu.sh"
cd "$SCRIPT_DIR/.."                       # cuad-demo.py uses relative paths
mkdir -p opt-profiling-data "$SCRIPT_DIR/logs"

MODE="${1:-smoke}"
for P in "$MODEL1_PORT" "$MODEL2_PORT"; do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:$P/health" | grep -q 200 \
        || { echo "[ERR] nothing healthy on port $P -> run: bash morfeu/start_vllm_morfeu.sh"; exit 1; }
done

# Note: CUAD has 1 logical LLM operator (vs 3 in BioDex), so BUDGET=100 in CUAD equals
# the exact same 100-sample budget per logical operator as BUDGET=300 in BioDex.
if [ "$MODE" = "smoke" ]; then
    TIMEOUTS="${TIMEOUTS:-0 es}"; BUDGET="${BUDGET:-20}"; VAL="${VAL:-5}"; K=2; J=2
    export CUAD_TEST_SAMPLES="${CUAD_TEST_SAMPLES:-5}"
else
    TIMEOUTS="${TIMEOUTS:-0 es}"; BUDGET="${BUDGET:-100}"; VAL="${VAL:-20}"; K=5; J=4
    export CUAD_TEST_SAMPLES="${CUAD_TEST_SAMPLES:-50}"
fi
export CUAD_MAX_WORKERS="${CUAD_MAX_WORKERS:-8}"
export DISABLE_MID_EXECUTION_PRUNING=1

for T in $TIMEOUTS; do
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP
    if [ "$T" = "es" ]; then
        export TIMEOUT_MODE=none EARLY_STOP=1
        export ES_Z="${ES_Z:-1.645}" ES_MIN_UPDATES="${ES_MIN_UPDATES:-3}" ES_N_MIN="${ES_N_MIN:-4}"
        TAG="es"
    elif [ "$T" = "qprune" ]; then
        export TIMEOUT_MODE=none
        export QUALITY_PRUNE_RHO="${QUALITY_PRUNE_RHO_VALUE:-0.5}"
        export QUALITY_PRUNE_MIN_SAMPLES="${QUALITY_PRUNE_MIN_SAMPLES:-4}"
        TAG="qprune-rho${QUALITY_PRUNE_RHO}"
    elif [ "$T" = "dyn" ]; then
        export TIMEOUT_MODE=dynamic
        export SAMPLE_TIMEOUT="${DYN_FALLBACK:-120}"
        TAG="dyn"
    elif [ "$T" -eq 0 ]; then
        export TIMEOUT_MODE=none; TAG="0s"
    else
        export TIMEOUT_MODE=fixed SAMPLE_TIMEOUT="$T"; TAG="${T}s"
    fi

    POLICY_ARGS=(--policy "${POLICY:-maxquality}")
    [ -n "$TIME_BUDGET" ] && POLICY_ARGS+=(--time-budget "$TIME_BUDGET")

    SEED="${SEED:-0}"
    [ "$SEED" != "0" ] && TAG="${TAG}-s${SEED}"
    EXP="${EXP_OVERRIDE:-morfeu-cuad-${MODE}-timeout-${TAG}}"
    if [ "$MODE" = "full" ] && [ -f "opt-profiling-data/${EXP}-metrics.json" ] && [ -z "$FORCE" ]; then
        echo "[skip] $EXP already has metrics (set FORCE=1 to overwrite)"; continue
    fi
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${EXP}-trajectory.jsonl"
    export TIMEOUT_LOG_CSV="opt-profiling-data/${EXP}-timeouts.csv"
    rm -f "$MAB_TRAJECTORY_LOG" "$TIMEOUT_LOG_CSV"
    echo "=== $EXP  (timeout_mode=$TIMEOUT_MODE early_stop=${EARLY_STOP:-0} seed=$SEED budget=$BUDGET k=$K j=$J val=$VAL test=$CUAD_TEST_SAMPLES) ==="
    python cuad-demo.py --k "$K" --j "$J" --sample-budget "$BUDGET" --seed "$SEED" \
        --val-examples "$VAL" --num-contracts "$CUAD_TEST_SAMPLES" \
        "${POLICY_ARGS[@]}" --exp-name "$EXP" \
        --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" --progress \
        2>&1 | tee "$SCRIPT_DIR/logs/${EXP}.log"
    echo "=== done: $EXP -> $MAB_TRAJECTORY_LOG, $TIMEOUT_LOG_CSV ==="
done
