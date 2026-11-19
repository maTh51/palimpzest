#!/bin/bash
# =============================================================================
# Morfeu Initial Validation Suite: Proving Dynamic Stopping Rounds & Ablations
# Tests:
#   1. Decoupling j & ES_MIN_UPDATES=1 with varying ES_N_MIN in {4, 6, 8}
#      -> Demonstrates organic stopping across different rounds (t = 4, 5, 6, 7...)
#   2. Ablation of Stopping Criteria (ES_MODE = both vs efficacy vs futility)
#   3. Gray-Zone Expansion (ES_FUTILITY_GAMMA = 0.5)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_morfeu.sh"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$(pwd)/patched_src:$PYTHONPATH"
export DISABLE_MID_EXECUTION_PRUNING=1
export TIMEOUT_MODE=none
export CUAD_MAX_WORKERS=8

mkdir -p opt-profiling-data "$SCRIPT_DIR/logs"

# Ensure vLLM is running on Morfeu
if ! curl -sf "http://localhost:${MODEL1_PORT}/health" >/dev/null 2>&1 || ! curl -sf "http://localhost:${MODEL2_PORT}/health" >/dev/null 2>&1; then
    echo "[vLLM] Starting Morfeu vLLM servers..."
    bash "$SCRIPT_DIR/start_vllm_morfeu.sh"
fi

run_cuad_val() {
    local exp=$1 budget=$2 k=$3 j=$4 val=$5 test_n=$6 seed=$7 es_flag=$8 min_up=$9 n_min=${10} mode=${11:-both} gamma=${12:-1.0} z_val=${13:-1.645}
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP ES_MODE ES_MIN_UPDATES ES_N_MIN ES_FUTILITY_GAMMA ES_Z
    export TIMEOUT_MODE=none
    export CUAD_TEST_SAMPLES="$test_n"
    if [ "$es_flag" = "1" ]; then
        export EARLY_STOP=1
        export ES_MODE="$mode"
        export ES_MIN_UPDATES="$min_up"
        export ES_N_MIN="$n_min"
        export ES_FUTILITY_GAMMA="$gamma"
        export ES_Z="$z_val"
    fi
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${exp}-trajectory.jsonl"
    rm -f "$MAB_TRAJECTORY_LOG"
    echo "================================================================================"
    echo " [MORFEU VALIDATION] $exp"
    echo "   B=$budget, k=$k, j=$j, val=$val, test=$test_n, seed=$seed"
    echo "   EARLY_STOP=$es_flag, ES_MODE=$mode, ES_MIN_UPDATES=$min_up, ES_N_MIN=$n_min, GAMMA=$gamma"
    echo "================================================================================"
    python cuad-demo.py --k "$k" --j "$j" --sample-budget "$budget" --seed "$seed" \
        --val-examples "$val" --num-contracts "$test_n" \
        --priors-file cuad-priors.json --priors-mode default \
        --policy maxquality --optimizer-strategy pareto \
        --exp-name "$exp" --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
        --progress 2>&1 | tee "$SCRIPT_DIR/logs/${exp}.log"
}

# 1. Baseline reference with j=2 (B=60, k=5, j=2, val=15, test=15) -> runs ~11 rounds
run_cuad_val "morfeu-val-cuad-baseline-j2"   60 5 2 15 15 0 0 1 4 "both"     1.0 1.645

# 2. Dynamic Stopping without Round-3 Lock (ES_MIN_UPDATES=1, varying j and ES_N_MIN)
run_cuad_val "morfeu-val-cuad-j1-nmin4"      60 5 1 15 15 0 1 1 4 "both"     1.0 1.645
run_cuad_val "morfeu-val-cuad-j2-nmin5"      60 5 2 15 15 0 1 1 5 "both"     1.0 1.645
run_cuad_val "morfeu-val-cuad-j2-nmin6"      60 5 2 15 15 0 1 1 6 "both"     1.0 1.645
run_cuad_val "morfeu-val-cuad-j2-nmin8"      60 5 2 15 15 0 1 1 8 "both"     1.0 1.645

# 3. Gray-Zone Expansion & Mode Ablation (ES_MIN_UPDATES=1, j=2, ES_N_MIN=4)
run_cuad_val "morfeu-val-cuad-grayzone-g05"  60 5 2 15 15 0 1 1 4 "both"     0.5 1.645
run_cuad_val "morfeu-val-cuad-mode-efficacy" 60 5 2 15 15 0 1 1 4 "efficacy" 1.0 1.282

echo "=== [MORFEU VALIDATION] Summary of Stopping Rounds & Metrics ==="
python -c '
import json, os
exps = [
    "morfeu-val-cuad-baseline-j2",
    "morfeu-val-cuad-j1-nmin4",
    "morfeu-val-cuad-j2-nmin5",
    "morfeu-val-cuad-j2-nmin6",
    "morfeu-val-cuad-j2-nmin8",
    "morfeu-val-cuad-grayzone-g05",
    "morfeu-val-cuad-mode-efficacy",
]
for e in exps:
    m_p = f"opt-profiling-data/{e}-metrics.json"
    t_p = f"opt-profiling-data/{e}-trajectory.jsonl"
    m = json.load(open(m_p)) if os.path.exists(m_p) else {}
    lines = [json.loads(l) for l in open(t_p) if l.strip()] if os.path.exists(t_p) else []
    op1 = [l for l in lines if not l["logical_op_id"].startswith("0-")]
    rounds = len(op1)
    last = op1[-1] if op1 else {}
    samples = sum(v.get("samples_drawn", 0) for v in last.get("physical_operators", {}).values())
    ci = last.get("convergence_info") or {}
    print(f"{e:32s} | F1={m.get(\"f1\",0):.4f} | OptTime={m.get(\"optimization_time\",0):6.1f}s | StopRound={rounds:2d} | Samples={samples:2d} | Reason={str(ci.get(\"reason\")):10s} | nL={ci.get(\"n_leader\")} | gap={ci.get(\"gap\")}")
'
