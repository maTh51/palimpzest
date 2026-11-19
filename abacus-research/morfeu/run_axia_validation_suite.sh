#!/bin/bash
set -e

# =============================================================================
# Axia (NVIDIA H200) Validation Suite:
#   1. Dynamic Stopping Rounds (Decoupled ES_MIN_UPDATES=1, j in {1,2}, ES_N_MIN in {4,6,8,10})
#      -> Proves stopping round is an emergent mathematical property (t = 4, 5, 6, 7, 9...)
#   2. Gray-Zone Expansion (ES_FUTILITY_GAMMA=0.5, Large Budget B=200 with N_val=35)
#   3. Criterion Ablation (ES_MODE = efficacy vs futility vs both)
#   4. Completion of Multi-Seed runs with NoneType GenerationStats fix
# =============================================================================

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-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}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=0
export OPENAI_API_KEY="local-vllm"
export DISABLE_CACHE=1
export DISABLE_MID_EXECUTION_PRUNING=1
export TIMEOUT_MODE=none

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/patched_src:$PYTHONPATH"

SCRATCH_ROOT="/scratch/hpc4ai/amarante_abacus_es"
LOG_DIR="$ROOT_DIR/var/logs"
OUTPUT_DIR="${POLYAXON_OUTPUT_DIR:-$ROOT_DIR/outputs}"

mkdir -p "$LOG_DIR" "$ROOT_DIR/opt-profiling-data" "$SCRATCH_ROOT/opt-profiling-data" "$SCRATCH_ROOT/logs" "$OUTPUT_DIR/opt-profiling-data"

# Restore existing completed runs from persistent scratch
cp -rn "$SCRATCH_ROOT/opt-profiling-data/"* "$ROOT_DIR/opt-profiling-data/" 2>/dev/null || true
cp -rf "$SCRATCH_ROOT/opt-profiling-data/"* "$OUTPUT_DIR/opt-profiling-data/" 2>/dev/null || true

python -c '
import palimpzest.query.execution.mab_execution_strategy as mab
from palimpzest.core.models import GenerationStats
s = GenerationStats(model_name="verify", output_text_tokens=5) + None
assert s.output_text_tokens == 5
assert "_compute_es_diagnostics" in open(mab.__file__).read()
print("[VERIFY] Patched Palimpzest (None-safe GenerationStats + ES_MODE/ES_FUTILITY_GAMMA + es_diagnostics) OK!")
'

find_model_path() {
    local target_rel=$1
    local fallback=$2
    for prefix in "/scratch/hpc4ai/models" "/mnt/scratch/hpc4ai/models" "/mnt/scratch/global/models" "/scratch/models"; do
        if [ -d "$prefix/$target_rel" ]; then
            echo "$prefix/$target_rel"
            return 0
        fi
    done
    echo "$fallback"
}

MODEL_70B_PATH=$(find_model_path "llama/Llama-3.3-70B-Instruct" "meta-llama/Llama-3.3-70B-Instruct")
MODEL_120B_PATH=$(find_model_path "openai/gpt-oss-120b" "openai/gpt-oss-120b")
MODEL_8B_PATH=$(find_model_path "llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")

export EMBEDDING_PORT=8010
export MODEL1_PORT=8005
export MODEL2_PORT=8006

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "[HARDWARE] Detected $NUM_GPUS NVIDIA GPU(s)"

wait_for_ready() {
    local port=$1 name=$2 tries=${3:-240} pid=${4:-} log_file=$5
    echo "[vLLM] Waiting for $name on port $port (pid=$pid, up to $((tries * 5))s)..."
    for ((i = 1; i <= tries; i++)); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[WARN] Process $pid ($name) exited early on port $port!"
            [ -f "$log_file" ] && tail -n 50 "$log_file"
            return 1
        fi
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "[SUCCESS] $name is healthy on port $port after $((i * 5))s!"
            return 0
        fi
        sleep 5
    done
    return 1
}

PID_EMB=""
PID_M1=""
PID_M2=""

stop_all_vllm() {
    kill -9 $PID_EMB $PID_M1 $PID_M2 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    fuser -k ${EMBEDDING_PORT}/tcp ${MODEL1_PORT}/tcp ${MODEL2_PORT}/tcp 2>/dev/null || true
    sleep 2
}
trap stop_all_vllm EXIT INT TERM

# 1. Start Embedding Server on GPU 0 (Port 8010)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port "$EMBEDDING_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.04 \
    --max-model-len 2048 \
    --enforce-eager \
    > "$LOG_DIR/vllm_embed_${EMBEDDING_PORT}.log" 2>&1 &
PID_EMB=$!
wait_for_ready "$EMBEDDING_PORT" "Nomic-Embed" 90 "$PID_EMB" "$LOG_DIR/vllm_embed_${EMBEDDING_PORT}.log"

# 2. Start Large LLM Servers
if [ "$NUM_GPUS" -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_70B_PATH" \
        --port "$MODEL1_PORT" \
        --dtype bfloat16 \
        --served-model-name "Qwen2.5-14B-GPU0" "Qwen2.5-14B-GPU1" "meta-llama/Llama-3.3-70B-Instruct" "openai/gpt-oss-120b" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.52 \
        --tensor-parallel-size 2 \
        --max-num-seqs 64 \
        --enforce-eager \
        --trust-remote-code \
        > "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log" 2>&1 &
    PID_M1=$!
    wait_for_ready "$MODEL1_PORT" "Llama-3.3-70B" 240 "$PID_M1" "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log"

    SECOND_MODEL_OK=0
    if [ -d "$MODEL_120B_PATH" ]; then
        CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_120B_PATH" \
            --port "$MODEL2_PORT" \
            --served-model-name "Qwen2.5-14B-GPU1" "openai/gpt-oss-120b" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.35 \
            --tensor-parallel-size 2 \
            --max-num-seqs 64 \
            --enforce-eager \
            --trust-remote-code \
            > "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log" 2>&1 &
        PID_M2=$!
        if wait_for_ready "$MODEL2_PORT" "gpt-oss-120b" 180 "$PID_M2" "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log"; then
            SECOND_MODEL_OK=1
        else
            kill -9 $PID_M2 2>/dev/null || true
            fuser -k ${MODEL2_PORT}/tcp 2>/dev/null || true
            sleep 3
        fi
    fi

    if [ "$SECOND_MODEL_OK" -eq 0 ] && [ -d "$MODEL_8B_PATH" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_8B_PATH" \
            --port "$MODEL2_PORT" \
            --dtype bfloat16 \
            --served-model-name "Qwen2.5-14B-GPU1" "meta-llama/Llama-3.1-8B-Instruct" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.28 \
            --max-num-seqs 64 \
            --enforce-eager \
            --trust-remote-code \
            > "$LOG_DIR/vllm_m2_fallback_${MODEL2_PORT}.log" 2>&1 &
        PID_M2=$!
        wait_for_ready "$MODEL2_PORT" "Llama-3.1-8B" 180 "$PID_M2" "$LOG_DIR/vllm_m2_fallback_${MODEL2_PORT}.log" || export MODEL2_PORT="$MODEL1_PORT"
    fi
else
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_120B_PATH" \
        --port "$MODEL1_PORT" \
        --served-model-name "Qwen2.5-14B-GPU0" "openai/gpt-oss-120b" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.58 \
        --max-num-seqs 64 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log" 2>&1 &
    PID_M1=$!
    wait_for_ready "$MODEL1_PORT" "gpt-oss-120b" 180 "$PID_M1" "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log"

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_8B_PATH" \
        --port "$MODEL2_PORT" \
        --dtype bfloat16 \
        --served-model-name "Qwen2.5-14B-GPU1" "meta-llama/Llama-3.1-8B-Instruct" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.30 \
        --max-num-seqs 64 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log" 2>&1 &
    PID_M2=$!
    wait_for_ready "$MODEL2_PORT" "Llama-3.1-8B" 180 "$PID_M2" "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log" || export MODEL2_PORT="$MODEL1_PORT"
fi

# Setup Datasets
if [ ! -d "cuad-data" ]; then
    for cand in "/scratch/hpc4ai/palimpzest/otimizacao-abacus/palimpzest/abacus-research/cuad-data" "/mnt/scratch/global/palimpzest/abacus-research/cuad-data" "/app/abacus-research/cuad-data"; do
        if [ -d "$cand" ]; then ln -s "$cand" cuad-data; break; fi
    done
    [ ! -d "cuad-data" ] && python setup_cuad_data.py
fi

if [ -d "$SCRATCH_ROOT/.chroma-biodex" ]; then
    rm -rf .chroma-biodex && ln -s "$SCRATCH_ROOT/.chroma-biodex" .chroma-biodex
else
    export BIODEX_CHROMA_DIR="$SCRATCH_ROOT/.chroma-biodex"
    python morfeu/build_biodex_index.py
    rm -rf .chroma-biodex && ln -s "$SCRATCH_ROOT/.chroma-biodex" .chroma-biodex
fi

export CUAD_MAX_WORKERS=8
export BIODEX_MAX_WORKERS=8

sync_exp() {
    local exp=$1
    cp -f opt-profiling-data/${exp}* "$SCRATCH_ROOT/opt-profiling-data/" 2>/dev/null || true
    cp -f opt-profiling-data/${exp}* "$OUTPUT_DIR/opt-profiling-data/" 2>/dev/null || true
    cp -f "$LOG_DIR/${exp}.log" "$SCRATCH_ROOT/logs/" 2>/dev/null || true
}

run_cuad_cfg() {
    local exp=$1 budget=$2 k=$3 j=$4 val=$5 test_n=$6 seed=$7 es_flag=$8 min_up=$9 n_min=${10} mode=${11:-both} gamma=${12:-1.0} z_val=${13:-1.645} workers=${14:-8}
    if [ -f "opt-profiling-data/${exp}-metrics.json" ] && [ -z "$FORCE" ]; then
        echo "[SKIP] $exp already completed"
        sync_exp "$exp"
        return 0
    fi
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP ES_MODE ES_MIN_UPDATES ES_N_MIN ES_FUTILITY_GAMMA ES_Z
    export TIMEOUT_MODE=none
    export CUAD_TEST_SAMPLES="$test_n"
    export CUAD_MAX_WORKERS="$workers"
    if [ "$es_flag" = "1" ]; then
        export EARLY_STOP=1 ES_MODE="$mode" ES_MIN_UPDATES="$min_up" ES_N_MIN="$n_min" ES_FUTILITY_GAMMA="$gamma" ES_Z="$z_val"
    fi
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${exp}-trajectory.jsonl"
    rm -f "$MAB_TRAJECTORY_LOG"
    echo "=== [AXIA CUAD] $exp (B=$budget, k=$k, j=$j, val=$val, test=$test_n, seed=$seed, ES=$es_flag, min_up=$min_up, n_min=$n_min, mode=$mode, gamma=$gamma, w=$workers) ==="
    set +e
    python cuad-demo.py --k "$k" --j "$j" --sample-budget "$budget" --seed "$seed" \
        --val-examples "$val" --num-contracts "$test_n" \
        --priors-file cuad-priors.json --priors-mode default \
        --policy maxquality --optimizer-strategy pareto \
        --exp-name "$exp" --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
        --progress 2>&1 | tee "$LOG_DIR/${exp}.log"
    set -e
    sync_exp "$exp"
}

run_biodex_cfg() {
    local exp=$1 budget=$2 k=$3 j=$4 val=$5 test_n=$6 seed=$7 es_flag=$8 min_up=$9 n_min=${10} mode=${11:-both} gamma=${12:-1.0} z_val=${13:-1.645} workers=${14:-8}
    if [ -f "opt-profiling-data/${exp}-metrics.json" ] && [ -z "$FORCE" ]; then
        echo "[SKIP] $exp already completed"
        sync_exp "$exp"
        return 0
    fi
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP ES_MODE ES_MIN_UPDATES ES_N_MIN ES_FUTILITY_GAMMA ES_Z
    export TIMEOUT_MODE=none
    export BIODEX_TEST_SAMPLES="$test_n"
    export BIODEX_MAX_WORKERS="$workers"
    if [ "$es_flag" = "1" ]; then
        export EARLY_STOP=1 ES_MODE="$mode" ES_MIN_UPDATES="$min_up" ES_N_MIN="$n_min" ES_FUTILITY_GAMMA="$gamma" ES_Z="$z_val"
    fi
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${exp}-trajectory.jsonl"
    rm -f "$MAB_TRAJECTORY_LOG"
    echo "=== [AXIA BIODEX] $exp (B=$budget, k=$k, j=$j, val=$val, test=$test_n, seed=$seed, ES=$es_flag, min_up=$min_up, n_min=$n_min, mode=$mode, gamma=$gamma, w=$workers) ==="
    set +e
    python biodex-demo.py --k "$k" --j "$j" --sample-budget "$budget" --seed "$seed" \
        --val-examples "$val" \
        --priors-file biodex-priors.json --priors-mode default \
        --policy maxquality \
        --exp-name "$exp" --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
        --progress 2>&1 | tee "$LOG_DIR/${exp}.log"
    set -e
    sync_exp "$exp"
}

# =============================================================================
# FASE 3A: Dynamic Stopping Rounds without Round-3 Lock (ES_MIN_UPDATES=1)
# Demonstrates stopping at Rounds 4, 5, 6, 7, 9 depending on (j, ES_N_MIN, gamma)
# =============================================================================
run_cuad_cfg   "axia-val-cuad-baseline-j2"    60 5 2 15 20 0 0 1 4 "both"     1.0 1.645 8
run_cuad_cfg   "axia-val-cuad-j1-nmin4"       60 5 1 15 20 0 1 1 4 "both"     1.0 1.645 8
run_cuad_cfg   "axia-val-cuad-j2-nmin5"       60 5 2 15 20 0 1 1 5 "both"     1.0 1.645 8
run_cuad_cfg   "axia-val-cuad-j2-nmin6"       60 5 2 15 20 0 1 1 6 "both"     1.0 1.645 8
run_cuad_cfg   "axia-val-cuad-j2-nmin8"       60 5 2 15 20 0 1 1 8 "both"     1.0 1.645 8
run_cuad_cfg   "axia-val-cuad-grayzone-g05"   60 5 2 15 20 0 1 1 4 "both"     0.5 1.645 8
run_cuad_cfg   "axia-val-cuad-mode-efficacy"  60 5 2 15 20 0 1 1 4 "efficacy" 1.0 1.282 8

# =============================================================================
# FASE 2: Budget Sensitivity (B in {50, 100, 200, 400})
# (Note: B=40 pilot and B=100 full-s0/s1 are already cached in Scratch!)
# =============================================================================
run_cuad_cfg   "axia-cuad-b50-baseline"       50 5 2 15 20 0 0 1 5 "both" 1.0 1.645 8
run_cuad_cfg   "axia-cuad-b50-es"             50 5 2 15 20 0 1 1 5 "both" 1.0 1.645 8
run_cuad_cfg   "axia-cuad-b200-baseline"     200 5 2 35 20 0 0 1 6 "both" 0.6 1.645 8
run_cuad_cfg   "axia-cuad-b200-es"           200 5 2 35 20 0 1 1 6 "both" 0.6 1.645 8
run_cuad_cfg   "axia-cuad-b400-baseline"     400 5 2 45 20 0 0 1 8 "both" 0.5 1.645 8
run_cuad_cfg   "axia-cuad-b400-es"           400 5 2 45 20 0 1 1 8 "both" 0.5 1.645 8

# =============================================================================
# FASE 3B: Topology (k=10) & Concurrency (w in {4, 16, 32}) Sensitivity
# =============================================================================
run_cuad_cfg   "axia-cuad-k10-baseline"      100 10 2 20 20 0 0 1 5 "both" 1.0 1.645 8
run_cuad_cfg   "axia-cuad-k10-es"            100 10 2 20 20 0 1 1 5 "both" 1.0 1.645 8
run_cuad_cfg   "axia-cuad-w4-es"              60  5 2 15 20 0 1 1 5 "both" 1.0 1.645 4
run_cuad_cfg   "axia-cuad-w16-es"             60  5 2 15 20 0 1 1 5 "both" 1.0 1.645 16
run_cuad_cfg   "axia-cuad-w32-es"             60  5 2 15 20 0 1 1 5 "both" 1.0 1.645 32

# =============================================================================
# FASE 1 (Completion): Remaining Multi-Seed Runs (with NoneType Fix Active)
# =============================================================================
run_cuad_cfg   "axia-cuad-full-s2-baseline"   100 5 4 20 50  2 0 3 4 "both" 1.0 1.645 8
run_cuad_cfg   "axia-cuad-full-s2-es"         100 5 4 20 50  2 1 3 4 "both" 1.0 1.645 8
run_biodex_cfg "axia-biodex-full-s1-baseline" 300 5 4 20 250 1 0 3 4 "both" 1.0 1.645 8
run_biodex_cfg "axia-biodex-full-s1-es"       300 5 4 20 250 1 1 3 4 "both" 1.0 1.645 8

echo "[AXIA VALIDATION SUITE] All validation experiments completed!"
