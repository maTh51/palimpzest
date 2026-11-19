#!/bin/bash
set -e

# =============================================================================
# Axia (NVIDIA H200) Full Stress-Test Campaign: Baseline Abacus vs. Nosso Método (ES)
# Models:
#   - Port 8005 (Qwen2.5-14B-GPU0 alias): meta-llama/Llama-3.3-70B-Instruct
#   - Port 8006 (Qwen2.5-14B-GPU1 alias): openai/gpt-oss-120b (fallback: Llama-3.1-8B / Port 8005)
#   - Port 8010 (Embedding server):       nomic-ai/nomic-embed-text-v1
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

# Restore any previously completed runs from persistent scratch (checkpoint-resume support)
echo "=== Checking existing files in $SCRATCH_ROOT/opt-profiling-data ==="
ls -la "$SCRATCH_ROOT/opt-profiling-data/" 2>/dev/null || true
cp -rn "$SCRATCH_ROOT/opt-profiling-data/"* "$ROOT_DIR/opt-profiling-data/" 2>/dev/null || true
cp -rf "$SCRATCH_ROOT/opt-profiling-data/"* "$OUTPUT_DIR/opt-profiling-data/" 2>/dev/null || true

if [ -f "$ROOT_DIR/opt-profiling-data/axia-pilot-summary.txt" ]; then
    echo "=== Found completed Pilot Summary in Scratch ==="
    cat "$ROOT_DIR/opt-profiling-data/axia-pilot-summary.txt"
fi

echo "=========================================================================="
echo " [AXIA FULL STRESS SUITE] Starting Campaign (Llama-3.3-70B + gpt-oss-120b)"
echo " Working Dir : $ROOT_DIR"
echo " Scratch Dir : $SCRATCH_ROOT"
echo " Outputs Dir : $OUTPUT_DIR"
echo "=========================================================================="
nvidia-smi

# Verify patched Palimpzest is active
python -c '
import palimpzest as pz
import palimpzest.query.execution.mab_execution_strategy as mab
print("[VERIFY] Loaded palimpzest from:", pz.__file__)
print("[VERIFY] Loaded mab_execution_strategy from:", mab.__file__)
assert "EARLY_STOP" in open(mab.__file__).read(), "ERROR: EARLY_STOP not found in mab_execution_strategy.py!"
print("[VERIFY] Efficacy + Futility Early Stopping (EARLY_STOP) verified OK!")
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

echo "[MODELS] Llama-3.3-70B path : $MODEL_70B_PATH"
echo "[MODELS] gpt-oss-120b path  : $MODEL_120B_PATH"
echo "[MODELS] Llama-3.1-8B path  : $MODEL_8B_PATH"

export EMBEDDING_PORT=8010
export MODEL1_PORT=8005
export MODEL2_PORT=8006

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "[HARDWARE] Detected $NUM_GPUS NVIDIA GPU(s)"

wait_for_ready() {
    local port=$1 name=$2 tries=${3:-240} pid=${4:-}
    local log_file=$5
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
    echo "[WARN] Timeout waiting for $name on port $port"
    [ -f "$log_file" ] && tail -n 50 "$log_file"
    return 1
}

PID_EMB=""
PID_M1=""
PID_M2=""

stop_all_vllm() {
    echo "[vLLM] Cleaning up vLLM servers..."
    kill -9 $PID_EMB $PID_M1 $PID_M2 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    fuser -k ${EMBEDDING_PORT}/tcp ${MODEL1_PORT}/tcp ${MODEL2_PORT}/tcp 2>/dev/null || true
    sleep 2
}
trap stop_all_vllm EXIT INT TERM

# 1. Start Embedding Server on GPU 0 (Port 8010)
echo "[vLLM] Starting nomic-ai/nomic-embed-text-v1 on port $EMBEDDING_PORT (GPU 0)..."
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

# 2. Start Large LLM Servers across available H200 GPUs
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "[vLLM] 2x H200 GPU Mode ($NUM_GPUS GPUs):"
    echo "  -> Launching Llama-3.3-70B-Instruct on Port $MODEL1_PORT (TP=2 across GPUs 0,1, util=0.52)..."
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
        echo "  -> Trying openai/gpt-oss-120b on Port $MODEL2_PORT (TP=2 across GPUs 0,1, util=0.35)..."
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
            echo "[WARN] gpt-oss-120b did not fit alongside 70B on shared 2xH200; cleaning Port $MODEL2_PORT..."
            kill -9 $PID_M2 2>/dev/null || true
            fuser -k ${MODEL2_PORT}/tcp 2>/dev/null || true
            sleep 3
        fi
    fi

    if [ "$SECOND_MODEL_OK" -eq 0 ] && [ -d "$MODEL_8B_PATH" ]; then
        echo "  -> Trying Llama-3.1-8B-Instruct on Port $MODEL2_PORT (GPU 1, util=0.28)..."
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
        if wait_for_ready "$MODEL2_PORT" "Llama-3.1-8B" 180 "$PID_M2" "$LOG_DIR/vllm_m2_fallback_${MODEL2_PORT}.log"; then
            SECOND_MODEL_OK=1
        else
            echo "[WARN] Secondary server on Port $MODEL2_PORT did not start; routing MODEL2_PORT -> MODEL1_PORT ($MODEL1_PORT)"
            kill -9 $PID_M2 2>/dev/null || true
            export MODEL2_PORT="$MODEL1_PORT"
        fi
    fi
else
    echo "[vLLM] Single H200 GPU Mode (141 GB VRAM):"
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_120B_PATH" \
        --port "$MODEL1_PORT" \
        --served-model-name "Qwen2.5-14B-GPU0" "Qwen2.5-14B-GPU1" "openai/gpt-oss-120b" "meta-llama/Llama-3.3-70B-Instruct" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.80 \
        --max-num-seqs 64 \
        --enforce-eager \
        --trust-remote-code \
        > "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log" 2>&1 &
    PID_M1=$!
    wait_for_ready "$MODEL1_PORT" "gpt-oss-120b" 180 "$PID_M1" "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log"
    export MODEL2_PORT="$MODEL1_PORT"
fi

echo "[PREFLIGHT] Active endpoints: MODEL1_PORT=$MODEL1_PORT, MODEL2_PORT=$MODEL2_PORT, EMBEDDING_PORT=$EMBEDDING_PORT"
curl -s "http://localhost:${MODEL1_PORT}/v1/models" | head -c 400; echo ""
curl -s "http://localhost:${MODEL2_PORT}/v1/models" | head -c 400; echo ""

# Setup Datasets
if [ ! -d "cuad-data" ]; then
    for cand in "/scratch/hpc4ai/palimpzest/otimizacao-abacus/palimpzest/abacus-research/cuad-data" "/mnt/scratch/global/palimpzest/abacus-research/cuad-data" "/app/abacus-research/cuad-data"; do
        if [ -d "$cand" ]; then ln -s "$cand" cuad-data; break; fi
    done
    [ ! -d "cuad-data" ] && python setup_cuad_data.py
fi

if [ -d "$SCRATCH_ROOT/.chroma-biodex" ]; then
    echo "[DATA] Reusing cached 768-d Nomic .chroma-biodex from $SCRATCH_ROOT/.chroma-biodex"
    rm -rf .chroma-biodex && ln -s "$SCRATCH_ROOT/.chroma-biodex" .chroma-biodex
else
    echo "[DATA] Building 768-d Nomic .chroma-biodex in $SCRATCH_ROOT/.chroma-biodex ..."
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

run_cuad_pair() {
    local tag=$1 budget=$2 k=$3 j=$4 val=$5 test_n=$6 seed=$7 z_val=$8 priors_mode=${9:-default}
    export CUAD_TEST_SAMPLES="$test_n"
    for mode in baseline es; do
        unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP
        export TIMEOUT_MODE=none
        if [ "$mode" = "es" ]; then
            export EARLY_STOP=1 ES_Z="$z_val" ES_MIN_UPDATES="3" ES_N_MIN="4"
        fi
        local exp="axia-cuad-${tag}-${mode}"
        if [ -f "opt-profiling-data/${exp}-metrics.json" ] && [ -z "$FORCE" ]; then
            echo "[SKIP] $exp already completed"
            sync_exp "$exp"
            continue
        fi
        export MAB_TRAJECTORY_LOG="opt-profiling-data/${exp}-trajectory.jsonl"
        rm -f "$MAB_TRAJECTORY_LOG"
        echo "=== [CUAD] $exp (B=$budget, k=$k, j=$j, val=$val, test=$test_n, seed=$seed, z=$z_val, priors=$priors_mode) ==="
        python cuad-demo.py --k "$k" --j "$j" --sample-budget "$budget" --seed "$seed" \
            --val-examples "$val" --num-contracts "$test_n" \
            --priors-file cuad-priors.json --priors-mode "$priors_mode" \
            --policy maxquality --optimizer-strategy pareto \
            --exp-name "$exp" --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
            --progress 2>&1 | tee "$LOG_DIR/${exp}.log"
        sync_exp "$exp"
    done
}

run_biodex_pair() {
    local tag=$1 budget=$2 k=$3 j=$4 val=$5 test_n=$6 seed=$7 z_val=$8 priors_mode=${9:-default}
    export BIODEX_TEST_SAMPLES="$test_n"
    for mode in baseline es; do
        unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP
        export TIMEOUT_MODE=none
        if [ "$mode" = "es" ]; then
            export EARLY_STOP=1 ES_Z="$z_val" ES_MIN_UPDATES="3" ES_N_MIN="4"
        fi
        local exp="axia-biodex-${tag}-${mode}"
        if [ -f "opt-profiling-data/${exp}-metrics.json" ] && [ -z "$FORCE" ]; then
            echo "[SKIP] $exp already completed"
            sync_exp "$exp"
            continue
        fi
        export MAB_TRAJECTORY_LOG="opt-profiling-data/${exp}-trajectory.jsonl"
        rm -f "$MAB_TRAJECTORY_LOG"
        echo "=== [BIODEX] $exp (B=$budget, k=$k, j=$j, val=$val, test=$test_n, seed=$seed, z=$z_val, priors=$priors_mode) ==="
        python biodex-demo.py --k "$k" --j "$j" --sample-budget "$budget" --seed "$seed" \
            --val-examples "$val" \
            --priors-file biodex-priors.json --priors-mode "$priors_mode" \
            --policy maxquality \
            --exp-name "$exp" --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
            --progress 2>&1 | tee "$LOG_DIR/${exp}.log"
        sync_exp "$exp"
    done
}

# =============================================================================
# 0. FAST PILOT VERIFICATION (Runs in ~12 min if not already cached in Scratch)
# =============================================================================
run_cuad_pair   "pilot" 40 4 2 10 10 0 "1.645" "default"
run_biodex_pair "pilot" 90 4 2 10 20 0 "1.645" "default"

# =============================================================================
# 1. MULTI-SEED CORE CAMPAIGN (Seeds 0, 1, 2)
# =============================================================================
for S in 0 1 2; do
    run_cuad_pair   "full-s${S}" 100 5 4 20 50  "$S" "1.645" "default"
    run_biodex_pair "full-s${S}" 300 5 4 20 250 "$S" "1.645" "default"
done

# =============================================================================
# 2. SCALABILITY STRESS TESTS (Large Budget B=250/600, Wide Exploration k=10)
# =============================================================================
run_cuad_pair   "stress-b250" 250 5  4 25 50  0 "1.645" "default"
run_biodex_pair "stress-b600" 600 5  4 25 250 0 "1.645" "default"
run_cuad_pair   "stress-k10"  150 10 4 20 50  0 "1.645" "default"
run_biodex_pair "stress-k10"  450 10 4 20 250 0 "1.645" "default"

# =============================================================================
# 3. ADVERSARIAL & COLD-START PRIORS STRESS TESTS (none, reversed)
# =============================================================================
run_cuad_pair   "priors-none"     100 5 4 20 50  0 "1.645" "none"
run_cuad_pair   "priors-reversed" 100 5 4 20 50  0 "1.645" "reversed"
run_biodex_pair "priors-none"     300 5 4 20 250 0 "1.645" "none"
run_biodex_pair "priors-reversed" 300 5 4 20 250 0 "1.645" "reversed"

# =============================================================================
# 4. CONFIDENCE THRESHOLD SENSITIVITY (z=1.282 [90%] vs. z=2.326 [99%])
# =============================================================================
run_cuad_pair   "sens-z128" 100 5 4 20 50  0 "1.282" "default"
run_cuad_pair   "sens-z233" 100 5 4 20 50  0 "2.326" "default"
run_biodex_pair "sens-z128" 300 5 4 20 250 0 "1.282" "default"
run_biodex_pair "sens-z233" 300 5 4 20 250 0 "2.326" "default"

echo "[AXIA FULL STRESS SUITE] Completed all experiments!"
