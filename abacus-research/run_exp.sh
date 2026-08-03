#!/bin/bash
set -e

# =============================================================================
# Experiment: Impact of SAMPLE_TIMEOUT on Sentinel Pruning
# =============================================================================
# For each timeout value:
#   1. Start vLLM servers (embedding + 2x Qwen 14B AWQ)
#   2. Run cuad-demo.py with that timeout
#   3. Kill all vLLM servers (clears request queue)
#
# Set DISABLE_MID_EXECUTION_PRUNING=1 to isolate timeout-only behavior
# (no mid-execution speculative Pareto pruning, only end-of-round Abacus pruning).
#
# Timeout values to test (in seconds). Use 0 for the baseline (no timeout).
# =============================================================================

TIMEOUTS=(0 30 60 120)

# ---- vLLM settings (mirrored from vllm2.sh) --------------------------------
export HF_HOME="/mnt/scratch/global/huggingface_cache/huggingface"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

EMBEDDING_PORT=8010
MODEL1_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"
MODEL1_PORT=8006
MODEL2_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"
MODEL2_PORT=8005

LOG_DIR="/mnt/scratch/matheuspimenta/palimpzest/abacus-research/var/logs"
mkdir -p "$LOG_DIR"

# ---- Helpers ----------------------------------------------------------------
wait_for_ready() {
    local PORT=$1
    echo "[INFO] Waiting for model on port $PORT to become ready..."
    until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
        sleep 5
    done
    echo "[INFO] Model on port $PORT is ready!"
}

start_vllm() {
    echo ""
    echo "[vLLM] Starting embedding server (port $EMBEDDING_PORT)..."
    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model nomic-ai/nomic-embed-text-v1 \
        --port $EMBEDDING_PORT \
        --trust-remote-code \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.05 &
    PID_EMBEDDING=$!
    wait_for_ready $EMBEDDING_PORT

    echo "[vLLM] Starting $MODEL1_NAME on port $MODEL1_PORT..."
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL1_NAME" \
        --served-model-name "Qwen2.5-14B-GPU1" \
        --port $MODEL1_PORT \
        --max-model-len 32000 \
        --gpu-memory-utilization 0.90 \
        --enforce-eager &
    PID1=$!
    wait_for_ready $MODEL1_PORT

    echo "[vLLM] Starting $MODEL2_NAME on port $MODEL2_PORT..."
    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL2_NAME" \
        --served-model-name "Qwen2.5-14B-GPU0" \
        --port $MODEL2_PORT \
        --max-model-len 8192 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 &
    PID2=$!
    wait_for_ready $MODEL2_PORT

    echo "[vLLM] All servers ready."
}

stop_vllm() {
    echo "[vLLM] Stopping all vLLM servers..."
    kill $PID_EMBEDDING $PID1 $PID2 2>/dev/null || true
    # Also kill any stray vllm processes on our ports
    fuser -k ${EMBEDDING_PORT}/tcp 2>/dev/null || true
    fuser -k ${MODEL1_PORT}/tcp    2>/dev/null || true
    fuser -k ${MODEL2_PORT}/tcp    2>/dev/null || true
    sleep 5
    echo "[vLLM] Servers stopped."
}

# Trap script exit/error/SIGINT/SIGTERM to guarantee cleanup
trap 'stop_vllm' EXIT INT TERM

# Set USE_ADAPTIVE=1 to run Strategy 1 Operator Self-Adaptive Timeout
USE_ADAPTIVE=${USE_ADAPTIVE:-1}

for TIMEOUT in "${TIMEOUTS[@]}"; do
    export DISABLE_MID_EXECUTION_PRUNING=1
    export DISABLE_CACHE=1

    if [ "$USE_ADAPTIVE" -eq 1 ]; then
        export ADAPTIVE_TIMEOUT=1
        export TIMEOUT_MULTIPLIER=${TIMEOUT_MULTIPLIER:-1.5}
        export TIMEOUT_MARGIN=${TIMEOUT_MARGIN:-10.0}
        export SAFETY_TIMEOUT=${SAFETY_TIMEOUT:-180.0}
        EXP_NAME="adaptive-timeout-${TIMEOUT_MULTIPLIER}x-${TIMEOUT_MARGIN}s"
    elif [ "$TIMEOUT" -eq 0 ]; then
        unset ADAPTIVE_TIMEOUT
        EXP_NAME="timeout-baseline-no-timeout"
        unset SAMPLE_TIMEOUT
    else
        unset ADAPTIVE_TIMEOUT
        EXP_NAME="timeout-pruning-${TIMEOUT}s"
        export SAMPLE_TIMEOUT="${TIMEOUT}"
    fi

    echo ""
    echo "======================================================"
    echo " Starting experiment : ${EXP_NAME}"
    echo " SAMPLE_TIMEOUT      : ${TIMEOUT}s  (0 = disabled)"
    echo "======================================================"

    start_vllm

    echo ""
    echo "[EXP] Running: ${EXP_NAME} ..."
    echo ""

    python cuad-demo.py \
        --k 5 \
        --j 4 \
        --sample-budget 100 \
        --seed 0 \
        --num-contracts 50 \
        --policy maxquality \
        --exp-name "${EXP_NAME}" \
        --optimizer-strategy pareto

    echo ""
    echo "[EXP] Finished: ${EXP_NAME}"

    stop_vllm

    echo ""
    echo "[EXP] Results saved to opt-profiling-data/${EXP_NAME}-{stats,metrics}.json"
    echo ""
done

echo "========================================================"
echo " All experiments finished."
echo " Results are in: opt-profiling-data/"
echo "========================================================"