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
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

EMBEDDING_PORT=8010
MODEL1_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"
MODEL1_PORT=${MODEL1_PORT:-8005}
MODEL2_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"
MODEL2_PORT=${MODEL2_PORT:-8006}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../log}"
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
    echo "[vLLM] Starting $MODEL2_NAME on port $MODEL1_PORT (Optimized for NVIDIA H200 with 65k Context)..."
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL2_NAME" \
        --served-model-name "Qwen2.5-14B-GPU0" "Qwen2.5-14B-GPU1" \
        --port "$MODEL1_PORT" \
        --max-model-len 65536 \
        --max-num-seqs 256 \
        --max-num-batched-tokens 65536 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code &
    PID1=$!
    wait_for_ready "$MODEL1_PORT"

    echo "[vLLM] Server ready on port $MODEL1_PORT."
}

stop_vllm() {
    echo "[vLLM] Stopping vLLM server..."
    kill $PID1 2>/dev/null || true
    fuser -k ${MODEL1_PORT}/tcp 2>/dev/null || true
    fuser -k ${MODEL2_PORT}/tcp 2>/dev/null || true
    sleep 3
    echo "[vLLM] Server stopped."
}

# Trap script exit/error/SIGINT/SIGTERM to guarantee cleanup
trap 'stop_vllm' EXIT INT TERM

# ---- Main loop --------------------------------------------------------------
for TIMEOUT in "${TIMEOUTS[@]}"; do
    export DISABLE_MID_EXECUTION_PRUNING=1
    export DISABLE_CACHE=1

    if [ "$TIMEOUT" -eq 0 ]; then
        EXP_NAME="timeout-baseline-no-timeout"
        unset SAMPLE_TIMEOUT
    else
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
        --sample-budget 300 \
        --seed 0 \
        --num-contracts 50 \
        --policy maxquality \
        --exp-name "${EXP_NAME}" \
        --optimizer-strategy pareto \
        --model1-port "${MODEL1_PORT:-8005}" \
        --model2-port "${MODEL2_PORT:-8005}"

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
echo " Generating timeout-specific presentation figures..."
echo "========================================================"
python analyze_operators_and_pareto.py