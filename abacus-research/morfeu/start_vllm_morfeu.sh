#!/bin/bash
# Start the embedding server and the Qwen 14B server on ONE Morfeu GPU.
# Usage:  bash morfeu/start_vllm_morfeu.sh        (run from abacus-research/)
#         bash morfeu/start_vllm_morfeu.sh stop
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_morfeu.sh"
LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"

stop_all() {
    fuser -k "${EMBEDDING_PORT}/tcp" 2>/dev/null || true
    fuser -k "${MODEL1_PORT}/tcp" 2>/dev/null || true
    fuser -k "${MODEL2_PORT}/tcp" 2>/dev/null || true
    sleep 3
    # vLLM's EngineCore child can survive its API server and keep holding GPU memory.
    # Kill leftovers owned by this user across both GPUs.
    for G in 0 1; do
        local UUID; UUID=$(nvidia-smi --id="$G" --query-gpu=uuid --format=csv,noheader)
        for PID in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2==u{print $1}'); do
            if [ "$(ps -o user= -p "$PID" 2>/dev/null)" = "$USER" ]; then kill "$PID" 2>/dev/null || true; fi
        done
    done
    sleep 2
    echo "[vLLM] stopped ports $EMBEDDING_PORT, $MODEL1_PORT, $MODEL2_PORT"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
if [ "$1" = "stop" ]; then stop_all; exit 0; fi

wait_for_ready() {
    local PORT=$1 NAME=$2 PID=$3
    echo "[vLLM] waiting for $NAME on port $PORT (pid=$PID) ..."
    for _ in $(seq 1 180); do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" | grep -q 200; then
            echo "[vLLM] $NAME ready"; return 0
        fi
        if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
            echo "[vLLM] ERROR: $NAME (pid=$PID) exited prematurely! Check $LOG_DIR"
            exit 1
        fi
        sleep 5
    done
    echo "[vLLM] $NAME did not start; see $LOG_DIR"; exit 1
}

echo "[vLLM] Initial GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 1) GPU 0: Qwen 14B Model 1
CUDA_VISIBLE_DEVICES=0 setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL" --served-model-name Qwen2.5-14B-GPU0 \
    --port "$MODEL1_PORT" --max-model-len "$LLM_MAX_LEN" --dtype float16 \
    --gpu-memory-utilization "$GPU0_UTIL" --max-num-seqs 64 --trust-remote-code \
    > "$LOG_DIR/vllm_gpu0.log" 2>&1 < /dev/null &
PID_GPU0=$!
wait_for_ready "$MODEL1_PORT" "Qwen 14B (GPU 0)" "$PID_GPU0"

# 2) GPU 1: Embedder
CUDA_VISIBLE_DEVICES=1 setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL" --runner pooling --convert embed \
    --port "$EMBEDDING_PORT" --trust-remote-code \
    --gpu-memory-utilization "$EMBED_GPU_UTIL" --max-model-len 2048 --enforce-eager \
    > "$LOG_DIR/vllm_embed.log" 2>&1 < /dev/null &
PID_EMB=$!
wait_for_ready "$EMBEDDING_PORT" "Embedder (GPU 1)" "$PID_EMB"

# 3) GPU 1: Qwen 14B Model 2
CUDA_VISIBLE_DEVICES=1 setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL" --served-model-name Qwen2.5-14B-GPU1 \
    --port "$MODEL2_PORT" --max-model-len "$LLM_MAX_LEN" --dtype float16 \
    --gpu-memory-utilization "$GPU1_UTIL" --max-num-seqs 64 --trust-remote-code \
    > "$LOG_DIR/vllm_gpu1.log" 2>&1 < /dev/null &
PID_GPU1=$!
wait_for_ready "$MODEL2_PORT" "Qwen 14B (GPU 1)" "$PID_GPU1"

echo "[vLLM] All 3 servers up across GPU 0 & GPU 1! Stop with: bash morfeu/start_vllm_morfeu.sh stop"
