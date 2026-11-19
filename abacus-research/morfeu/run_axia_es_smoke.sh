#!/bin/bash
set -e

# =============================================================================
# Axia (NVIDIA H200) Pilot Verification Test: Baseline (0s) vs. Early Stop (ES)
# Models:
#   - Port 8005 (Qwen2.5-14B-GPU0 alias): meta-llama/Llama-3.3-70B-Instruct
#   - Port 8006 (Qwen2.5-14B-GPU1 alias): openai/gpt-oss-120b (fallback: Qwen2.5-32B / Llama-3.1-8B)
#   - Port 8010 (Embedding server):       nomic-ai/nomic-embed-text-v1
# Datasets:
#   1. CUAD Pilot   (B=40, k=4, j=2, val=10, test=10, Seed 0): 0s vs ES
#   2. BioDex Pilot (B=90, k=4, j=2, val=10, test=20, Seed 0): 0s vs ES
# =============================================================================

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-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}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export OPENAI_API_KEY="local-vllm"
export DISABLE_CACHE=1
export DISABLE_MID_EXECUTION_PRUNING=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Inject our patched Palimpzest (with Efficacy + Futility Early Stopping & Dynamic Timeout)
export PYTHONPATH="$ROOT_DIR/patched_src:$PYTHONPATH"

SCRATCH_ROOT="/scratch/hpc4ai/amarante_abacus_es"
LOG_DIR="$ROOT_DIR/var/logs"
mkdir -p "$LOG_DIR" "$ROOT_DIR/opt-profiling-data" "$SCRATCH_ROOT/opt-profiling-data" "$SCRATCH_ROOT/logs"

echo "=========================================================================="
echo " [AXIA PILOT] Starting Verification Suite (Llama-3.3-70B + gpt-oss-120b)"
echo " Working Dir : $ROOT_DIR"
echo " Scratch Dir : $SCRATCH_ROOT"
echo " PYTHONPATH  : $PYTHONPATH"
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

# Locate models on Axia /scratch
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
            [ -f "$log_file" ] && tail -n 40 "$log_file"
            return 1
        fi
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "[SUCCESS] $name is healthy on port $port after $((i * 5))s!"
            return 0
        fi
        sleep 5
    done
    echo "[WARN] Timeout waiting for $name on port $port"
    [ -f "$log_file" ] && tail -n 40 "$log_file"
    return 1
}

PID_EMB=""
PID_M1=""
PID_M2=""

stop_all_vllm() {
    echo "[vLLM] Cleaning up vLLM servers..."
    kill $PID_EMB $PID_M1 $PID_M2 2>/dev/null || true
    fuser -k ${EMBEDDING_PORT}/tcp ${MODEL1_PORT}/tcp ${MODEL2_PORT}/tcp 2>/dev/null || true
    sleep 3
}
trap stop_all_vllm EXIT INT TERM

# 1. Start Embedding Server (nomic-embed-text-v1) on Port 8010 (GPU 0)
echo "[vLLM] Starting nomic-ai/nomic-embed-text-v1 on port $EMBEDDING_PORT (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port "$EMBEDDING_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.04 \
    --max-model-len 2048 \
    > "$LOG_DIR/vllm_embed_${EMBEDDING_PORT}.log" 2>&1 &
PID_EMB=$!
wait_for_ready "$EMBEDDING_PORT" "Nomic-Embed" 90 "$PID_EMB" "$LOG_DIR/vllm_embed_${EMBEDDING_PORT}.log"

# 2. Start Large LLM Servers across available H200 GPUs
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "[vLLM] Multi-GPU H200 Mode ($NUM_GPUS GPUs):"
    echo "  -> Launching Llama-3.3-70B-Instruct on Port $MODEL1_PORT (TP=2 across GPUs 0,1, util=0.52)..."
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_70B_PATH" \
        --port "$MODEL1_PORT" \
        --dtype bfloat16 \
        --served-model-name "Qwen2.5-14B-GPU0" "meta-llama/Llama-3.3-70B-Instruct" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.52 \
        --tensor-parallel-size 2 \
        --max-num-seqs 64 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log" 2>&1 &
    PID_M1=$!
    wait_for_ready "$MODEL1_PORT" "Llama-3.3-70B (Port $MODEL1_PORT)" 240 "$PID_M1" "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log"

    # Try launching gpt-oss-120b on Port 8006 (TP=2 across GPUs 0,1, util=0.36)
    SECOND_MODEL_OK=0
    if [ -d "$MODEL_120B_PATH" ]; then
        echo "  -> Launching openai/gpt-oss-120b on Port $MODEL2_PORT (TP=2 across GPUs 0,1, util=0.36)..."
        CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_120B_PATH" \
            --port "$MODEL2_PORT" \
            --served-model-name "Qwen2.5-14B-GPU1" "openai/gpt-oss-120b" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.36 \
            --tensor-parallel-size 2 \
            --max-num-seqs 64 \
            --trust-remote-code \
            > "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log" 2>&1 &
        PID_M2=$!
        if wait_for_ready "$MODEL2_PORT" "gpt-oss-120b (Port $MODEL2_PORT)" 180 "$PID_M2" "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log"; then
            SECOND_MODEL_OK=1
        else
            echo "[WARN] gpt-oss-120b could not start on shared 2xH200; falling back to Llama-3.1-8B-Instruct on Port $MODEL2_PORT (GPU 1)..."
            kill $PID_M2 2>/dev/null || true
            fuser -k ${MODEL2_PORT}/tcp 2>/dev/null || true
            sleep 3
        fi
    fi

    if [ "$SECOND_MODEL_OK" -eq 0 ]; then
        echo "  -> Launching Llama-3.1-8B-Instruct on Port $MODEL2_PORT (GPU 1, util=0.30)..."
        CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_8B_PATH" \
            --port "$MODEL2_PORT" \
            --dtype bfloat16 \
            --served-model-name "Qwen2.5-14B-GPU1" "meta-llama/Llama-3.1-8B-Instruct" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.30 \
            --max-num-seqs 64 \
            --trust-remote-code \
            > "$LOG_DIR/vllm_m2_fallback_${MODEL2_PORT}.log" 2>&1 &
        PID_M2=$!
        wait_for_ready "$MODEL2_PORT" "Llama-3.1-8B (Port $MODEL2_PORT)" 180 "$PID_M2" "$LOG_DIR/vllm_m2_fallback_${MODEL2_PORT}.log"
    fi
else
    echo "[vLLM] Single H200 GPU Mode (141 GB VRAM):"
    # Try gpt-oss-120b on GPU 0 (util 0.58) + Llama-3.1-8B on GPU 0 (util 0.30)
    M1_OK=0
    if [ -d "$MODEL_120B_PATH" ]; then
        echo "  -> Trying openai/gpt-oss-120b on Port $MODEL1_PORT (GPU 0, util=0.58)..."
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
        if wait_for_ready "$MODEL1_PORT" "gpt-oss-120b (Port $MODEL1_PORT)" 180 "$PID_M1" "$LOG_DIR/vllm_m1_${MODEL1_PORT}.log"; then
            M1_OK=1
        else
            echo "[WARN] gpt-oss-120b did not start on single GPU; falling back to Qwen2.5-14B/Llama-3.1-8B on Port $MODEL1_PORT..."
            kill $PID_M1 2>/dev/null || true
            fuser -k ${MODEL1_PORT}/tcp 2>/dev/null || true
            sleep 3
        fi
    fi

    if [ "$M1_OK" -eq 0 ]; then
        FALLBACK_M1="Qwen/Qwen2.5-14B-Instruct-AWQ"
        [ -d "$MODEL_8B_PATH" ] && FALLBACK_M1="$MODEL_8B_PATH"
        CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
            --model "$FALLBACK_M1" \
            --port "$MODEL1_PORT" \
            --dtype bfloat16 \
            --served-model-name "Qwen2.5-14B-GPU0" "meta-llama/Llama-3.1-8B-Instruct" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.45 \
            --max-num-seqs 64 \
            --trust-remote-code \
            > "$LOG_DIR/vllm_m1_fallback_${MODEL1_PORT}.log" 2>&1 &
        PID_M1=$!
        wait_for_ready "$MODEL1_PORT" "Model 1 Fallback (Port $MODEL1_PORT)" 180 "$PID_M1" "$LOG_DIR/vllm_m1_fallback_${MODEL1_PORT}.log"
    fi

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
    wait_for_ready "$MODEL2_PORT" "Model 2 (Port $MODEL2_PORT)" 180 "$PID_M2" "$LOG_DIR/vllm_m2_${MODEL2_PORT}.log"
fi

echo "[PREFLIGHT] Checking active models on ports $EMBEDDING_PORT, $MODEL1_PORT, $MODEL2_PORT:"
curl -s "http://localhost:${MODEL1_PORT}/v1/models" | head -c 500; echo ""
curl -s "http://localhost:${MODEL2_PORT}/v1/models" | head -c 500; echo ""

# =============================================================================
# Setup Datasets (CUAD & BioDex ChromaDB 768-d Nomic Index)
# =============================================================================
if [ ! -d "cuad-data" ]; then
    for cand in "/scratch/hpc4ai/palimpzest/otimizacao-abacus/palimpzest/abacus-research/cuad-data" "/mnt/scratch/global/palimpzest/abacus-research/cuad-data" "/app/abacus-research/cuad-data"; do
        if [ -d "$cand" ]; then
            echo "[DATA] Symlinking cuad-data from $cand"
            ln -s "$cand" cuad-data
            break
        fi
    done
    [ ! -d "cuad-data" ] && python setup_cuad_data.py
fi

if [ -d "$SCRATCH_ROOT/.chroma-biodex" ]; then
    echo "[DATA] Using cached 768-d Nomic .chroma-biodex from $SCRATCH_ROOT/.chroma-biodex"
    rm -rf .chroma-biodex
    ln -s "$SCRATCH_ROOT/.chroma-biodex" .chroma-biodex
else
    echo "[DATA] Building 768-d Nomic .chroma-biodex in $SCRATCH_ROOT/.chroma-biodex ..."
    export BIODEX_CHROMA_DIR="$SCRATCH_ROOT/.chroma-biodex"
    python morfeu/build_biodex_index.py
    rm -rf .chroma-biodex
    ln -s "$SCRATCH_ROOT/.chroma-biodex" .chroma-biodex
fi

# =============================================================================
# PART 1: CUAD Pilot Verification (Baseline Abacus vs. Nosso Método ES)
# B=40, k=4, j=2, val=10, test=10, Seed 0
# =============================================================================
export CUAD_MAX_WORKERS=8
export CUAD_TEST_SAMPLES=10

for MODE in baseline es; do
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP
    export TIMEOUT_MODE=none
    if [ "$MODE" = "es" ]; then
        export EARLY_STOP=1
        export ES_Z="1.645" ES_MIN_UPDATES="3" ES_N_MIN="4"
    fi
    EXP="axia-pilot-cuad-${MODE}"
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${EXP}-trajectory.jsonl"
    rm -f "$MAB_TRAJECTORY_LOG"

    echo ""
    echo "=========================================================================="
    echo " [1/2 CUAD PILOT] Running $EXP (early_stop=${EARLY_STOP:-0}, B=40, k=4, j=2, val=10, test=10)"
    echo "=========================================================================="
    python cuad-demo.py \
        --k 4 --j 2 --sample-budget 40 --seed 0 \
        --val-examples 10 --num-contracts 10 \
        --policy maxquality --optimizer-strategy pareto \
        --exp-name "$EXP" \
        --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
        --progress 2>&1 | tee "$LOG_DIR/${EXP}.log"

    cp -f opt-profiling-data/${EXP}* "$SCRATCH_ROOT/opt-profiling-data/" 2>/dev/null || true
    cp -f "$LOG_DIR/${EXP}.log" "$SCRATCH_ROOT/logs/" 2>/dev/null || true
done

# =============================================================================
# PART 2: BioDex Pilot Verification (Baseline Abacus vs. Nosso Método ES)
# B=90 (30/op), k=4, j=2, val=10, test=20, Seed 0
# =============================================================================
export BIODEX_MAX_WORKERS=8
export BIODEX_TEST_SAMPLES=20

for MODE in baseline es; do
    unset SAMPLE_TIMEOUT QUALITY_PRUNE_RHO EARLY_STOP
    export TIMEOUT_MODE=none
    if [ "$MODE" = "es" ]; then
        export EARLY_STOP=1
        export ES_Z="1.645" ES_MIN_UPDATES="3" ES_N_MIN="4"
    fi
    EXP="axia-pilot-biodex-${MODE}"
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${EXP}-trajectory.jsonl"
    rm -f "$MAB_TRAJECTORY_LOG"

    echo ""
    echo "=========================================================================="
    echo " [2/2 BIODEX PILOT] Running $EXP (early_stop=${EARLY_STOP:-0}, B=90, k=4, j=2, val=10, test=20)"
    echo "=========================================================================="
    python biodex-demo.py \
        --k 4 --j 2 --sample-budget 90 --seed 0 \
        --val-examples 10 \
        --policy maxquality \
        --exp-name "$EXP" \
        --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" \
        --progress 2>&1 | tee "$LOG_DIR/${EXP}.log"

    cp -f opt-profiling-data/${EXP}* "$SCRATCH_ROOT/opt-profiling-data/" 2>/dev/null || true
    cp -f "$LOG_DIR/${EXP}.log" "$SCRATCH_ROOT/logs/" 2>/dev/null || true
done

# =============================================================================
# PART 3: Generate Summary Comparison Report
# =============================================================================
python - <<'PY' | tee "opt-profiling-data/axia-pilot-summary.txt"
import json, glob, os

print("\n" + "="*95)
print(" AXIA PILOT VERIFICATION SUMMARY: BASELINE ABACUS vs. NOSSO METODO (ES)")
print("="*95)
summary = {}
for exp in ["axia-pilot-cuad-baseline", "axia-pilot-cuad-es", "axia-pilot-biodex-baseline", "axia-pilot-biodex-es"]:
    mfile = f"opt-profiling-data/{exp}-metrics.json"
    tfile = f"opt-profiling-data/{exp}-trajectory.jsonl"
    if not os.path.exists(mfile):
        print(f" [MISSING] {mfile}")
        continue
    m = json.load(open(mfile))
    rounds, samples, stop_reasons = 0, 0, []
    if os.path.exists(tfile):
        lines = [json.loads(line) for line in open(tfile) if line.strip()]
        rounds = max((r.get("round", 0) for r in lines), default=0)
        samples = sum(1 for r in lines if r.get("event") == "sample")
        stop_reasons = [r.get("reason") for r in lines if r.get("event") == "early_stop"]
    summary[exp] = {
        "metrics": m,
        "rounds": rounds,
        "samples": samples,
        "stop_reasons": stop_reasons,
    }
    q_str = f"F1={m.get('f1', 0):.4f} (P={m.get('precision', 0):.4f}, R={m.get('recall', 0):.4f})" if "f1" in m else f"rp@5={m.get('rp@k', 0):.4f}"
    print(f" {exp:28s} | {q_str:34s} | OptTime={m.get('optimization_time', 0):6.1f}s | TotTime={m.get('total_execution_time', 0):6.1f}s | Rounds={rounds:2d} | Samples={samples:3d} | Stop={stop_reasons}")

with open("opt-profiling-data/axia-pilot-summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("="*95 + "\n")
PY

cp -f opt-profiling-data/axia-pilot-summary.* "$SCRATCH_ROOT/opt-profiling-data/" 2>/dev/null || true
echo "[AXIA PILOT] All pilot verification tests finished successfully!"
