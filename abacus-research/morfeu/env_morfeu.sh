# Shared environment for running Abacus/BioDex on Morfeu (single 24 GB GPU).
# Usage:  source morfeu/env_morfeu.sh
# Axia/H200 settings live in the original run_*.sh / polyaxon_*.yaml files and are NOT used here.

export GPU_ID="${GPU_ID:-1}"                      # GPU 0 is busy on Morfeu; override with GPU_ID=0 if free
export CUDA_VISIBLE_DEVICES="$GPU_ID"

export HF_HOME="${HF_HOME:-/home/amarante/.cache/hf_user}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# biodex-demo.py checks for an API key even though everything is served locally by vLLM
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Ports
export EMBEDDING_PORT="${EMBEDDING_PORT:-8010}"   # nomic embedder on GPU 1
export MODEL1_PORT="${MODEL1_PORT:-8005}"         # Qwen2.5-14B on GPU 0
export MODEL2_PORT="${MODEL2_PORT:-8006}"         # Qwen2.5-14B on GPU 1

# Models
export EMBED_MODEL="nomic-ai/nomic-embed-text-v1"
export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"

# Memory allocations:
# GPU 0 (24GB): Dedicated to Model 1 -> 0.84 util
# GPU 1 (24GB): Embedder (0.08 util) + Model 2 (0.80 util)
export GPU0_UTIL="${GPU0_UTIL:-0.84}"
export GPU1_UTIL="${GPU1_UTIL:-0.80}"
export EMBED_GPU_UTIL="${EMBED_GPU_UTIL:-0.08}"
export LLM_MAX_LEN="${LLM_MAX_LEN:-32768}"

# Index built with nomic embeddings (the shared/tarball indexes use 1536-d OpenAI embeddings)
export BIODEX_CHROMA_DIR="${BIODEX_CHROMA_DIR:-.chroma-biodex}"

export DISABLE_CACHE=1
