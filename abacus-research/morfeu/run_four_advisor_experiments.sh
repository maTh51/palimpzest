#!/bin/bash
# Campanha experimental dedicada aos 4 estudos solicitados pelo usuário:
#   1) Consistência e Replicabilidade do Fluxo (morfeu-biodex-consist-es-rep1 e rep2, Seed 0, Budget 300)
#   2) Demonstração Empírica de Parada por Eficácia vs. Futilidade (morfeu-biodex-es-efficacy-b900)
# Todos os perfis (-metrics.json, -profiling.json, -records.json, -output.csv, -trajectory.jsonl, -timeouts.csv)
# são salvos em opt-profiling-data/ e os logs de console em morfeu/logs/.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_morfeu.sh"
cd "$SCRIPT_DIR/.."
mkdir -p opt-profiling-data "$SCRIPT_DIR/logs"

export PATH="/home/amarante/.conda/envs/kunumi/bin:$PATH"
export BIODEX_MAX_WORKERS="${BIODEX_MAX_WORKERS:-8}"
export DISABLE_MID_EXECUTION_PRUNING=1
# Para validar consistência da Fase 1 (100% do orçamento B=300 e VAL=20) e medir RP@5 em subconjunto pareado rápido (30 docs):
export BIODEX_TEST_SAMPLES="${BIODEX_TEST_SAMPLES:-30}"

run_exp() {
    local EXP="$1"
    local SEED="$2"
    local BUDGET="$3"
    local ES_FLAG="$4"
    local ES_Z_VAL="$5"
    local ES_MIN_UPD="$6"

    if [ -f "opt-profiling-data/${EXP}-metrics.json" ] && [ -z "$FORCE" ]; then
        echo "[skip] $EXP já possui perfis salvos em opt-profiling-data/"
        return 0
    fi

    export TIMEOUT_MODE=none
    export EARLY_STOP="$ES_FLAG"
    export ES_Z="$ES_Z_VAL"
    export ES_MIN_UPDATES="$ES_MIN_UPD"
    export ES_N_MIN=4
    export MAB_TRAJECTORY_LOG="opt-profiling-data/${EXP}-trajectory.jsonl"
    export TIMEOUT_LOG_CSV="opt-profiling-data/${EXP}-timeouts.csv"
    rm -f "$MAB_TRAJECTORY_LOG" "$TIMEOUT_LOG_CSV"

    echo "=== Iniciando $EXP (seed=$SEED budget=$BUDGET early_stop=$ES_FLAG z=$ES_Z_VAL min_upd=$ES_MIN_UPD test=$BIODEX_TEST_SAMPLES) $(date) ==="
    python biodex-demo.py --k 5 --j 4 --sample-budget "$BUDGET" --seed "$SEED" \
        --val-examples 20 --policy maxquality --exp-name "$EXP" \
        --model1-port "$MODEL1_PORT" --model2-port "$MODEL2_PORT" --progress \
        2>&1 | tee "$SCRIPT_DIR/logs/${EXP}.log"
    echo "=== Concluído $EXP $(date) ==="
}

# 1. Duas réplicas de consistência com parâmetros idênticos ao morfeu-biodex-full-timeout-es (Seed 0, Budget 300, z=1.645)
run_exp "morfeu-biodex-consist-es-rep1" 0 300 1 1.645 3
run_exp "morfeu-biodex-consist-es-rep2" 0 300 1 1.645 3

# 2. Execução com orçamento estendido (B=900, z=0.85) para capturar parada por EFICÁCIA (Δ >= SE_atual) em perfil real
run_exp "morfeu-biodex-es-efficacy-b900" 0 900 1 0.85 3

echo "[four_advisor_experiments] Todos os experimentos concluídos e salvos em opt-profiling-data/!"
