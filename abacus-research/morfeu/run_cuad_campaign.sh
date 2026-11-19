#!/bin/bash
# Complete CUAD Experimental Campaign on Morfeu:
#   1) Smoke test (0s vs es)
#   2) Full Multi-Seed Campaign (Seeds 0, 1, 2: Baseline 0s vs ES)
#   3) Consistency Replicas (Seed 0: rep1, rep2)
# Saves all profiles (-metrics.json, -profiling.json, -stats.json, -output.csv,
# -records.json, -trajectory.jsonl, -timeouts.csv) in opt-profiling-data/.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo ">>> [1/4] Starting vLLM servers on Morfeu..."
bash morfeu/start_vllm_morfeu.sh

echo ">>> [2/4] Smoke test on CUAD (0s vs es)..."
TIMEOUTS="0 es" bash morfeu/run_cuad_morfeu.sh smoke

echo ">>> [3/4] Full Multi-Seed Campaign on CUAD (Seeds 0, 1, 2: 0s vs es)..."
for S in 0 1 2; do
    SEED=$S TIMEOUTS="0 es" bash morfeu/run_cuad_morfeu.sh full
done

echo ">>> [4/4] Consistency Replicas on CUAD (Seed 0: rep1, rep2)..."
for REP in 1 2; do
    SEED=0 TIMEOUTS="es" CUAD_TEST_SAMPLES=15 EXP_OVERRIDE="morfeu-cuad-consist-es-rep${REP}" \
        bash morfeu/run_cuad_morfeu.sh full
done

echo ">>> All CUAD runs completed! Stopping vLLM servers..."
bash morfeu/start_vllm_morfeu.sh stop
