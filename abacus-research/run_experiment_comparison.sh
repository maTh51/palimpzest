#!/bin/bash
set -e

# Run directory
cd /home/matheuspimenta/kunumi/palimpzest/abacus-research

echo "======================================================================"
echo "STARTING ABACUS PRUNING EXPERIMENT COMPARISON"
echo "======================================================================"

# 1. RUN 1: Pareto Strategy (with Pruning Enabled)
echo "--> Running Pareto Strategy with Pruning..."
conda run -n kunumi python cuad-demo.py --k 5 --j 2 --sample-budget 20 --seed 0 --exp-name cuad-pareto-pruned --optimizer-strategy pareto --num-contracts 5

echo ""
echo "Waiting 200 seconds for the GPU queue and vLLM cache to clear..."
sleep 200

echo ""
echo "----------------------------------------------------------------------"
echo "--> Running Greedy Strategy with Pruning..."
# 2. RUN 2: Greedy Strategy (with Pruning Enabled)
conda run -n kunumi python cuad-demo.py --k 5 --j 2 --sample-budget 20 --seed 0 --exp-name cuad-greedy-pruned --optimizer-strategy greedy --num-contracts 5

echo ""
echo "Waiting 200 seconds for the GPU queue and vLLM cache to clear..."
sleep 200

echo ""
echo "----------------------------------------------------------------------"
echo "--> Running Baseline (No Pruning)..."
# 3. RUN 3: Baseline (with Pruning Disabled via env variable)
DISABLE_PRUNING=1 conda run -n kunumi python cuad-demo.py --k 5 --j 2 --sample-budget 20 --seed 0 --exp-name cuad-baseline-nopruning --optimizer-strategy pareto --num-contracts 5

echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE. Check the terminal outputs above to compare:"
echo "1. Optimization Time vs. Plan Exec. Time"
echo "2. Precision, Recall, and F1 Quality Scores"
echo "3. Total Execution Costs"
echo "======================================================================"
