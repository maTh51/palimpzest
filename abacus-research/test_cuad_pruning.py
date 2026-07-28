import os
import sys
import time
import importlib
import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.tasks import OptimizePhysicalExpression
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel

# Global counters
calls_count = 0
pruned_count = 0

original_should_prune = OptimizePhysicalExpression._should_prune_plan_cost

def wrapped_should_prune(self, group, plan_cost, policy, optimizer_strategy):
    global calls_count, pruned_count
    calls_count += 1
    res = original_should_prune(self, group, plan_cost, policy, optimizer_strategy)
    if res:
        pruned_count += 1
    return res

OptimizePhysicalExpression._should_prune_plan_cost = wrapped_should_prune

def run_optimization(plan, policy, strategy, available_models, disable_pruning=False, epsilon="0.0", multiplier="1.0", margin="1.0"):
    global calls_count, pruned_count
    calls_count = 0
    pruned_count = 0
    
    if disable_pruning:
        os.environ["DISABLE_PRUNING"] = "1"
    else:
        if "DISABLE_PRUNING" in os.environ:
            del os.environ["DISABLE_PRUNING"]
            
    os.environ["PRUNING_EPSILON"] = str(epsilon)
    os.environ["PRUNING_CONSTRAINT_MULTIPLIER"] = str(multiplier)
    os.environ["PRUNING_BEAT_INCUMBENT_MARGIN"] = str(margin)
    
    cost_model = SampleBasedCostModel()
    optimizer = Optimizer(
        policy=policy,
        cost_model=cost_model,
        available_models=available_models,
        optimizer_strategy=strategy,
        verbose=False,
    )
    
    start_time = time.time()
    physical_plans = optimizer.optimize(plan)
    elapsed = time.time() - start_time
    
    return {
        "plans": physical_plans,
        "time": elapsed,
        "calls": calls_count,
        "pruned": pruned_count,
    }

def main():
    # Import CUADDataset and build_cuad_query from cuad-demo.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    cuad_demo = importlib.import_module("cuad-demo")
    CUADDataset = cuad_demo.CUADDataset
    build_cuad_query = cuad_demo.build_cuad_query

    print("Building CUAD query...")
    # Instantiate dataset and build query
    dataset = CUADDataset(split="test", num_contracts=5, seed=42)
    query = build_cuad_query(dataset, "one-convert")
    
    # Use the full set of candidate models to evaluate a large search space
    available_models = [
        Model.GPT_4o,
        Model.GPT_4o_MINI,
        Model.LLAMA3_1_8B,
        Model.LLAMA3_3_70B,
        Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    ]
    
    # Get configuration from environment or CLI if provided
    epsilon = os.getenv("EPSILON", "0.0")
    multiplier = os.getenv("MULTIPLIER", "1.0")
    margin = os.getenv("MARGIN", "1.0")
    
    print(f"Testing pruning with parameters: EPSILON={epsilon}, MULTIPLIER={multiplier}, MARGIN={margin}")
    
    scenarios = [
        ("CUAD: MinCost / Pareto", MinCost(), OptimizationStrategyType.PARETO),
        ("CUAD: MaxQuality / Pareto", MaxQuality(), OptimizationStrategyType.PARETO),
        ("CUAD: MinTime / Pareto", MinTime(), OptimizationStrategyType.PARETO),
        ("CUAD: MinCost / Greedy", MinCost(), OptimizationStrategyType.GREEDY),
        ("CUAD: MaxQuality / Greedy", MaxQuality(), OptimizationStrategyType.GREEDY),
    ]
    
    print("=" * 115)
    print(f"{'Scenario':<25} | {'Pruning Status':<18} | {'Opt. Time (s)':<13} | {'Calls':<8} | {'Pruned':<8} | {'Plans Found':<11}")
    print("=" * 115)
    
    for name, policy, strategy in scenarios:
        # Run without pruning
        res_no = run_optimization(query, policy, strategy, available_models, disable_pruning=True)
        # Run with pruning using active parameters
        res_with = run_optimization(
            query, policy, strategy, available_models,
            disable_pruning=False, epsilon=epsilon, multiplier=multiplier, margin=margin
        )
        
        # Verify correctness (ensure both runs found the same plans)
        no_plans = [str(p) for p in res_no["plans"]]
        with_plans = [str(p) for p in res_with["plans"]]
        correct = no_plans == with_plans
        status = "CORRECT" if correct else "DIFF PLAN"
        
        # Format printing
        prune_status_str = f"Enabled (e={epsilon})"
        print(f"{name:<25} | {'Disabled':<18} | {res_no['time']:<13.4f} | {res_no['calls']:<8} | {res_no['pruned']:<8} | {len(res_no['plans']):<11}")
        print(f"{name:<25} | {prune_status_str:<18} | {res_with['time']:<13.4f} | {res_with['calls']:<8} | {res_with['pruned']:<8} | {len(res_with['plans']):<11} ({status})")
        
        # calculate reduction in calls
        if res_no['calls'] > 0:
            pruning_pct = (res_with['pruned'] / res_no['calls']) * 100
            print(f"{'':<25} | Candidate pruning rate: {pruning_pct:.1f}% ({res_with['pruned']} / {res_no['calls']} candidates pruned)")
        print("-" * 115)

if __name__ == "__main__":
    main()
