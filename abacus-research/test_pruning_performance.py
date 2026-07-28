import os
import time
import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.tasks import OptimizePhysicalExpression
from pydantic import Field
from palimpzest.core.lib.schemas import TextFile

class Email(TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender: str = Field(description="The email address of the sender")
    subject: str = Field(description="The subject of the email")


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

def run_optimization(plan, policy, strategy, disable_pruning=False):
    global calls_count, pruned_count
    calls_count = 0
    pruned_count = 0
    
    if disable_pruning:
        os.environ["DISABLE_PRUNING"] = "1"
    else:
        if "DISABLE_PRUNING" in os.environ:
            del os.environ["DISABLE_PRUNING"]
            
    from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
    cost_model = SampleBasedCostModel()
    optimizer = Optimizer(
        policy=policy,
        cost_model=cost_model,
        available_models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.LLAMA3_1_8B],
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
    # Construct logical plan
    enron_eval_tiny_data_path = "/home/matheuspimenta/kunumi/palimpzest/testdata/enron-eval-tiny"
    if not os.path.exists(enron_eval_tiny_data_path):
        enron_eval_tiny_data_path = "../testdata/enron-eval-tiny"
        
    emails = pz.core.data.iter_dataset.TextFileDataset(id="enron-eval-tiny", path=enron_eval_tiny_data_path)
    emails = emails.sem_add_columns(Email)
    emails = emails.sem_filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy"'
    )
    emails = emails.sem_filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    
    scenarios = [
        ("MinCost / Pareto", MinCost(), OptimizationStrategyType.PARETO),
        ("MaxQuality / Pareto", MaxQuality(), OptimizationStrategyType.PARETO),
        ("MinTime / Pareto", MinTime(), OptimizationStrategyType.PARETO),
        ("MinCost / Greedy", MinCost(), OptimizationStrategyType.GREEDY),
        ("MaxQuality / Greedy", MaxQuality(), OptimizationStrategyType.GREEDY),
    ]
    
    print("=" * 105)
    print(f"{'Scenario':<25} | {'Pruning':<8} | {'Opt. Time (s)':<13} | {'Calls':<8} | {'Pruned':<8} | {'Plans Found':<11}")
    print("=" * 105)
    
    for name, policy, strategy in scenarios:
        # Run without pruning
        res_no = run_optimization(emails, policy, strategy, disable_pruning=True)
        # Run with pruning
        res_with = run_optimization(emails, policy, strategy, disable_pruning=False)
        
        # Verify correctness (ensure both runs found the same plans)
        no_plans = [str(p) for p in res_no["plans"]]
        with_plans = [str(p) for p in res_with["plans"]]
        correct = no_plans == with_plans
        status = "CORRECT" if correct else "INCORRECT"
        
        print(f"{name:<25} | {'Disabled':<8} | {res_no['time']:<13.4f} | {res_no['calls']:<8} | {res_no['pruned']:<8} | {len(res_no['plans']):<11}")
        print(f"{name:<25} | {'Enabled':<8} | {res_with['time']:<13.4f} | {res_with['calls']:<8} | {res_with['pruned']:<8} | {len(res_with['plans']):<11} ({status})")
        
        # calculate reduction in calls
        if res_no['calls'] > 0:
            saved_calls = res_no['calls'] - res_with['calls']
            saved_pct = (saved_calls / res_no['calls']) * 100
            print(f"{'':<25} | Avoided {saved_calls} / {res_no['calls']} cost evaluations ({saved_pct:.1f}% search space pruned)")
        print("-" * 105)

if __name__ == "__main__":
    main()
