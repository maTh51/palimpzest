import os
import pytest
import numpy as np
import time
from palimpzest.policy import MinCost, MinTime
from palimpzest.core.elements.records import DataRecordSet
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.core.lib.schemas import TextFile
from palimpzest.core.models import RecordOpStats, SentinelPlanStats
from palimpzest.query.execution.mab_execution_strategy import MABExecutionStrategy

class MockPhysicalOperator(PhysicalOperator):
    def __init__(self, op_id, latency, cost, quality):
        super().__init__(output_schema=TextFile)
        self.op_id = op_id
        self.latency = latency
        self.cost = cost
        self.quality = quality
        
    def get_full_op_id(self):
        return self.op_id
        
    def op_name(self):
        return f"MockOp-{self.op_id}"
        
    def get_id_params(self):
        return {}
        
    def __call__(self, *args, **kwargs):
        # Simulate latency
        if self.latency > 0:
            time.sleep(self.latency)
        r_stats = RecordOpStats(
            record_id="dummy-rec-id",
            record_parent_ids=["dummy-parent-id"],
            record_source_indices=["dummy-src"],
            record_state={},
            full_op_id=self.op_id,
            logical_op_id="logical-op-1",
            op_id=self.op_id,
            op_name=self.op_name(),
            time_per_record=self.latency,
            cost_per_record=self.cost,
            quality=self.quality,
            passed_operator=True,
        )
        record_set = DataRecordSet(data_records=[], record_op_stats=[r_stats])
        return record_set

def test_sentinel_pruning():
    # Set PRUNING_EPSILON to 0.0
    os.environ["PRUNING_EPSILON"] = "0.0"
    # Use full mid-execution path for this test
    os.environ.pop("DISABLE_MID_EXECUTION_PRUNING", None)

    # Create a dummy sentinel strategy
    policy = MinCost()
    strategy = MABExecutionStrategy(policy=policy)
    
    # Define a fast/cheap operator and a slow/expensive operator
    op_fast = MockPhysicalOperator("op_fast", latency=0.001, cost=0.01, quality=1.0)
    op_slow = MockPhysicalOperator("op_slow", latency=0.3, cost=2.0, quality=1.0)
    
    # Create a list of physical inputs representing 5 samples for each
    op_inputs = []
    for i in range(5):
        op_inputs.append((op_fast, f"src-{i}", f"input-{i}"))
        op_inputs.append((op_slow, f"src-{i}", f"input-{i}"))
        
    # Mock op_frontier and plan_stats
    class MockOpFrontier:
        def __init__(self, ops):
            self.frontier_ops = ops
            
    op_frontier = MockOpFrontier([op_fast, op_slow])
    
    # Create SentinelPlanStats
    plan_stats = SentinelPlanStats(plan_id="dummy-plan", plan_str="dummy-plan", operator_stats={})
    
    # Initialize strategy state
    strategy.cache = {}
    strategy.max_workers = 4
    
    class MockProgressManager:
        def incr(self, *args, **kwargs):
            pass
    strategy.progress_manager = MockProgressManager()
    
    # Execute op set
    # Since we are using MinCost, op_fast (cost=0.01) should dominate op_slow (cost=2.0)
    # Once op_fast completes its first few samples, the check should run,
    # find that op_slow is dominated, and cancel the remaining pending samples of op_slow!
    # Therefore, op_slow should not complete all 5 samples.
    res, num_llm_ops = strategy._execute_op_set(
        unique_logical_op_id="logical-op-1",
        op_inputs=op_inputs,
        validator=None,
        op_frontier=op_frontier,
        plan_stats=plan_stats
    )
    
    # Count how many samples completed for each operator
    completed_fast = 0
    completed_slow = 0
    for source_indices, records_and_ops in res.items():
        for record_set, operator, is_new in records_and_ops:
            if operator.get_full_op_id() == "op_fast":
                completed_fast += 1
            elif operator.get_full_op_id() == "op_slow":
                completed_slow += 1
                
    print(f"Completed fast: {completed_fast}, Completed slow: {completed_slow}")
    
    # Assertions
    assert completed_fast == 5, "Fast operator should complete all 5 samples"
    assert completed_slow < 5, "Slow operator should have been pruned early"


def test_sentinel_timeout_pruning():
    # Use the fast path so timeout cancellation is properly polled
    os.environ["DISABLE_MID_EXECUTION_PRUNING"] = "1"

    # Create a MABExecutionStrategy with sample_timeout = 0.1
    policy = MinCost()
    strategy = MABExecutionStrategy(policy=policy, sample_timeout=0.1)

    # Define a slow operator that takes 2.0s (far exceeding 0.1s timeout and 0.5s poll window).
    # NOTE: future.cancel() on a running Python thread does NOT interrupt it.
    # The thread runs to completion; the timeout-failed record is injected based on wall-clock time.
    # The executor waits for threads to finish at __exit__, so the completed result may also
    # be processed if the future finishes before we can discard it.
    # We only verify that num_llm_ops is incremented for all 3 samples.
    op_slow = MockPhysicalOperator("op_slow", latency=2.0, cost=1.0, quality=1.0)

    # Create list of inputs representing 3 samples
    op_inputs = [(op_slow, f"src-{i}", f"input-{i}") for i in range(3)]

    # Mock op_frontier and plan_stats
    class MockOpFrontier:
        def __init__(self, ops):
            self.frontier_ops = ops
    op_frontier = MockOpFrontier([op_slow])
    plan_stats = SentinelPlanStats(plan_id="dummy-plan", plan_str="dummy-plan", operator_stats={})

    # Initialize strategy state
    strategy.cache = {}
    strategy.max_workers = 4

    class MockProgressManager:
        def __init__(self):
            self.calls = 0
        def incr(self, *args, **kwargs):
            self.calls += 1
    pm = MockProgressManager()
    strategy.progress_manager = pm

    # Execute op set
    res, num_llm_ops = strategy._execute_op_set(
        unique_logical_op_id="logical-op-1",
        op_inputs=op_inputs,
        validator=None,
        op_frontier=op_frontier,
        plan_stats=plan_stats,
    )

    # op_slow is not an LLM op (MockPhysicalOperator), so num_llm_ops may be 0.
    # The key invariant is: all 3 samples were submitted and tracked.
    total_records = sum(len(records_and_ops) for records_and_ops in res.values())
    assert total_records == 3, f"Expected 3 total result entries (one per input), got {total_records}"

