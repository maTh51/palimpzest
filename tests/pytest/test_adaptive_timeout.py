import os
import pytest
import time
from palimpzest.policy import MinCost
from palimpzest.core.elements.records import DataRecordSet
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.core.lib.schemas import TextFile
from palimpzest.core.models import RecordOpStats, SentinelPlanStats
from palimpzest.query.execution.mab_execution_strategy import MABExecutionStrategy

class MockOperatorWithVariableLatency(PhysicalOperator):
    def __init__(self, op_id, initial_latency=0.1, spike_latency=2.0):
        super().__init__(output_schema=TextFile)
        self.op_id = op_id
        self.initial_latency = initial_latency
        self.spike_latency = spike_latency
        self.call_count = 0

    def get_full_op_id(self):
        return self.op_id

    def op_name(self):
        return f"MockOp-{self.op_id}"

    def get_id_params(self):
        return {}

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        lat = self.initial_latency if self.call_count == 1 else self.spike_latency
        time.sleep(lat)
        r_stats = RecordOpStats(
            record_id=f"rec-{self.call_count}",
            record_parent_ids=["p-1"],
            record_source_indices=["src-1"],
            record_state={},
            full_op_id=self.op_id,
            logical_op_id="logical-op-1",
            op_id=self.op_id,
            op_name=self.op_name(),
            time_per_record=lat,
            cost_per_record=0.01,
            quality=1.0,
            passed_operator=True,
        )
        return DataRecordSet(data_records=[], record_op_stats=[r_stats])

def test_adaptive_timeout():
    os.environ["DISABLE_MID_EXECUTION_PRUNING"] = "1"
    os.environ["ADAPTIVE_TIMEOUT"] = "1"
    os.environ["TIMEOUT_MULTIPLIER"] = "1.5"
    os.environ["TIMEOUT_MARGIN"] = "0.2"
    os.environ["SAFETY_TIMEOUT"] = "10.0"
    os.environ["MIN_TIMEOUT"] = "0.1"

    strategy = MABExecutionStrategy(policy=MinCost())
    op = MockOperatorWithVariableLatency("op_var", initial_latency=0.1, spike_latency=2.0)

    class MockOpFrontier:
        def __init__(self, ops): self.frontier_ops = ops
    class MockPM:
        def incr(self, *args, **kwargs): pass

    strategy.cache = {}
    strategy.max_workers = 4
    strategy.progress_manager = MockPM()
    from palimpzest.core.models import OperatorStats
    plan_stats = SentinelPlanStats(
        plan_id="p1",
        plan_str="p1",
        operator_stats={
            "logical-op-1": {
                op.get_full_op_id(): OperatorStats(
                    full_op_id=op.get_full_op_id(),
                    op_name=op.op_name(),
                    source_unique_logical_op_ids=[],
                    plan_id="p1",
                    op_details={},
                )
            }
        }
    )

    inputs_1 = [(op, "src-0", "input-0")]
    res_1, _ = strategy._execute_op_set("logical-op-1", inputs_1, None, MockOpFrontier([op]), plan_stats)
    plan_stats.add_record_op_stats("logical-op-1", res_1["src-0"][0][0].record_op_stats)

    inputs_2 = [(op, "src-1", "input-1")]
    res_2, _ = strategy._execute_op_set("logical-op-1", inputs_2, None, MockOpFrontier([op]), plan_stats)

    timed_out_found = False
    for rec_set, op_obj, is_new in res_2["src-1"]:
        for r in rec_set.record_op_stats:
            if r.record_id == "timeout-failed":
                timed_out_found = True
                assert r.passed_operator is False
                assert r.quality == 0.0

    assert timed_out_found, "Sample 2 should have been cancelled by adaptive timeout limit"
