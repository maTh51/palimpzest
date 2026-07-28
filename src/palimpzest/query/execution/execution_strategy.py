import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from chromadb.api.models.Collection import Collection

from palimpzest.constants import Cardinality
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, PlanStats, RecordOpStats, SentinelPlanStats
from palimpzest.policy import Policy
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import PZSentinelProgressManager
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)

class BaseExecutionStrategy:
    def __init__(self,
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 batch_size: int | None = None,
                 num_samples: int | None = None,
                 verbose: bool = False,
                 progress: bool = True,
                 *args,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.verbose = verbose
        self.progress = progress


class ExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing query plans. Defines how to execute a PhysicalPlan.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized ExecutionStrategy {self.__class__.__name__}")
        logger.debug(f"ExecutionStrategy initialized with config: {self.__dict__}")

    @abstractmethod
    def execute_plan(self, plan: PhysicalPlan) -> tuple[list[DataRecord], PlanStats]:
        """Execute a single plan according to strategy"""
        pass

    def _create_input_queues(self, plan: PhysicalPlan) -> dict[str, dict[str, list]]:
        """Initialize input queues for each operator in the plan."""
        input_queues = {f"{topo_idx}-{op.get_full_op_id()}": {} for topo_idx, op in enumerate(plan)}
        for topo_idx, op in enumerate(plan):
            full_op_id = op.get_full_op_id()
            unique_op_id = f"{topo_idx}-{full_op_id}"
            if isinstance(op, ScanPhysicalOp):
                scan_end_idx = (
                    len(op.datasource)
                    if self.num_samples is None
                    else min(self.scan_start_idx + self.num_samples, len(op.datasource))
                )
                input_queues[unique_op_id][f"source_{full_op_id}"] = [idx for idx in range(self.scan_start_idx, scan_end_idx)]
            elif isinstance(op, ContextScanOp):
                input_queues[unique_op_id][f"source_{full_op_id}"] = [None]
            else:
                for source_unique_full_op_id in plan.get_source_unique_full_op_ids(topo_idx, op):
                    input_queues[unique_op_id][source_unique_full_op_id] = []

        return input_queues

class SentinelExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing sentinel query plans. Defines how to execute a SentinelPlan."""
    """
    Specialized query processor that implements MAB sentinel strategy
    for coordinating optimization and execution.
    """
    def __init__(
        self,
        policy: Policy,
        k: int = 6,
        j: int = 4,
        sample_budget: int = 100,
        sample_cost_budget: float | None = None,
        priors: dict | None = None,
        use_final_op_quality: bool = False,
        seed: int = 42,
        exp_name: str | None = None,
        dont_use_priors: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.sample_cost_budget = sample_cost_budget
        self.policy = policy
        self.priors = priors
        self.use_final_op_quality = use_final_op_quality
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name
        self.dont_use_priors = dont_use_priors

        # general cache which maps hash(logical_op_id, phys_op_id, hash(input)) --> record_set
        self.cache: dict[int, DataRecordSet] = {}

        # progress manager used to track progress of the execution
        self.progress_manager: PZSentinelProgressManager | None = None

    def _score_quality(
        self,
        validator: Validator,
        source_indices_to_record_sets: dict[tuple[str], list[tuple[DataRecordSet, PhysicalOperator]]],
    ) -> tuple[dict[int, list[DataRecordSet]], GenerationStats]:
        # extract information about the logical operation performed at this stage of the sentinel plan;
        # NOTE: we can infer these fields from context clues, but in the long-term we should have a more
        #       principled way of getting these directly from attributes either stored in the sentinel_plan
        #       or in the PhysicalOperator
        def is_perfect_quality_op(op: PhysicalOperator):
            return (
                not isinstance(op, LLMConvert)
                and not isinstance(op, LLMFilter)
                and not isinstance(op, TopKOp)
                and not isinstance(op, JoinOp)
            )

        # create minimal set of futures necessary to compute quality of each output record
        futures, full_hashes, full_hash_to_bool_output = [], set(), {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for _, record_set_tuples in source_indices_to_record_sets.items():
                for record_set, op in record_set_tuples:
                    # if this operation does not involve an LLM, every record_op_stats object gets perfect quality
                    if is_perfect_quality_op(op):
                        for record_op_stats in record_set.record_op_stats:
                            record_op_stats.quality = 1.0
                        continue

                    # if the operation failed, assign 0.0 quality
                    if len(record_set) == 0:
                        record_set.record_op_stats[0].quality = 0.0
                        continue

                    # create future for map
                    if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                        fields = op.generated_fields
                        input_record: DataRecord = record_set.input
                        output = record_set.data_records[0].to_dict(project_cols=fields)
                        output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                        full_hash = f"{hash(input_record)}{hash(output_str)}"
                        if full_hash not in full_hashes:
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_map, op, fields, input_record, output, full_hash))

                    # create future for flat map
                    elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                        fields = op.generated_fields
                        input_record: DataRecord = record_set.input
                        output, output_strs = [], []
                        for data_record in record_set.data_records:
                            output.append(data_record.to_dict(project_cols=fields))
                            output_strs.append(data_record.to_json_str(project_cols=fields, bytes_to_str=True, sorted=True))
                        full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                        if full_hash not in full_hashes:
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_flat_map, op, fields, input_record, output, full_hash))

                    # create future for top-k
                    elif isinstance(op, TopKOp):
                        fields = op.generated_fields
                        input_record: DataRecord = record_set.input
                        output = record_set.data_records[0].to_dict(project_cols=fields)
                        output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                        full_hash = f"{hash(input_record)}{hash(output_str)}"
                        if full_hash not in full_hashes:
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_topk, op, fields, input_record, output, full_hash))

                    # create future for filter
                    elif isinstance(op, LLMFilter):
                        filter_str = op.filter_obj.filter_condition
                        input_record: DataRecord = record_set.input
                        output = record_set.data_records[0]._passed_operator
                        full_hash = f"{filter_str}{hash(input_record)}"
                        if full_hash not in full_hashes:
                            full_hash_to_bool_output[full_hash] = output
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_filter, op, filter_str, input_record, output, full_hash))

                    # create future for join
                    elif isinstance(op, JoinOp):
                        condition = op.condition
                        for left_idx, left_input_record in enumerate(record_set.input[0]):
                            for right_idx, right_input_record in enumerate(record_set.input[1]):
                                record_idx = left_idx * len(record_set.input[1]) + right_idx
                                output = record_set.data_records[record_idx]._passed_operator
                                full_hash = f"{condition}{hash(left_input_record)}{hash(right_input_record)}"
                                if full_hash not in full_hashes:
                                    full_hash_to_bool_output[full_hash] = output
                                    full_hashes.add(full_hash)
                                    futures.append(executor.submit(validator._score_join, op, condition, left_input_record, right_input_record, output, full_hash))

        # collect results from futures
        full_hash_to_score, validation_gen_stats = {}, GenerationStats()
        for future in as_completed(futures):
            score, gen_stats, full_hash = future.result()
            full_hash_to_score[full_hash] = score
            validation_gen_stats += gen_stats

        # compute quality of each output computed by this operator
        for _, record_set_tuples in source_indices_to_record_sets.items():
            for record_set, op in record_set_tuples:
                if is_perfect_quality_op(op) or len(record_set) == 0:
                    continue

                if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                    full_hash = f"{hash(input_record)}{hash(output_str)}"
                    record_set.record_op_stats[0].quality = full_hash_to_score[full_hash]

                elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output_strs = []
                    for data_record in record_set.data_records:
                        output_strs.append(data_record.to_json_str(project_cols=fields, bytes_to_str=True, sorted=True))
                    full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                    score = full_hash_to_score[full_hash]
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = score

                # TODO: this scoring function will (likely) bias towards small values of k since it
                # measures precision and not recall / F1; will need to revisit this in the future
                elif isinstance(op, TopKOp):
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                    full_hash = f"{hash(input_record)}{hash(output_str)}"
                    score = full_hash_to_score[full_hash]
                    record_set.record_op_stats[0].quality = score

                elif isinstance(op, LLMFilter):
                    filter_str = op.filter_obj.filter_condition
                    input_record: DataRecord = record_set.input
                    output = record_set.data_records[0]._passed_operator
                    full_hash = f"{filter_str}{hash(input_record)}"
                    if output == full_hash_to_bool_output[full_hash]:
                        record_set.record_op_stats[0].quality = full_hash_to_score[full_hash]
                    else:
                        record_set.record_op_stats[0].quality = 1.0 - full_hash_to_score[full_hash]

                elif isinstance(op, JoinOp):
                    condition = op.condition
                    for left_idx, left_input_record in enumerate(record_set.input[0]):
                        for right_idx, right_input_record in enumerate(record_set.input[1]):
                            record_idx = left_idx * len(record_set.input[1]) + right_idx
                            output = record_set.data_records[record_idx]._passed_operator
                            full_hash = f"{condition}{hash(left_input_record)}{hash(right_input_record)}"
                            if output == full_hash_to_bool_output[full_hash]:
                                record_set.record_op_stats[record_idx].quality = full_hash_to_score[full_hash]
                            else:
                                record_set.record_op_stats[record_idx].quality = 1.0 - full_hash_to_score[full_hash]

        # return the quality annotated record sets
        return source_indices_to_record_sets, validation_gen_stats

    def _execute_op_set(
        self,
        unique_logical_op_id: str,
        op_inputs: list[tuple[PhysicalOperator, str | tuple, int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]]],
        validator=None,
        op_frontier=None,
        plan_stats=None,
    ) -> tuple[dict[int, list[tuple[DataRecordSet, PhysicalOperator, bool]]], dict[str, int]]:
        import datetime, time, os

        def execute_op_wrapper(operator, source_indices, input):
            record_set = operator(input[0], input[1]) if isinstance(operator, JoinOp) else operator(input)
            return record_set, operator, source_indices, input

        def get_hash(operator, input):
            if isinstance(input, list):
                input = tuple(input)
            elif isinstance(input, tuple):
                input = (tuple(input[0]), tuple(input[1]))
            return hash(f"{operator.get_full_op_id()}{hash(input)}")

        def make_failed_record_set(operator, source_indices, input_data, elapsed, label="failed-op"):
            op_id = operator.get_full_op_id()
            parent_ids = [input_data._id] if hasattr(input_data, "_id") else [str(source_indices)]
            r_stats = RecordOpStats(
                record_id=label, record_parent_ids=parent_ids,
                record_source_indices=[source_indices], record_state={},
                full_op_id=op_id, logical_op_id=operator.logical_op_id or "logical-op",
                op_id=op_id, op_name=operator.op_name(),
                time_per_record=elapsed, cost_per_record=0.0, quality=0.0, passed_operator=False,
            )
            rs = DataRecordSet(data_records=[], record_op_stats=[r_stats])
            rs.input = input_data
            return rs

        # resolve timeout: attribute takes priority, then env var
        time_limit = getattr(self, "sample_timeout", None)
        if time_limit is None and os.getenv("SAMPLE_TIMEOUT"):
            try:
                time_limit = float(os.getenv("SAMPLE_TIMEOUT"))
            except ValueError:
                pass

        # initialize mapping from source indices to output record sets
        source_indices_to_record_sets_and_ops = {source_indices: [] for _, source_indices, _ in op_inputs}

        # check cache
        final_op_inputs = []
        for operator, source_indices, input in op_inputs:
            op_input_hash = get_hash(operator, input)
            if op_input_hash in self.cache and not os.getenv("DISABLE_CACHE"):
                record_set, operator = self.cache[op_input_hash]
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, False))
            else:
                final_op_inputs.append((operator, source_indices, input))

        num_llm_ops = 0
        output_record_sets = []

        ts = datetime.datetime.now().strftime("[%H:%M:%S]")
        print(f"\n{ts} [Sentinel Batch] Executing {len(final_op_inputs)} sample(s) for Logical Op {unique_logical_op_id}...", flush=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(execute_op_wrapper, operator, source_indices, input)
                for operator, source_indices, input in final_op_inputs
            ]
            future_to_info = {fut: tup for fut, tup in zip(futures, final_op_inputs)}
            future_to_start_time = {fut: time.time() for fut in futures}
            timeout_count = 0

            # ------------------------------------------------------------------
            # FAST PATH: timeout cancellation only, no mid-batch dominance checks.
            # Use this when DISABLE_MID_EXECUTION_PRUNING=1 (default for experiments).
            # ------------------------------------------------------------------
            if os.getenv("DISABLE_MID_EXECUTION_PRUNING"):
                remaining = set(futures)
                while remaining:
                    # poll for completions with short timeout
                    done_now = set()
                    try:
                        for fut in as_completed(list(remaining), timeout=0.5):
                            done_now.add(fut)
                            break
                    except Exception:
                        pass

                    for future in list(remaining):
                        if future not in done_now:
                            continue
                        remaining.discard(future)
                        operator, source_indices, input_data = future_to_info[future]
                        op_id = operator.get_full_op_id()
                        try:
                            record_set, operator, source_indices, input_data = future.result()
                            if isinstance(operator, JoinOp):
                                record_set = record_set[0]
                            if not os.getenv("DISABLE_CACHE"):
                                self.cache[get_hash(operator, input_data)] = (record_set, operator)
                            record_set.input = input_data
                            if self._is_llm_op(operator):
                                num_llm_ops += 1
                                self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=record_set.get_total_cost())
                            output_record_sets.append((record_set, operator, source_indices, input_data))
                        except Exception as e:
                            logger.error(f"Error executing operator: {e}")
                            elapsed = time.time() - future_to_start_time[future]
                            record_set = make_failed_record_set(operator, source_indices, input_data, elapsed)
                            if self._is_llm_op(operator):
                                num_llm_ops += 1
                                if self.progress_manager is not None:
                                    self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=0.0)
                            output_record_sets.append((record_set, operator, source_indices, input_data))
                            print(f"\n[Sentinel Exception] Operator {op_id} failed: {e}.\n", flush=True)

                    # wall-clock timeout check
                    if time_limit is not None:
                        now = time.time()
                        for future in list(remaining):
                            elapsed = now - future_to_start_time[future]
                            if elapsed > time_limit:
                                timeout_count += 1
                                future.cancel()
                                remaining.discard(future)
                                operator, source_indices, input_data = future_to_info[future]
                                op_id = operator.get_full_op_id()
                                ts2 = datetime.datetime.now().strftime("[%H:%M:%S]")
                                print(f"{ts2} [Sentinel Timeout] Operator {op_id} timed out (> {time_limit:.2f}s). Cancelling.", flush=True)
                                record_set = make_failed_record_set(operator, source_indices, input_data, time_limit, "timeout-failed")
                                if self._is_llm_op(operator):
                                    num_llm_ops += 1
                                    if self.progress_manager is not None:
                                        self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=0.0)
                                output_record_sets.append((record_set, operator, source_indices, input_data))

            # ------------------------------------------------------------------
            # FULL PATH: mid-execution speculative Pareto pruning + timeout.
            # Active when DISABLE_MID_EXECUTION_PRUNING is unset.
            # ------------------------------------------------------------------
            else:
                import numpy as np
                epsilon = float(os.getenv("PRUNING_EPSILON", "0.0"))

                op_id_to_futures = {}
                for fut, (operator, source_indices, input_data) in zip(futures, final_op_inputs):
                    op_id = operator.get_full_op_id()
                    op_id_to_futures.setdefault(op_id, []).append(fut)

                op_id_to_stats = {
                    op_id: {"total_samples": len(f_list), "completed_records": [],
                            "quality_scores": [], "total_time": 0.0, "total_cost": 0.0, "aborted": False}
                    for op_id, f_list in op_id_to_futures.items()
                }

                from concurrent.futures import wait, FIRST_COMPLETED
                active_futures = set(futures)
                while active_futures:
                    done, _ = wait(active_futures, timeout=0.1, return_when=FIRST_COMPLETED)

                    for future in done:
                        if future not in active_futures:
                            continue
                        active_futures.remove(future)
                        operator, source_indices, input_data = future_to_info[future]
                        op_id = operator.get_full_op_id()
                        stats = op_id_to_stats[op_id]
                        try:
                            record_set, operator, source_indices, input_data = future.result()
                        except Exception as e:
                            logger.error(f"Error executing operator: {e}")
                            elapsed = time.time() - future_to_start_time[future]
                            record_set = make_failed_record_set(operator, source_indices, input_data, elapsed)
                            stats["quality_scores"].append(0.0)
                            stats["total_time"] += elapsed
                            stats["completed_records"].append((record_set, operator, source_indices, input_data))
                            if self._is_llm_op(operator):
                                num_llm_ops += 1
                                if self.progress_manager is not None:
                                    self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=0.0)
                            print(f"\n[Sentinel Exception] Operator {op_id} failed: {e}.\n", flush=True)
                            continue

                        if isinstance(operator, JoinOp):
                            record_set = record_set[0]
                        if not os.getenv("DISABLE_CACHE"):
                            self.cache[get_hash(operator, input_data)] = (record_set, operator)
                        record_set.input = input_data
                        if self._is_llm_op(operator):
                            num_llm_ops += 1
                            self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=record_set.get_total_cost())

                        quality = 1.0
                        if validator is not None:
                            temp_dict = {source_indices: [(record_set, operator)]}
                            self._score_quality(validator, temp_dict)
                            q_scores = [r.quality for r in record_set.record_op_stats if r.quality is not None]
                            quality = float(np.mean(q_scores)) if q_scores else 1.0

                        stats["quality_scores"].append(quality)
                        stats["total_cost"] += record_set.get_total_cost()
                        stats["total_time"] += sum(getattr(r_s, "time_per_record", 0.0) for r_s in record_set.record_op_stats)
                        stats["completed_records"].append((record_set, operator, source_indices, input_data))

                    # Timeout check
                    if time_limit is not None:
                        now = time.time()
                        for future in list(active_futures):
                            elapsed = now - future_to_start_time[future]
                            if elapsed > time_limit:
                                timeout_count += 1
                                future.cancel()
                                active_futures.remove(future)
                                operator, source_indices, input_data = future_to_info[future]
                                op_id = operator.get_full_op_id()
                                ts2 = datetime.datetime.now().strftime("[%H:%M:%S]")
                                print(f"{ts2} [Sentinel Timeout] Operator {op_id} timed out (> {time_limit:.2f}s). Cancelling.", flush=True)
                                stats = op_id_to_stats[op_id]
                                record_set = make_failed_record_set(operator, source_indices, input_data, time_limit, "timeout-failed")
                                stats["quality_scores"].append(0.0)
                                stats["total_time"] += time_limit
                                stats["completed_records"].append((record_set, operator, source_indices, input_data))
                                if self._is_llm_op(operator):
                                    num_llm_ops += 1
                                    if self.progress_manager is not None:
                                        self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=0.0)

                    # Mid-execution Pareto dominance pruning
                    if op_frontier is not None and plan_stats is not None:
                        frontier_ops_metrics = {}
                        for op in op_frontier.frontier_ops:
                            oid = op.get_full_op_id()
                            hist_cost, hist_time, hist_quality_lst, hist_samples = 0.0, 0.0, [], 0
                            op_stats = plan_stats.operator_stats.get(unique_logical_op_id, {}).get(oid)
                            if op_stats is not None:
                                hist_samples = len(op_stats.record_op_stats_lst)
                                hist_cost = sum(r.cost_per_record for r in op_stats.record_op_stats_lst)
                                hist_time = sum(r.time_per_record for r in op_stats.record_op_stats_lst)
                                hist_quality_lst = [r.quality for r in op_stats.record_op_stats_lst if r.quality is not None]
                            batch_s = op_id_to_stats.get(oid)
                            if batch_s is not None:
                                total_s = hist_samples + len(batch_s["quality_scores"])
                                if total_s > 0:
                                    total_q = hist_quality_lst + batch_s["quality_scores"]
                                    frontier_ops_metrics[oid] = {
                                        "cost": (hist_cost + batch_s["total_cost"]) / total_s,
                                        "time": (hist_time + batch_s["total_time"]) / total_s,
                                        "quality": float(np.mean(total_q)) if total_q else 1.0,
                                    }
                            elif hist_samples > 0:
                                frontier_ops_metrics[oid] = {
                                    "cost": hist_cost / hist_samples,
                                    "time": hist_time / hist_samples,
                                    "quality": float(np.mean(hist_quality_lst)) if hist_quality_lst else 1.0,
                                }

                        policy_dict = self.policy.get_dict()
                        for oid, batch_s in op_id_to_stats.items():
                            n_done = len(batch_s["quality_scores"])
                            n_total = batch_s["total_samples"]
                            if 1 <= n_done < n_total and not batch_s["aborted"]:
                                hist_cost, hist_time, hist_qs, hist_n = 0.0, 0.0, 0.0, 0
                                op_stats = plan_stats.operator_stats.get(unique_logical_op_id, {}).get(oid)
                                if op_stats is not None:
                                    hist_n = len(op_stats.record_op_stats_lst)
                                    hist_cost = sum(r.cost_per_record for r in op_stats.record_op_stats_lst)
                                    hist_time = sum(r.time_per_record for r in op_stats.record_op_stats_lst)
                                    hist_qs = sum(r.quality for r in op_stats.record_op_stats_lst if r.quality is not None)
                                total_opt = hist_n + n_total
                                opt_q = (hist_qs + sum(batch_s["quality_scores"]) + (n_total - n_done)) / total_opt
                                opt_t = (hist_time + batch_s["total_time"]) / total_opt
                                opt_c = (hist_cost + batch_s["total_cost"]) / total_opt
                                is_dominated = any(
                                    (True if policy_dict.get("cost", 0.0) == 0.0 else m["cost"] <= opt_c * (1 + epsilon)) and
                                    (True if policy_dict.get("time", 0.0) == 0.0 else m["time"] <= opt_t * (1 + epsilon)) and
                                    (True if policy_dict.get("quality", 0.0) == 0.0 else m["quality"] >= opt_q * (1 - epsilon))
                                    for other_oid, m in frontier_ops_metrics.items() if other_oid != oid
                                )
                                if is_dominated:
                                    print(f"\n[Sentinel Pruning] Operator {oid} is optimistically dominated (Epsilon={epsilon}). Aborting remaining {n_total - n_done} samples.\n", flush=True)
                                    batch_s["aborted"] = True
                                    for fut in op_id_to_futures[oid]:
                                        if fut in active_futures:
                                            active_futures.remove(fut)
                                            fut.cancel()

                for oid, batch_s in op_id_to_stats.items():
                    output_record_sets.extend(batch_s["completed_records"])

            # Summary log
            if timeout_count > 0 and time_limit is not None:
                ts2 = datetime.datetime.now().strftime("[%H:%M:%S]")
                print(f"\n{ts2} [Sentinel Summary] Batch done. Cancelled {timeout_count} timed-out sample(s) (> {time_limit:.2f}s).\n", flush=True)

            for record_set, operator, source_indices, input in output_record_sets:
                record_set.input = input
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, True))

        return source_indices_to_record_sets_and_ops, num_llm_ops

    def _is_llm_op(self, physical_op: PhysicalOperator) -> bool:
        is_llm_convert = isinstance(physical_op, LLMConvert)
        is_llm_filter = isinstance(physical_op, LLMFilter)
        is_llm_topk = isinstance(physical_op, TopKOp) and isinstance(physical_op.index, Collection)
        is_llm_join = isinstance(physical_op, JoinOp)
        return is_llm_convert or is_llm_filter or is_llm_topk or is_llm_join

    @abstractmethod
    def execute_sentinel_plan(self, sentinel_plan: SentinelPlan, train_dataset: dict[str, Dataset], validator: Validator) -> SentinelPlanStats:
        """Execute a SentinelPlan according to strategy"""
        pass
