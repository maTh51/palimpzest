import json
import csv
import glob
import os
import sys

def parse_run(exp_prefix):
    metrics_path = f"opt-profiling-data/{exp_prefix}-metrics.json"
    timeouts_path = f"opt-profiling-data/{exp_prefix}-timeouts.csv"
    trajectory_path = f"opt-profiling-data/{exp_prefix}-trajectory.jsonl"
    
    res = {"name": exp_prefix}
    
    # Metrics
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
            res["rp@5"] = m.get("rp@5", 0.0)
            res["opt_time_s"] = round(m.get("optimization_time", 0.0), 1)
            res["plan_exec_time_s"] = round(m.get("plan_execution_time", 0.0), 1)
            res["total_time_s"] = round(m.get("total_execution_time", 0.0), 1)
            res["cost"] = m.get("total_execution_cost", 0.0)
    else:
        res["rp@5"] = "N/A"
        res["opt_time_s"] = "N/A"
        res["plan_exec_time_s"] = "N/A"
        res["total_time_s"] = "N/A"
        res["cost"] = "N/A"
        
    # Timeouts CSV
    if os.path.exists(timeouts_path):
        rows = []
        with open(timeouts_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        res["total_samples"] = len(rows)
        res["successful_samples"] = sum(1 for r in rows if r["outcome"] == "ok")
        res["timed_out_samples"] = sum(1 for r in rows if r["outcome"] == "timeout")
        res["timeout_rate"] = f"{(res['timed_out_samples'] / max(1, len(rows))) * 100:.1f}%"
        lats = [float(r["latency_s"]) for r in rows if r["outcome"] == "ok"]
        res["avg_sample_latency"] = round(sum(lats)/len(lats), 2) if lats else 0.0
        res["max_sample_latency"] = round(max(lats), 2) if lats else 0.0
    else:
        res["total_samples"] = 0
        res["successful_samples"] = 0
        res["timed_out_samples"] = 0
        res["timeout_rate"] = "N/A"
        res["avg_sample_latency"] = 0.0
        res["max_sample_latency"] = 0.0
        
    # Trajectory
    if os.path.exists(trajectory_path):
        with open(trajectory_path) as f:
            lines = [json.loads(l) for l in f]
        res["rounds"] = len(lines)
        if lines:
            last = lines[-1]
            ops = last.get("physical_operators", {})
            res["tracked_ops"] = len(ops)
            res["pareto_ops"] = [k for k, v in ops.items() if v.get("status") == "PARETO_OPTIMAL"]
            res["retired_ops"] = [k for k, v in ops.items() if v.get("status") == "RETIRED"]
    else:
        res["rounds"] = 0
        res["tracked_ops"] = 0
        res["pareto_ops"] = []
        res["retired_ops"] = []
        
    return res

if __name__ == "__main__":
    runs = sorted(
        os.path.basename(p)[: -len("-metrics.json")]
        for p in glob.glob("opt-profiling-data/morfeu-biodex-full-timeout-*-metrics.json")
    )
    print(f"{'Experiment':<44} | {'rp@5':<6} | {'Opt Time':<10} | {'Total Time':<10} | {'Samples':<8} | {'Timeouts':<12} | {'Tracked Ops'}")
    print("-" * 120)
    for r in runs:
        info = parse_run(r)
        rp = f"{info['rp@5']:.4f}" if isinstance(info['rp@5'], float) else str(info['rp@5'])
        print(f"{info['name']:<44} | {rp:<6} | {str(info['opt_time_s']) + 's':<10} | {str(info['total_time_s']) + 's':<10} | {info['total_samples']:<8} | {str(info['timed_out_samples']) + ' (' + str(info['timeout_rate']) + ')':<12} | {info['tracked_ops']}")

