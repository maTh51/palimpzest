#!/usr/bin/env python3
"""Analyze all CUAD experimental profiles and write a compact summary JSON."""
import json
from pathlib import Path

DATA_DIR = Path("/home/amarante/Jobs/Kunumi/palimpzest/abacus-research/opt-profiling-data")
OUT_JSON = DATA_DIR / "morfeu-cuad-summary-analysis.json"

EXP_NAMES = [
    "morfeu-cuad-smoke-timeout-0s",
    "morfeu-cuad-smoke-timeout-es",
    "morfeu-cuad-full-timeout-0s",
    "morfeu-cuad-full-timeout-es",
    "morfeu-cuad-full-timeout-0s-s1",
    "morfeu-cuad-full-timeout-es-s1",
    "morfeu-cuad-full-timeout-0s-s2",
    "morfeu-cuad-full-timeout-es-s2",
    "morfeu-cuad-consist-es-rep1",
    "morfeu-cuad-consist-es-rep2",
]


def extract_stats_tokens(exp_name: str) -> dict:
    stats_file = DATA_DIR / f"{exp_name}-stats.json"
    if not stats_file.exists():
        return {}
    data = json.loads(stats_file.read_text())
    opt_in_tok = 0.0
    opt_out_tok = 0.0
    opt_calls = 0.0
    sentinel_stats = data.get("sentinel_plan_stats") or {}
    for _, pstats in sentinel_stats.items():
        op_stats = pstats.get("operator_stats") or {}
        for _, phys_map in op_stats.items():
            for _, phys in phys_map.items():
                for rec in phys.get("record_op_stats_lst") or []:
                    opt_in_tok += float(rec.get("input_text_tokens") or 0.0)
                    opt_out_tok += float(rec.get("output_text_tokens") or 0.0)
                    opt_calls += float(rec.get("total_llm_calls") or 0.0)

    exec_in_tok = 0.0
    exec_out_tok = 0.0
    exec_calls = 0.0
    plan_stats = data.get("plan_stats") or {}
    for _, pstats in plan_stats.items():
        op_stats = pstats.get("operator_stats") or {}
        for _, phys_or_map in op_stats.items():
            if isinstance(phys_or_map, dict) and "record_op_stats_lst" in phys_or_map:
                phys_iter = [phys_or_map]
            elif isinstance(phys_or_map, dict):
                phys_iter = phys_or_map.values()
            else:
                phys_iter = []
            for phys in phys_iter:
                if not isinstance(phys, dict):
                    continue
                for rec in phys.get("record_op_stats_lst") or []:
                    exec_in_tok += float(rec.get("input_text_tokens") or 0.0)
                    exec_out_tok += float(rec.get("output_text_tokens") or 0.0)
                    exec_calls += float(rec.get("total_llm_calls") or 0.0)

    return {
        "opt_input_tokens": int(round(opt_in_tok)),
        "opt_output_tokens": int(round(opt_out_tok)),
        "opt_total_tokens": int(round(opt_in_tok + opt_out_tok)),
        "opt_llm_calls": int(round(opt_calls)),
        "exec_input_tokens": int(round(exec_in_tok)),
        "exec_output_tokens": int(round(exec_out_tok)),
        "exec_total_tokens": int(round(exec_in_tok + exec_out_tok)),
        "exec_llm_calls": int(round(exec_calls)),
    }


def extract_trajectory_info(exp_name: str) -> dict:
    traj_file = DATA_DIR / f"{exp_name}-trajectory.jsonl"
    if not traj_file.exists():
        return {}
    lines = [json.loads(line) for line in traj_file.read_text().splitlines() if line.strip()]
    op1_snaps = [row for row in lines if row.get("logical_op_id") == "1-81dd1f8073"]
    rounds = len(op1_snaps)
    if not op1_snaps:
        return {"rounds": 0}
    last = op1_snaps[-1]
    phys = last.get("physical_operators") or {}
    retired_pareto = [k for k, v in phys.items() if v.get("prune_reason") == "DOMINATED_BY_PARETO"]
    pareto_opt = [k for k, v in phys.items() if v.get("status") == "PARETO_OPTIMAL"]
    active = [k for k, v in phys.items() if v.get("status") == "ACTIVE_FRONTIER"]

    round_progression = []
    for r_idx, snap in enumerate(op1_snaps, start=1):
        p_map = snap.get("physical_operators") or {}
        best_mu = max((v["metrics"]["mean"]["quality"] for v in p_map.values() if v.get("samples_drawn", 0) > 0), default=0.0)
        round_progression.append({
            "round": r_idx,
            "global_total_samples": snap.get("global_total_samples"),
            "converged": snap.get("converged"),
            "convergence_reason": (snap.get("convergence_info") or {}).get("reason") if snap.get("convergence_info") else None,
            "best_val_mu": round(best_mu, 4),
            "total_ops_seen": len(p_map),
            "retired_ops": sum(1 for v in p_map.values() if v.get("status") == "RETIRED"),
        })

    return {
        "rounds": rounds,
        "final_global_samples": last.get("global_total_samples"),
        "converged": last.get("converged"),
        "convergence_info": last.get("convergence_info"),
        "total_ops_evaluated": len(phys),
        "retired_pareto_count": len(retired_pareto),
        "pareto_optimal_count": len(pareto_opt),
        "active_frontier_count": len(active),
        "round_progression": round_progression,
    }


def main():
    summary = {}
    for exp in EXP_NAMES:
        m_file = DATA_DIR / f"{exp}-metrics.json"
        metrics = json.loads(m_file.read_text()) if m_file.exists() else {}
        tok = extract_stats_tokens(exp)
        traj = extract_trajectory_info(exp)
        summary[exp] = {
            "metrics": {
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "optimization_time": metrics.get("optimization_time"),
                "plan_execution_time": metrics.get("plan_execution_time"),
                "total_execution_time": metrics.get("total_execution_time"),
                "plan_str": metrics.get("plan_str"),
            },
            "tokens": tok,
            "trajectory": traj,
        }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
