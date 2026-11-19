"""Margin that justifies rho: for every final-Pareto heavy op, min over snapshots (samples>=4)
of  mean_quality(op) / max_peer mean_quality.  Any rho below this min never prunes a Pareto op.
Also: for every op that ended RETIRED or never Pareto, the same ratio (what rho can catch)."""
import json

D = "opt-profiling-data"
for p in ["morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-dyn", "morfeu-biodex-full-timeout-30s"]:
    traj = [json.loads(l) for l in open(f"{D}/{p}-trajectory.jsonl")]
    fin = {}
    for s in traj:
        for o, v in s["physical_operators"].items():
            fin[o] = v
    ratios = {}
    for s in traj:
        ops = s["physical_operators"]
        best = max(w["metrics"]["mean"]["quality"] for w in ops.values()) or 1e-9
        for o, v in ops.items():
            if v["metrics"]["mean"]["time"] <= 1.0 or v["samples_drawn"] < 4:
                continue
            ratios.setdefault(o, []).append(v["metrics"]["mean"]["quality"] / best)
    par = {o: min(r) for o, r in ratios.items() if fin[o]["status"] == "PARETO_OPTIMAL"}
    non = {o: min(r) for o, r in ratios.items() if fin[o]["status"] != "PARETO_OPTIMAL"}
    print(f"\n{p}")
    print(f"  Pareto ops   min ratio: {sorted(round(x,3) for x in par.values())}")
    print(f"  Others (sorted)       : {sorted(round(x,3) for x in non.values())}")
