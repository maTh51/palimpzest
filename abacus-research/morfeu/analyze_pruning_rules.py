"""Cross-validate early-pruning rules on all 3 runs + test a parameter-free rule.

Rules evaluated at every trajectory snapshot, for every physical op of a logical op:
  A) threshold:   samples>=N and mean_quality < tau
  B) dominance:   ucb_quality(op) < max_{peers} lcb_quality(peer)   (no tuned threshold:
                  the op's optimistic quality is below a peer's pessimistic quality)
  C) relative:    samples>=N and mean_quality < rho * best_mean_quality_among_peers
Also prints the early (first-snapshot) quality of every op that ends PARETO_OPTIMAL,
which is the margin any threshold must respect.
"""
import json
from collections import defaultdict

D = "opt-profiling-data"
RUNS = ["morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-dyn", "morfeu-biodex-full-timeout-30s"]


def load(p):
    with open(f"{D}/{p}-trajectory.jsonl") as f:
        return [json.loads(l) for l in f]


def final(traj):
    out = {}
    for s in traj:
        for o, v in s["physical_operators"].items():
            out[o] = v
    return out


def evaluate(traj, fin, rule):
    pruned_at = {}
    for s in traj:
        ops = s["physical_operators"]
        for o, v in ops.items():
            if o in pruned_at:
                continue
            if rule(o, v, ops):
                pruned_at[o] = v["samples_drawn"]
    saved_n = saved_s = lost = 0
    for o, n in pruned_at.items():
        extra = fin[o]["samples_drawn"] - n
        saved_n += extra
        saved_s += extra * fin[o]["metrics"]["mean"]["time"]
        lost += fin[o]["status"] == "PARETO_OPTIMAL"
    return len(pruned_at), saved_n, saved_s, lost


heavy = lambda v: v["metrics"]["mean"]["time"] > 1.0

for p in RUNS:
    traj = load(p); fin = final(traj)
    total_s = sum(v["samples_drawn"] * v["metrics"]["mean"]["time"] for v in fin.values())
    n_par = sum(v["status"] == "PARETO_OPTIMAL" and heavy(v) for v in fin.values())
    print(f"\n### {p}  (heavy-op sample-seconds={total_s:.0f}, heavy Pareto ops={n_par})")

    # early quality of final-Pareto heavy ops
    first = {}
    for s in traj:
        for o, v in s["physical_operators"].items():
            first.setdefault(o, (v["samples_drawn"], v["metrics"]["mean"]["quality"]))
    eq = [(round(first[o][1], 3), first[o][0]) for o, v in fin.items() if v["status"] == "PARETO_OPTIMAL" and heavy(v)]
    print(f"  final-Pareto ops: (quality at first snapshot, samples) = {sorted(eq)}")

    print(f"  {'rule':<34} {'pruned':>6} {'saved_n':>7} {'saved_s':>8} {'saved%':>6} {'pareto_lost':>11}")
    rules = {}
    for tau in [0.10, 0.15, 0.20]:
        rules[f"A thresh N>=4 tau={tau}"] = (lambda t: lambda o, v, ops: heavy(v) and v["samples_drawn"] >= 4
                                             and v["metrics"]["mean"]["quality"] < t)(tau)
    rules["B dominance ucb<max peer lcb"] = lambda o, v, ops: heavy(v) and v["metrics"]["ucb"]["quality"] < max(
        w["metrics"]["lcb"]["quality"] for k, w in ops.items() if k != o)
    for rho in [0.25, 0.33, 0.5]:
        rules[f"C relative N>=4 rho={rho}"] = (lambda r: lambda o, v, ops: heavy(v) and v["samples_drawn"] >= 4
                                               and v["metrics"]["mean"]["quality"] < r * max(
                                                   w["metrics"]["mean"]["quality"] for w in ops.values()))(rho)
    for name, rule in rules.items():
        n, sn, ss, lost = evaluate(traj, fin, rule)
        print(f"  {name:<34} {n:6d} {sn:7d} {ss:8.0f} {100*ss/total_s:5.1f}% {lost:11d}")
