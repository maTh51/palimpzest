"""Offline, data-driven analysis to choose the dynamic-timeout equation.

Uses the unconstrained run (timeout-0s) as ground truth: every sample ran to completion,
so we know its true latency and the final fate (Pareto / retired) of its physical operator.

Parts:
  1. Where does optimization time go? (per logical op / physical op, final status, quality)
  2. Is latency predictive of quality? (does cutting slow samples cut bad operators?)
  3. Replay candidate timeout equations  T = k * Q_q(lat of logical op so far)  on the 0s run:
       - seconds saved
       - fraction of cut samples belonging to final-Pareto operators (false cuts)
  4. Replay early-pruning rules on the trajectory snapshots:
       retire op if samples>=N and mean quality < tau  -> samples saved / Pareto ops lost

Run from abacus-research/:  python morfeu/analyze_timeout_equation.py
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

D = "opt-profiling-data"
RUNS = ["morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-qprune-rho0.5"]
BASE = RUNS[0]


def load_csv(prefix):
    with open(f"{D}/{prefix}-timeouts.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["latency_s"] = float(r["latency_s"])
        r["wall_time"] = float(r["wall_time"])
    rows.sort(key=lambda r: r["wall_time"])
    return rows


def load_traj(prefix):
    with open(f"{D}/{prefix}-trajectory.jsonl") as f:
        return [json.loads(l) for l in f]


def final_ops(traj):
    """full_op_id -> final snapshot (last time it appears)."""
    out = {}
    for snap in traj:
        for oid, v in snap["physical_operators"].items():
            out[oid] = {**v, "logical": snap["logical_op_id"]}
    return out


def hr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------- 1
hr("1. WHERE DOES THE OPTIMIZATION TIME GO?  (sum of sample latencies)")
for prefix in RUNS:
    rows = load_csv(prefix)
    fin = final_ops(load_traj(prefix))
    tot = sum(r["latency_s"] for r in rows)
    by_status = defaultdict(float)
    n_status = defaultdict(int)
    for r in rows:
        st = fin.get(r["full_op_id"], {}).get("status", "UNKNOWN")
        by_status[st] += r["latency_s"]
        n_status[st] += 1
    print(f"\n{prefix}: total sample-seconds = {tot:.0f}s, samples = {len(rows)}")
    for st in sorted(by_status, key=lambda s: -by_status[s]):
        print(f"   {st:<16} {by_status[st]:8.0f}s ({100*by_status[st]/max(tot,1):5.1f}%)  samples={n_status[st]}")

rows0 = load_csv(BASE)
fin0 = final_ops(load_traj(BASE))

hr("1b. PER PHYSICAL OPERATOR (baseline 0s) - heavy ops only (mean latency > 1s)")
per_op = defaultdict(list)
for r in rows0:
    if r["outcome"] == "ok":
        per_op[r["full_op_id"]].append(r["latency_s"])
print(f"{'logical':<14} {'physical':<24} {'class':<34} {'n':>3} {'med':>6} {'p90':>6} {'max':>6} {'sum':>7} {'quality':>7} status")
op_table = []
for oid, lats in per_op.items():
    if np.mean(lats) < 1.0:
        continue
    f = fin0.get(oid, {})
    q = f.get("metrics", {}).get("mean", {}).get("quality", float("nan"))
    cls = f.get("metadata", {}).get("operator_class", "?")
    op_table.append((f.get("logical", "?"), oid, cls, lats, q, f.get("status", "?")))
op_table.sort(key=lambda x: (x[0], -np.median(x[3])))
for lg, oid, cls, lats, q, st in op_table:
    print(f"{lg:<14} {oid:<24} {cls[:34]:<34} {len(lats):>3} {np.median(lats):6.1f} {np.quantile(lats,.9):6.1f} "
          f"{max(lats):6.1f} {sum(lats):7.0f} {q:7.3f} {st}")

# ----------------------------------------------------------------------------- 2
hr("2. IS LATENCY PREDICTIVE OF QUALITY? (within each logical op, baseline 0s)")
by_logical = defaultdict(list)
for lg, oid, cls, lats, q, st in op_table:
    by_logical[lg].append((np.median(lats), q, st))
for lg, ops in by_logical.items():
    if len(ops) < 3:
        continue
    med = np.array([o[0] for o in ops]); q = np.array([o[1] for o in ops])
    rho = np.corrcoef(np.argsort(np.argsort(med)), np.argsort(np.argsort(q)))[0, 1]
    par = [o for o in ops if o[2] == "PARETO_OPTIMAL"]
    print(f"  {lg}: {len(ops)} ops, Spearman(latency, quality) = {rho:+.2f}; "
          f"Pareto ops median latency = {[round(o[0],1) for o in par]}, "
          f"max median latency among all = {med.max():.1f}s")

# ----------------------------------------------------------------------------- 3
hr("3. OFFLINE REPLAY OF TIMEOUT EQUATIONS ON THE 0s RUN")
print("T(op) = k * Quantile_q(successful latencies of logical op so far), warmup=5, no clamp")
print("saved = sum(lat - T) over cut samples; false_cut = cut samples whose operator ends PARETO_OPTIMAL\n")
total0 = sum(r["latency_s"] for r in rows0)
pareto_ids = {o for o, v in fin0.items() if v.get("status") == "PARETO_OPTIMAL"}


def replay(q, k, warmup=5, floor=0.0):
    hist = defaultdict(list)
    saved = cut = false_cut = 0
    for r in rows0:
        lg, lat = r["logical_op_id"], r["latency_s"]
        h = hist[lg]
        T = None
        if len(h) >= warmup:
            T = max(k * float(np.quantile(h, q)), floor)
        if T is not None and lat > T:
            saved += lat - T
            cut += 1
            false_cut += r["full_op_id"] in pareto_ids
            # a cut sample is NOT added to history (matches the implementation)
        else:
            h.append(lat)
    return saved, cut, false_cut


results = []
for q in [0.5, 0.75, 0.85, 0.9, 0.95]:
    for k in [1.0, 1.2, 1.5, 2.0]:
        s, c, fc = replay(q, k)
        results.append((q, k, s, c, fc))
print(f"{'q':>5} {'k':>4} {'saved_s':>8} {'saved%':>7} {'cuts':>5} {'false_cuts':>10} {'false%':>7}")
for q, k, s, c, fc in results:
    print(f"{q:5.2f} {k:4.1f} {s:8.0f} {100*s/total0:6.1f}% {c:5d} {fc:10d} {100*fc/max(c,1):6.1f}%")

# ----------------------------------------------------------------------------- 3b
hr("3b. PEER-RELATIVE TIMEOUT: T = k * median latency of the FASTEST-GOOD peer")
print("Idea: a physical op is only worth waiting for while it could still beat its peers.")
print("Here we just report, per logical op, how latency of Pareto vs non-Pareto ops separates.")
for lg in sorted({r["logical_op_id"] for r in rows0}):
    lp = [r["latency_s"] for r in rows0 if r["logical_op_id"] == lg and r["full_op_id"] in pareto_ids]
    ln = [r["latency_s"] for r in rows0 if r["logical_op_id"] == lg and r["full_op_id"] not in pareto_ids]
    if not lp or not ln or max(lp + ln) < 1.0:
        continue
    print(f"  {lg}: PARETO   n={len(lp):3d} p50={np.median(lp):6.1f} p90={np.quantile(lp,.9):6.1f} max={max(lp):6.1f}")
    print(f"  {'':<{len(lg)}}  NON-PAR  n={len(ln):3d} p50={np.median(ln):6.1f} p90={np.quantile(ln,.9):6.1f} max={max(ln):6.1f}"
          f"  | time on non-Pareto = {sum(ln):.0f}s")

# ----------------------------------------------------------------------------- 4
hr("4. EARLY-PRUNING RULES ON TRAJECTORY SNAPSHOTS (baseline 0s)")
print("Rule: retire op at first snapshot where samples_drawn >= N and mean quality < tau.")
print("Saved samples = final samples_drawn - samples at prune time. Lost = final-Pareto ops pruned.\n")
traj0 = load_traj(BASE)
print(f"{'N':>3} {'tau':>5} {'pruned':>6} {'saved_samples':>13} {'saved_s(est)':>12} {'pareto_lost':>11}")
for N in [2, 3, 4, 6]:
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25]:
        pruned_at = {}
        for snap in traj0:
            for oid, v in snap["physical_operators"].items():
                if oid in pruned_at:
                    continue
                if v["samples_drawn"] >= N and v["metrics"]["mean"]["quality"] < tau:
                    pruned_at[oid] = v["samples_drawn"]
        saved_n = saved_s = lost = 0
        for oid, n_at in pruned_at.items():
            f = fin0[oid]
            extra = f["samples_drawn"] - n_at
            saved_n += extra
            saved_s += extra * f["metrics"]["mean"]["time"]
            lost += f["status"] == "PARETO_OPTIMAL"
        print(f"{N:3d} {tau:5.2f} {len(pruned_at):6d} {saved_n:13d} {saved_s:12.0f} {lost:11d}")
