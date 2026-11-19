"""Offline replay of a noise-aware early-termination rule.

Facts measured from the profiling files (per-record qualities, see report):
  - per-sample quality std of heavy ops  sigma ~= 0.33  (range 0.25-0.42)
  - top operators differ by 0.00-0.10 in mean quality, i.e. 1-2 standard errors

Minimum difference the MAB can resolve with n samples per op (one-sided z):
  Delta_min(n) = z * sigma * sqrt(2/n)
  n=15 -> 0.20 ; n=25 -> 0.15 ; n=60 -> 0.10 ; n=260 -> 0.05      (z=1.645, sigma=0.33)
So with a 300-sample budget spread over ~10 operators, differences < ~0.15 between the top
operators can NOT be resolved; samples spent trying to resolve them are wasted.

Rule eps-converged (per logical op, evaluated at every trajectory snapshot):
  CI_i = z * sigma / sqrt(n_i)                    (ops with n_i >= N_MIN)
  leader = argmax mean;   converged iff for all other i:  (mean_i + CI_i) - (mean_L - CI_L) <= eps
  i.e. even optimistically no op beats the leader (pessimistically) by more than eps.
We report, for several eps, the stop point and the regret vs. the run's final leader.
"""
import json
import math
from collections import defaultdict

D = "opt-profiling-data"
RUNS = ["morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-dyn",
        "morfeu-biodex-full-timeout-30s", "morfeu-biodex-full-timeout-qprune-rho0.5"]
SIGMA, Z, N_MIN = 0.33, 1.645, 4

print("Delta_min(n) = z*sigma*sqrt(2/n):", {n: round(Z * SIGMA * math.sqrt(2 / n), 3) for n in [10, 15, 25, 40, 60, 100]})

for EPS in [0.10, 0.15, 0.20, 0.25]:
    rows = []
    for run in RUNS:
        traj = [json.loads(l) for l in open(f"{D}/{run}-trajectory.jsonl")]
        by_lg = defaultdict(list)
        for s in traj:
            by_lg[s["logical_op_id"]].append(s["physical_operators"])
        for lg, snaps in by_lg.items():
            if len({o for s in snaps for o, v in s.items() if v["metrics"]["mean"]["time"] > 1.0}) < 2:
                continue
            final_q = {o: v["metrics"]["mean"]["quality"] for s in snaps for o, v in s.items()}
            ntot = lambda s: sum(v["samples_drawn"] for v in s.values())
            total = ntot(snaps[-1])

            def lead(s):
                c = [(v["metrics"]["mean"]["quality"], o) for o, v in s.items() if v["samples_drawn"] >= N_MIN]
                return max(c)[1] if c else None
            final_leader = lead(snaps[-1])
            stop = None
            for i, s in enumerate(snaps):
                L = lead(s)
                if L is None:
                    continue
                ci = lambda v: Z * SIGMA / math.sqrt(v["samples_drawn"])
                lcb_L = s[L]["metrics"]["mean"]["quality"] - ci(s[L])
                worst = max((v["metrics"]["mean"]["quality"] + ci(v) - lcb_L
                             for o, v in s.items() if o != L and v["samples_drawn"] >= N_MIN), default=-1)
                if worst <= EPS:
                    stop = i
                    break
            if stop is None:
                rows.append((run, lg, 1.0, True, 0.0)); continue
            L = lead(snaps[stop])
            rows.append((run, lg, ntot(snaps[stop]) / total, L == final_leader, final_q[final_leader] - final_q[L]))
    fr = [r[2] for r in rows]
    print(f"\neps={EPS:.2f}: mean fraction of samples used {100*sum(fr)/len(fr):.1f}% (saved {100*(1-sum(fr)/len(fr)):.1f}%), "
          f"same leader {sum(r[3] for r in rows)}/{len(rows)}, max regret {max(r[4] for r in rows):.3f}")
    for r in rows:
        print(f"   {r[0][-14:]:>14} {r[1]}: used {100*r[2]:5.1f}%  {'OK ' if r[3] else 'DIFF'} regret={r[4]:+.3f}")
