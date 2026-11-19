"""Offline calibration of an early-termination (convergence) rule for the Abacus MAB.

Question: how much of the sample budget is spent AFTER the optimizer already knows which
physical operator it will pick?  And how early can we stop without changing that pick?

For every logical op with >1 heavy physical op, walk the trajectory snapshots in order.
  leader(t) = argmax mean quality among ops with >= N_MIN samples at snapshot t
  final leader = leader at the last snapshot (what the run actually used)
Rule "stable(P)": stop at the first snapshot where the leader has been the same for P
consecutive snapshots.  For each P we report:
  - correct   : stopped leader == final leader
  - regret    : final mean quality(final leader) - final mean quality(stopped leader)
  - frac      : fraction of this logical op's samples already drawn at the stop point
Run from abacus-research/:  python morfeu/analyze_early_stop.py
"""
import json
from collections import defaultdict

D = "opt-profiling-data"
RUNS = ["morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-dyn",
        "morfeu-biodex-full-timeout-30s", "morfeu-biodex-full-timeout-qprune-rho0.5"]
N_MIN = 4
PS = [1, 2, 3, 4, 5, 6, 8]

summary = defaultdict(list)
for run in RUNS:
    traj = [json.loads(l) for l in open(f"{D}/{run}-trajectory.jsonl")]
    by_lg = defaultdict(list)
    for s in traj:
        by_lg[s["logical_op_id"]].append(s["physical_operators"])
    print(f"\n### {run}")
    for lg, snaps in by_lg.items():
        heavy = {o for s in snaps for o, v in s.items() if v["metrics"]["mean"]["time"] > 1.0}
        if len(heavy) < 2:
            continue

        def leader(s):
            c = [(v["metrics"]["mean"]["quality"], o) for o, v in s.items() if v["samples_drawn"] >= N_MIN]
            return max(c)[1] if c else None

        def n_tot(s):
            return sum(v["samples_drawn"] for v in s.values())

        leaders = [leader(s) for s in snaps]
        final_leader = leaders[-1]
        final_q = {o: v["metrics"]["mean"]["quality"] for s in snaps for o, v in s.items()}
        total = n_tot(snaps[-1])
        # last snapshot index at which the leader changed
        last_change = max(i for i in range(len(leaders)) if i == 0 or leaders[i] != leaders[i - 1])
        print(f"  {lg}: {len(snaps)} snapshots, total samples {total}; final leader {final_leader} "
              f"(q={final_q[final_leader]:.3f}) established at snapshot {last_change} "
              f"= {100*n_tot(snaps[last_change])/total:.0f}% of samples")
        for P in PS:
            run_len, stop = 0, None
            for i, l in enumerate(leaders):
                run_len = run_len + 1 if (l is not None and i > 0 and l == leaders[i - 1]) else (1 if l else 0)
                if run_len >= P:
                    stop = i
                    break
            if stop is None:
                summary[P].append((True, 0.0, 1.0)); continue
            sl = leaders[stop]
            ok = sl == final_leader
            regret = final_q[final_leader] - final_q[sl]
            frac = n_tot(snaps[stop]) / total
            summary[P].append((ok, regret, frac))
            print(f"     P={P}: stop@snap {stop:2d} frac={100*frac:5.1f}%  leader={'OK ' if ok else 'DIFF'} regret={regret:+.3f}")

print("\n=== SUMMARY over all (run, logical op) pairs ===")
print(f"{'P':>3} {'correct':>8} {'max_regret':>10} {'mean_frac_used':>15} {'mean_saved':>10}")
for P in PS:
    r = summary[P]
    print(f"{P:3d} {sum(x[0] for x in r):3d}/{len(r):<4d} {max(x[1] for x in r):10.3f} "
          f"{100*sum(x[2] for x in r)/len(r):14.1f}% {100*(1-sum(x[2] for x in r)/len(r)):9.1f}%")
