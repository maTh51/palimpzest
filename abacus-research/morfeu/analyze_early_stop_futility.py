"""Offline replay: efficacy + futility early termination (sequential-testing style).

For each logical op at each snapshot (ops with n >= N_MIN):
  L = leader (max mean quality), R = runner-up
  gap = mean_L - mean_R
  EFFICACY : gap >= z * sigma * sqrt(1/n_L + 1/n_R)                  (winner already resolved)
  FUTILITY : gap <  z * sigma * sqrt(1/n_L' + 1/n_R')                (can't be resolved even if
             n_X' = n_X + remaining / C   (remaining budget split over C contenders)
             AND the gap is already within noise now.
  contenders C = ops whose mean + z*sigma/sqrt(n) >= mean_L - z*sigma/sqrt(n_L)
Stop at the first snapshot where EFFICACY or FUTILITY holds (and >= MIN_SNAPS snapshots).
Regret is measured with the run's FINAL means (the most information we have).
"""
import json
import math
from collections import defaultdict

D = "opt-profiling-data"
RUNS = ["morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-dyn",
        "morfeu-biodex-full-timeout-30s", "morfeu-biodex-full-timeout-qprune-rho0.5"]
SIGMA, N_MIN = 0.33, 4

for Z in [1.0, 1.645]:
    for MIN_SNAPS in [2, 3, 4]:
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
                lead = lambda s: max([(v["metrics"]["mean"]["quality"], o) for o, v in s.items()
                                      if v["samples_drawn"] >= N_MIN], default=(None, None))[1]
                final_leader = lead(snaps[-1])
                stop, why = None, ""
                for i, s in enumerate(snaps):
                    if i + 1 < MIN_SNAPS:
                        continue
                    el = {o: v for o, v in s.items() if v["samples_drawn"] >= N_MIN}
                    if len(el) < 2:
                        continue
                    ranked = sorted(el, key=lambda o: -el[o]["metrics"]["mean"]["quality"])
                    L, R = ranked[0], ranked[1]
                    mL, nL = el[L]["metrics"]["mean"]["quality"], el[L]["samples_drawn"]
                    mR, nR = el[R]["metrics"]["mean"]["quality"], el[R]["samples_drawn"]
                    gap = mL - mR
                    if gap >= Z * SIGMA * math.sqrt(1 / nL + 1 / nR):
                        stop, why = i, "efficacy"; break
                    lcbL = mL - Z * SIGMA / math.sqrt(nL)
                    C = sum(1 for o in el if el[o]["metrics"]["mean"]["quality"] + Z * SIGMA / math.sqrt(el[o]["samples_drawn"]) >= lcbL)
                    remaining = total - ntot(s)
                    add = remaining / max(C, 1)
                    if gap < Z * SIGMA * math.sqrt(1 / (nL + add) + 1 / (nR + add)):
                        stop, why = i, "futility"; break
                if stop is None:
                    rows.append((run, lg, 1.0, True, 0.0, "none")); continue
                Ls = lead(snaps[stop])
                rows.append((run, lg, ntot(snaps[stop]) / total, Ls == final_leader, final_q[final_leader] - final_q[Ls], why))
        fr = [r[2] for r in rows]
        print(f"\nz={Z} min_snaps={MIN_SNAPS}: samples used {100*sum(fr)/len(fr):.1f}% (saved {100*(1-sum(fr)/len(fr)):.1f}%), "
              f"same leader {sum(r[3] for r in rows)}/{len(rows)}, max regret {max(r[4] for r in rows):.3f}, "
              f"mean regret {sum(r[4] for r in rows)/len(rows):.3f}")
        for r in rows:
            print(f"   {r[0][-14:]:>14} {r[1]}: used {100*r[2]:5.1f}% {r[5]:<9} {'OK ' if r[3] else 'DIFF'} regret={r[4]:+.3f}")
