import json, math, os, glob
import numpy as np

data_dir = "/home/amarante/Jobs/Kunumi/palimpzest/abacus-research/opt-profiling-data"
axia_dir = "/tmp/axia_collected/opt-profiling-data"

def load_prof(exp):
    for d in [axia_dir, data_dir]:
        p = os.path.join(d, f"{exp}-profiling.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None

def load_traj(exp):
    for d in [axia_dir, data_dir]:
        p = os.path.join(d, f"{exp}-trajectory.jsonl")
        if os.path.exists(p):
            return [json.loads(l) for l in open(p) if l.strip()]
    return []

def evaluate_stopping_on_baseline(exp, target_op, total_budget_per_op, es_min_updates=1, es_n_min=4, es_z=1.645, es_mode="both", gamma=1.0):
    prof = load_prof(exp)
    traj = load_traj(exp)
    if not traj:
        return None
    # Get per-operator per-record quality map from profiling.json if available
    op_rec_qual = {}
    if prof and prof.get("sentinel_plan_stats"):
        sps = list(prof["sentinel_plan_stats"].values())[0]
        op_stats = sps.get("operator_stats", {}).get(target_op, {})
        for full_op_id, st in op_stats.items():
            rec_map = {}
            for r in st.get("record_op_stats", []):
                rid = r.get("record_id")
                q = r.get("quality")
                if q is not None:
                    if rid not in rec_map or q > rec_map[rid]:
                        rec_map[rid] = q
            op_rec_qual[full_op_id] = rec_map

    steps = [s for s in traj if s["logical_op_id"] == target_op]
    for step_idx, s in enumerate(steps, 1):
        phys = s["physical_operators"]
        tot_samples = sum(v.get("samples_drawn", 0) for v in phys.values())
        rem_budget = max(total_budget_per_op - tot_samples, 0.0)
        if step_idx < es_min_updates:
            continue
        eligible = [k for k, v in phys.items() if v.get("samples_drawn", 0) >= es_n_min]
        if len(eligible) < 2:
            continue
        # compute pooled sigma from actual records processed up to this step
        ss, dof = 0.0, 0
        for k in eligible:
            recs = phys[k].get("records_processed_ids", [])
            qs = [op_rec_qual.get(k, {}).get(rid) for rid in recs if op_rec_qual.get(k, {}).get(rid) is not None]
            if len(qs) > 1:
                ss += float(np.var(qs, ddof=1)) * (len(qs) - 1)
                dof += len(qs) - 1
        if dof > 0:
            sigma = math.sqrt(ss / dof)
        else:
            sigma = 0.22
        ranked = sorted(eligible, key=lambda k: -phys[k]["metrics"]["mean"]["quality"])
        L, R = ranked[0], ranked[1]
        mL, nL = phys[L]["metrics"]["mean"]["quality"], phys[L]["samples_drawn"]
        mR, nR = phys[R]["metrics"]["mean"]["quality"], phys[R]["samples_drawn"]
        gap = mL - mR
        se_curr = es_z * sigma * math.sqrt(1.0/nL + 1.0/nR)
        lcb_L = mL - es_z * sigma / math.sqrt(nL)
        contenders = sum(
            1 for k in eligible
            if phys[k]["metrics"]["mean"]["quality"] + es_z * sigma / math.sqrt(phys[k]["samples_drawn"]) >= lcb_L
        )
        add = rem_budget / max(contenders, 1)
        se_fut = gamma * es_z * sigma * math.sqrt(1.0/(nL + add) + 1.0/(nR + add))
        reason = None
        if es_mode in ("both", "efficacy") and gap >= se_curr:
            reason = "efficacy"
        elif es_mode in ("both", "futility") and gap < se_fut:
            reason = "futility"
        elif es_mode == "efficacy" and contenders <= 1:
            reason = "efficacy_1_contender"
        if reason is not None:
            return {
                "stop_round": step_idx,
                "max_rounds": len(steps),
                "samples_at_stop": tot_samples,
                "total_samples": sum(v.get("samples_drawn", 0) for v in steps[-1]["physical_operators"].values()),
                "reason": reason,
                "leader": L[-10:],
                "final_baseline_leader": max(steps[-1]["physical_operators"].items(), key=lambda kv: kv[1]["metrics"]["mean"]["quality"])[0][-10:],
                "gap": round(gap, 4),
                "se_curr": round(se_curr, 4),
                "se_fut": round(se_fut, 4),
                "nL": nL, "nR": nR,
            }
    return {
        "stop_round": len(steps),
        "max_rounds": len(steps),
        "samples_at_stop": sum(v.get("samples_drawn", 0) for v in steps[-1]["physical_operators"].values()),
        "total_samples": sum(v.get("samples_drawn", 0) for v in steps[-1]["physical_operators"].values()),
        "reason": "budget_exhausted",
        "leader": max(steps[-1]["physical_operators"].items(), key=lambda kv: kv[1]["metrics"]["mean"]["quality"])[0][-10:],
        "final_baseline_leader": max(steps[-1]["physical_operators"].items(), key=lambda kv: kv[1]["metrics"]["mean"]["quality"])[0][-10:],
    }

print("=== TEST 1: Variação de ES_N_MIN (com ES_MIN_UPDATES=1, sem trava na rodada 3!) ===")
for exp, op, b in [
    ("axia-cuad-full-s0-baseline", "1-81dd1f8073", 100),
    ("axia-cuad-full-s1-baseline", "1-81dd1f8073", 100),
    ("axia-biodex-full-s0-baseline", "1-2aa5d04bc1", 100),
    ("axia-biodex-full-s0-baseline", "2-7991bbf1ca", 100),
    ("axia-biodex-full-s0-baseline", "3-7b7b565330", 100),
]:
    print(f"\n--- {exp} ({op}) ---")
    for n_min in [4, 5, 6, 8, 10, 12]:
        res = evaluate_stopping_on_baseline(exp, op, b, es_min_updates=1, es_n_min=n_min, es_z=1.645, es_mode="both", gamma=1.0)
        print(f"  ES_N_MIN={n_min:2d} | StopRound={res['stop_round']:2d}/{res['max_rounds']:2d} | Samples={res['samples_at_stop']:3d}/{res['total_samples']:3d} | Reason={res['reason']:10s} | Leader={res['leader']} (nL={res.get('nL')}, gap={res.get('gap')}, SE_fut={res.get('se_fut')})")

print("\n=== TEST 2: Variação da Largura da Zona Cinza (gamma / Orçamento B / Modo) com ES_MIN_UPDATES=1, ES_N_MIN=5 ===")
for exp, op in [
    ("axia-cuad-full-s0-baseline", "1-81dd1f8073"),
    ("axia-cuad-full-s1-baseline", "1-81dd1f8073"),
    ("axia-biodex-full-s0-baseline", "1-2aa5d04bc1"),
    ("axia-biodex-full-s0-baseline", "2-7991bbf1ca"),
    ("axia-biodex-full-s0-baseline", "3-7b7b565330"),
]:
    print(f"\n--- {exp} ({op}) ---")
    for cfg_name, b, mode, gam, z in [
        ("Dual Padrão (B=100, gamma=1.0)", 100, "both", 1.0, 1.645),
        ("Zona Cinza Ampla (B=250, gamma=0.6)", 250, "both", 0.6, 1.645),
        ("Zona Cinza Muito Ampla (B=400, gamma=0.4)", 400, "both", 0.4, 1.645),
        ("Ablação: Só Eficácia (z=1.28)", 100, "efficacy", 1.0, 1.282),
    ]:
        res = evaluate_stopping_on_baseline(exp, op, b, es_min_updates=1, es_n_min=5, es_z=z, es_mode=mode, gamma=gam)
        print(f"  {cfg_name:42s} | StopRound={res['stop_round']:2d}/{res['max_rounds']:2d} | Samples={res['samples_at_stop']:3d}/{res['total_samples']:3d} | Reason={res['reason']:18s} | Leader={res['leader']}")
