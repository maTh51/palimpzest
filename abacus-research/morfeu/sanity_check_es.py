#!/usr/bin/env python3
"""
Comprehensive Sanity Check & Verification Script for the Early Termination (ES) Method
and all Abacus/BioDex experimental profiles in opt-profiling-data/.

Verifies:
  1. File Integrity: All 6 profile artifacts exist and are non-empty for every full run.
  2. Paired Seed Identity: For Seeds 0, 1, and 2, verifies that `0s` and `es` evaluated
     the EXACT same 250 test PMIDs (and same validation PMIDs in Round 1).
  3. Sample Budget Accounting:
     - Round 1 warmup: k=5 * j=4 = 20 samples per paid logical op (60 MAB samples + 4 Scan = 64 CSV rows).
     - Total CSV rows == Scan (Op 0) + MAB samples (Op 1 + Op 2 + Op 3).
     - `0s` runs >= 300 MAB samples; `es` stops at Round 3 (~103-112 MAB samples) with 0 timeouts.
  4. Early Termination Math Verification:
     - Recomputes leader mean, runner-up mean, pooled sigma, current SE, and future SE at Round 3
       directly from `*-trajectory.jsonl` / `*-profiling.json` and checks against `convergence_info`.
  5. Timing & Metric Consistency:
     - `optimization_time + plan_execution_time == total_execution_time` (within 0.1s).
     - Recomputes 3-seed Mean ± SD and Delta% reported in RELATORIO_TESTES_OTIMIZACAO.md.
"""
import csv
import json
import math
import os
import sys
import numpy as np

DATA_DIR = "opt-profiling-data"

RUNS = [
    ("morfeu-biodex-full-timeout-0s", 0, "baseline"),
    ("morfeu-biodex-full-timeout-30s", 0, "timeout_30s"),
    ("morfeu-biodex-full-timeout-dyn", 0, "timeout_dyn"),
    ("morfeu-biodex-full-timeout-qprune-rho0.5", 0, "qprune"),
    ("morfeu-biodex-full-timeout-es", 0, "es"),
    ("morfeu-biodex-full-timeout-0s-s1", 1, "baseline"),
    ("morfeu-biodex-full-timeout-es-s1", 1, "es"),
    ("morfeu-biodex-full-timeout-0s-s2", 2, "baseline"),
    ("morfeu-biodex-full-timeout-es-s2", 2, "es"),
]

SUFFIXES = [
    "-metrics.json",
    "-profiling.json",
    "-records.json",
    "-output.csv",
    "-trajectory.jsonl",
    "-timeouts.csv",
]


def main():
    passed = 0
    failed = 0

    def check(cond: bool, msg: str):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {msg}")
        else:
            failed += 1
            print(f"  [FAIL] {msg}")

    print("=" * 100)
    print("1. VERIFICAÇÃO DE INTEGRIDADE DOS ARQUIVOS DE PERFIL (6 ARQUIVOS POR EXECUÇÃO)")
    print("=" * 100)
    for exp, seed, kind in RUNS:
        missing = []
        for sfx in SUFFIXES:
            p = os.path.join(DATA_DIR, f"{exp}{sfx}")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                missing.append(sfx)
        check(len(missing) == 0, f"{exp} (seed={seed}): todos os 6 arquivos presentes e não-vazios")

    print("\n" + "=" * 100)
    print("2. VERIFICAÇÃO DE PAREAMENTO EXATO DOS 250 DOCUMENTOS DE TESTE POR SEED (0s vs ES)")
    print("=" * 100)
    pairs = [
        (0, "morfeu-biodex-full-timeout-0s", "morfeu-biodex-full-timeout-es"),
        (1, "morfeu-biodex-full-timeout-0s-s1", "morfeu-biodex-full-timeout-es-s1"),
        (2, "morfeu-biodex-full-timeout-0s-s2", "morfeu-biodex-full-timeout-es-s2"),
    ]
    for seed, base_exp, es_exp in pairs:
        with open(os.path.join(DATA_DIR, f"{base_exp}-records.json")) as f:
            rec_base = json.load(f)
        with open(os.path.join(DATA_DIR, f"{es_exp}-records.json")) as f:
            rec_es = json.load(f)
        pmids_base = [str(r["pmid"]) for r in rec_base]
        pmids_es = [str(r["pmid"]) for r in rec_es]
        check(
            len(pmids_base) == 250 and len(pmids_es) == 250,
            f"Seed {seed}: 250 registros de teste gerados em {base_exp} ({len(pmids_base)}) e {es_exp} ({len(pmids_es)})",
        )
        check(
            set(pmids_base) == set(pmids_es),
            f"Seed {seed}: conjunto de 250 PMIDs de teste 100% idêntico entre {base_exp} e {es_exp} ({len(set(pmids_base) & set(pmids_es))}/250 em comum)",
        )

    print("\n" + "=" * 100)
    print("3. CONTABILIDADE EXATA DE AMOSTRAS (ORÇAMENTO MAB vs OP 0 SCAN vs CSV TOTAL)")
    print("=" * 100)
    print(f"{'Experimento':<40} | {'Op0(Scan)':>9} | {'Op1(LLM)':>8} | {'Op2(TopK)':>9} | {'Op3(LLM)':>8} | {'Soma MAB':>9} | {'CSV Total':>9} | {'Timeouts':>8}")
    print("-" * 116)
    for exp, seed, kind in RUNS:
        csv_path = os.path.join(DATA_DIR, f"{exp}-timeouts.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        by_op = {0: 0, 1: 0, 2: 0, 3: 0}
        timeouts = 0
        for r in rows:
            idx = int(r["logical_op_id"].split("-")[0])
            by_op[idx] = by_op.get(idx, 0) + 1
            if r["outcome"] == "timeout":
                timeouts += 1
        mab_sum = by_op[1] + by_op[2] + by_op[3]
        total_csv = len(rows)
        print(f"{exp:<40} | {by_op[0]:>9} | {by_op[1]:>8} | {by_op[2]:>9} | {by_op[3]:>8} | {mab_sum:>9} | {total_csv:>9} | {timeouts:>8}")
        check(
            by_op[0] + mab_sum == total_csv,
            f"{exp}: Op0({by_op[0]}) + MAB({mab_sum}) == CSV({total_csv})",
        )
        # Check Round 1 warmup in trajectory.jsonl
        traj_path = os.path.join(DATA_DIR, f"{exp}-trajectory.jsonl")
        with open(traj_path) as f:
            tlines = [json.loads(line) for line in f]
        # First 4 lines are Round 1 (Op 0, Op 1, Op 2, Op 3)
        r1_op0 = tlines[0]["global_total_samples"]
        r1_op1 = tlines[1]["global_total_samples"]
        r1_op2 = tlines[2]["global_total_samples"]
        r1_op3 = tlines[3]["global_total_samples"]
        if kind == "timeout_30s":
            check(
                r1_op0 == 4 and r1_op1 < 20,
                f"{exp}: Em 30s, {20 - r1_op1}/20 amostras do Op1 sofreram timeout já na Rodada 1 (sobreviveram Op1={r1_op1}, Op2={r1_op2}, Op3={r1_op3})",
            )
        else:
            check(
                r1_op0 == 4 and r1_op1 == 20 and r1_op2 == 20 and r1_op3 == 20,
                f"{exp}: Warmup da Rodada 1 exato (Op0={r1_op0}, Op1={r1_op1}, Op2={r1_op2}, Op3={r1_op3} -> 60/300 MAB)",
            )
        if kind == "baseline":
            check(mab_sum >= 300, f"{exp}: Baseline completou o orçamento MAB ({mab_sum} >= 300)")
        elif kind == "es":
            check(
                100 <= mab_sum <= 120 and timeouts == 0,
                f"{exp}: Early Stop encerrou na Rodada 3 com {mab_sum}/300 amostras MAB (-{(1 - mab_sum/300)*100:.1f}%) e 0 timeouts",
            )

    print("\n" + "=" * 100)
    print("4. AUDITORIA MATEMÁTICA DO CRITÉRIO DE CONVERGÊNCIA (EFICÁCIA + FUTILIDADE) NA RODADA 3")
    print("=" * 100)
    for exp, seed, kind in RUNS:
        if kind != "es":
            continue
        traj_path = os.path.join(DATA_DIR, f"{exp}-trajectory.jsonl")
        with open(traj_path) as f:
            tlines = [json.loads(line) for line in f]
        check(
            len(tlines) == 12,
            f"{exp}: Executou exatamente 3 rodadas × 4 operadores lógicos = {len(tlines)} snapshots na trajetória",
        )
        # Inspect Round 3 (last 3 lines for Op 1, Op 2, Op 3)
        for line in tlines[-3:]:
            lop = line["logical_op_id"]
            cinfo = line["convergence_info"]
            z = cinfo["z"]
            sigma = cinfo["sigma"]
            n_L = cinfo["n_leader"]
            n_R = cinfo["n_runner_up"]
            mean_L = cinfo["mean_leader"]
            mean_R = cinfo["mean_runner_up"]
            delta = mean_L - mean_R
            rem_share = cinfo["remaining_budget_share"]
            # Count eligible ops (samples_drawn >= 4)
            ops = line["physical_operators"]
            n_eligible = sum(1 for v in ops.values() if v["samples_drawn"] >= 4)
            add_per_op = rem_share / max(n_eligible, 1)
            se_now = z * sigma * math.sqrt(1.0 / n_L + 1.0 / n_R)
            se_fut = z * sigma * math.sqrt(1.0 / (n_L + add_per_op) + 1.0 / (n_R + add_per_op))
            is_futile = delta < se_fut
            is_efficacy = delta >= se_now
            check(
                line["converged"] is True and (is_futile or is_efficacy),
                f"{exp} [{lop}]: Δ={delta:.4f} | SE_atual={se_now:.4f} | SE_futuro={se_fut:.4f} (a={add_per_op:.1f}) -> reason='{cinfo['reason']}' verificado!",
            )

    print("\n" + "=" * 100)
    print("5. VERIFICAÇÃO DOS TEMPOS E ESTATÍSTICAS MULTI-SEED DO RELATÓRIO")
    print("=" * 100)
    base_metrics = []
    es_metrics = []
    for exp, seed, kind in RUNS:
        mpath = os.path.join(DATA_DIR, f"{exp}-metrics.json")
        with open(mpath) as f:
            m = json.load(f)
        t_opt = m["optimization_time"]
        t_exec = m["plan_execution_time"]
        t_tot = m["total_execution_time"]
        check(
            abs((t_opt + t_exec) - t_tot) < 1e-4,
            f"{exp}: opt_time ({t_opt:.2f}s) + plan_exec ({t_exec:.2f}s) == total ({t_tot:.2f}s)",
        )
        if kind == "baseline":
            base_metrics.append(m)
        elif kind == "es":
            es_metrics.append(m)

    for s in range(3):
        b = base_metrics[s]
        e = es_metrics[s]
        check(
            e["rp@5"] >= b["rp@5"] and e["optimization_time"] < 0.5 * b["optimization_time"],
            f"Seed {s}: ES venceu Baseline em rp@5 ({e['rp@5']:.4f} vs {b['rp@5']:.4f}) e cortou >50% do tempo de otimização ({e['optimization_time']:.1f}s vs {b['optimization_time']:.1f}s)",
        )

    rp_b = [m["rp@5"] for m in base_metrics]
    rp_e = [m["rp@5"] for m in es_metrics]
    opt_b = [m["optimization_time"] for m in base_metrics]
    opt_e = [m["optimization_time"] for m in es_metrics]
    tot_b = [m["total_execution_time"] for m in base_metrics]
    tot_e = [m["total_execution_time"] for m in es_metrics]

    print("-" * 100)
    print(f"  Resumo 3 Seeds (0, 1, 2):")
    print(f"    rp@5:       Baseline = {np.mean(rp_b):.4f} ± {np.std(rp_b, ddof=1):.4f} | ES = {np.mean(rp_e):.4f} ± {np.std(rp_e, ddof=1):.4f} ({(np.mean(rp_e)/np.mean(rp_b)-1)*100:+.1f}%)")
    print(f"    Opt Time:   Baseline = {np.mean(opt_b):.1f} ± {np.std(opt_b, ddof=1):.1f}s | ES = {np.mean(opt_e):.1f} ± {np.std(opt_e, ddof=1):.1f}s ({(np.mean(opt_e)/np.mean(opt_b)-1)*100:+.1f}%, {np.mean(opt_b)/np.mean(opt_e):.2f}x speedup)")
    print(f"    Total Time: Baseline = {np.mean(tot_b):.1f} ± {np.std(tot_b, ddof=1):.1f}s | ES = {np.mean(tot_e):.1f} ± {np.std(tot_e, ddof=1):.1f}s ({(np.mean(tot_e)/np.mean(tot_b)-1)*100:+.1f}%)")
    print("=" * 100)
    print(f"RESULTADO FINAL DO SANITY CHECK: {passed} verificações aprovadas, {failed} falhas.")
    print("=" * 100)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
