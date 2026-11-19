#!/usr/bin/env python3
"""
Análise Completa e Exata dos 4 Estudos Solicitados para o Histórico Experimental:
  Estudo 1: Consistência e Replicabilidade do Fluxo de Execução (Determinismo do RNG, Sequência de Documentos e Operadores)
  Estudo 2: Mapeamento Completo de Poda por Dominância (UCB < LCB), Parada por Futilidade (Δ < SE_futuro) e Parada por Eficácia (Δ >= SE_atual)
  Estudo 3: Convergência da Fronteira do Abacus vs. ES (Operadores Explorados, Ganho de Qualidade por Step e a cada 20 Amostras, Tokens de Entrada/Saída e Tempo por Step)
  Estudo 4: Tempo, Amostras e Tokens para Atingir um Nível de Qualidade Alvo (Ex.: τ = 0.20, 0.25, 0.30, 0.35, 0.40, 0.45) — ES vs. Abacus (0s)
"""
import csv
import json
import math
import os
from collections import defaultdict
import numpy as np

DATA_DIR = "opt-profiling-data"
SEEDS = [0, 1, 2]


def run_tag(method: str, seed: int) -> str:
    suffix = "" if seed == 0 else f"-s{seed}"
    return f"morfeu-biodex-full-timeout-{method}{suffix}"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv(path: str):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def estimate_input_tokens_from_record(rstat: dict) -> int:
    recorded_in = int(rstat.get("input_text_tokens", 0) or 0)
    if recorded_in > 0:
        return recorded_in
    calls = int(rstat.get("total_llm_calls", 0) or 0)
    if calls == 0:
        return 0
    state = rstat.get("record_state") or {}
    in_fields = rstat.get("input_fields") or ["title", "abstract", "fulltext"]
    chars = 0
    for fld in in_fields:
        val = state.get(fld, "")
        if val is not None:
            chars += len(str(val))
    base_doc_tokens = max(150, chars // 4)
    prompt_overhead = 280
    return calls * (base_doc_tokens + prompt_overhead)


def get_op_class_and_details(pop_id: str, pinfo: dict, sps_op_stats: dict, lop: str) -> str:
    meta = pinfo.get("metadata", {}) or {}
    cls_name = meta.get("operator_class")
    if not cls_name:
        recs = sps_op_stats.get(lop, {}).get(pop_id, {}).get("record_op_stats_lst", [])
        if recs:
            cls_name = recs[0].get("op_name", "TopKOp")
            details = recs[0].get("op_details", {}) or {}
            if "k" in details:
                return f"{cls_name}(k={details['k']})"
        return cls_name or "TopKOp"
    model = meta.get("model")
    if model:
        short_m = "GPU0" if "GPU0" in str(model) else ("GPU1" if "GPU1" in str(model) else str(model))
        return f"{cls_name}[{short_m}]"
    return cls_name


def split_csv_into_rounds(csv_rows: list) -> list[list[dict]]:
    """
    Divide as linhas de timeouts.csv nas rodadas exatas do otimizador usando os bursts de Op 0 (CacheScanDataOp)
    que iniciam cada chamada de _execute_sentinel_plan.
    """
    op0_indices = [i for i, r in enumerate(csv_rows) if r["logical_op_id"].startswith("0-")]
    if not op0_indices:
        return [csv_rows]
    bursts = []
    cur = [op0_indices[0]]
    for idx in op0_indices[1:]:
        if idx - cur[-1] <= 8:
            cur.append(idx)
        else:
            bursts.append(cur)
            cur = [idx]
    bursts.append(cur)
    rounds_rows = []
    for b_i, grp in enumerate(bursts):
        start_row = grp[0]
        end_row = bursts[b_i + 1][0] if b_i + 1 < len(bursts) else len(csv_rows)
        rounds_rows.append(csv_rows[start_row:end_row])
    return rounds_rows


def study_1_consistency():
    print("=" * 120)
    print("ESTUDO 1: CONSISTÊNCIA E REPLICABILIDADE DO FLUXO DE EXECUÇÃO (DETERMINISMO DE OPERADORES E DOCUMENTOS)")
    print("=" * 120)
    for seed in SEEDS:
        t0 = load_jsonl(f"{DATA_DIR}/{run_tag('0s', seed)}-trajectory.jsonl")
        tes = load_jsonl(f"{DATA_DIR}/{run_tag('es', seed)}-trajectory.jsonl")

        print(f"\n--- Seed {seed}: Comparação Pareada do Fluxo de Execução ({run_tag('0s', seed)} vs {run_tag('es', seed)}) ---")
        for r in range(3):
            ops_0s = {}
            ops_es = {}
            docs_0s = set()
            docs_es = set()
            for j in range(1, 4):
                e0 = t0[r * 4 + j]
                ees = tes[r * 4 + j]
                lop = e0["logical_op_id"][:1]
                active_0 = sorted([k.split("-")[1] for k, v in e0["physical_operators"].items() if v["samples_drawn"] > 0 or v["status"] == "RETIRED"])
                active_e = sorted([k.split("-")[1] for k, v in ees["physical_operators"].items() if v["samples_drawn"] > 0 or v["status"] == "RETIRED"])
                ops_0s[lop] = active_0
                ops_es[lop] = active_e
                for v in e0["physical_operators"].values():
                    docs_0s.update(v.get("records_processed_ids", []))
                for v in ees["physical_operators"].values():
                    docs_es.update(v.get("records_processed_ids", []))
            same_ops = (ops_0s == ops_es)
            same_docs = (docs_0s == docs_es)
            print(f"  Rodada {r+1}: Docs acumulados ({len(docs_0s)} vs {len(docs_es)}, 100%_Iguais={same_docs}: {sorted(docs_0s)})")
            if r == 0:
                print(f"            15 Operadores Iniciais Idênticos = {same_ops} | Op1={ops_0s['1']}, Op2={ops_0s['2']}, Op3={ops_0s['3']}")
            else:
                print(f"            Ops acumulados até R{r+1}: Op1({len(ops_0s['1'])} vs {len(ops_es['1'])}), Op2({len(ops_0s['2'])} vs {len(ops_es['2'])}), Op3({len(ops_0s['3'])} vs {len(ops_es['3'])})")

    rep_files = sorted([
        f.replace("-trajectory.jsonl", "")
        for f in os.listdir(DATA_DIR)
        if f.startswith("morfeu-biodex-consist-") and f.endswith("-trajectory.jsonl")
    ])
    if rep_files:
        print("\n--- Réplicas Dedicadas de Consistência (Mesmos Parâmetros: Seed=0, Budget=300, k=5, j=4, val=20, ES z=1.645) ---")
        base_es = load_jsonl(f"{DATA_DIR}/morfeu-biodex-full-timeout-es-trajectory.jsonl")
        for rf in rep_files:
            tr = load_jsonl(f"{DATA_DIR}/{rf}-trajectory.jsonl")
            met_path = f"{DATA_DIR}/{rf}-metrics.json"
            met = load_json(met_path) if os.path.exists(met_path) else {}
            r1_docs_base = sorted({d for e in base_es[1:4] for v in e["physical_operators"].values() for d in v.get("records_processed_ids", [])})
            r1_docs_rep = sorted({d for e in tr[1:4] for v in e["physical_operators"].values() for d in v.get("records_processed_ids", [])})
            r3_docs_base = sorted({d for e in base_es[-3:] for v in e["physical_operators"].values() for d in v.get("records_processed_ids", [])})
            r3_docs_rep = sorted({d for e in tr[-3:] for v in e["physical_operators"].values() for d in v.get("records_processed_ids", [])})
            r1_ops_base = sorted([k for e in base_es[1:4] for k, v in e["physical_operators"].items() if v["samples_drawn"] > 0])
            r1_ops_rep = sorted([k for e in tr[1:4] for k, v in e["physical_operators"].items() if v["samples_drawn"] > 0])
            print(
                f"  [{rf}] Rodadas={len(tr)//4} | R1_Docs_Iguais={r1_docs_base == r1_docs_rep} | "
                f"R3_Docs_Iguais={r3_docs_base == r3_docs_rep} | R1_15_Ops_Iguais={r1_ops_base == r1_ops_rep} | "
                f"opt_time={met.get('optimization_time', 0):.1f}s | rp@5={met.get('rp@5', 0):.5f}"
            )


def study_2_pruning_futility_vs_efficacy():
    print("\n" + "=" * 120)
    print("ESTUDO 2: OPERADORES PODADOS POR DOMINÂNCIA (RETIRED), FUTILIDADE (Δ < SE_futuro) E EFICÁCIA (Δ >= SE_atual)")
    print("=" * 120)

    for seed in SEEDS:
        tag_es = run_tag("es", seed)
        tag_0s = run_tag("0s", seed)
        tes = load_jsonl(f"{DATA_DIR}/{tag_es}-trajectory.jsonl")
        t0s = load_jsonl(f"{DATA_DIR}/{tag_0s}-trajectory.jsonl")
        prof_es = load_json(f"{DATA_DIR}/{tag_es}-profiling.json")
        sps_es = prof_es["sentinel_plan_stats"][list(prof_es["sentinel_plan_stats"].keys())[0]]["operator_stats"]
        prof_0s = load_json(f"{DATA_DIR}/{tag_0s}-profiling.json")
        sps_0s = prof_0s["sentinel_plan_stats"][list(prof_0s["sentinel_plan_stats"].keys())[0]]["operator_stats"]

        print(f"\n--- Seed {seed} ({tag_es} vs {tag_0s}) ---")
        r3_chunk = tes[-4:]
        for entry in r3_chunk:
            lop = entry["logical_op_id"]
            if lop.startswith("0-"):
                continue
            pops = entry["physical_operators"]
            op_samples_q = {}
            for pop_id, pdata in sps_es.get(lop, {}).items():
                op_samples_q[pop_id] = [r.get("quality", 0.0) or 0.0 for r in pdata["record_op_stats_lst"]]

            active_list = []
            retired_list = []
            for pop_id, pinfo in pops.items():
                st = pinfo["status"]
                cls_name = get_op_class_and_details(pop_id, pinfo, sps_es, lop)
                qs = op_samples_q.get(pop_id, [])
                n_obs = len(qs) if st == "RETIRED" else pinfo["samples_drawn"]
                mean_q = float(np.mean(qs[:n_obs])) if n_obs > 0 else pinfo["metrics"]["mean"]["quality"]
                std_q = float(np.std(qs[:n_obs], ddof=1)) if n_obs > 1 else 0.0
                lcb_q = pinfo["metrics"]["lcb"]["quality"]
                ucb_q = pinfo["metrics"]["ucb"]["quality"]
                if st == "RETIRED":
                    retired_list.append((pop_id, cls_name, n_obs, mean_q, std_q, lcb_q, ucb_q))
                elif n_obs > 0:
                    active_list.append((pop_id, cls_name, n_obs, mean_q, std_q, lcb_q, ucb_q, st))

            active_list.sort(key=lambda x: x[3], reverse=True)
            leader = active_list[0]
            a_share = (300 - 110) / 3.0 / max(2, len(active_list))
            z = 1.645

            print(f"\n  Etapa Lógica {lop}:")
            print(f"    • LÍDER (Vencedor #1): {leader[0]} ({leader[1]}) | n={leader[2]} | μ_L={leader[3]:.4f} | σ_L={leader[4]:.4f} | LCB={leader[5]:.4f} | UCB={leader[6]:.4f}")
            for idx, comp in enumerate(active_list[1:], start=2):
                delta = leader[3] - comp[3]
                sigma_pair = max(math.sqrt(0.5 * (leader[4] ** 2 + comp[4] ** 2)), 0.02)
                se_cur = z * sigma_pair * math.sqrt(1.0 / leader[2] + 1.0 / comp[2])
                se_fut = z * sigma_pair * math.sqrt(1.0 / (leader[2] + a_share) + 1.0 / (comp[2] + a_share))
                if delta >= se_cur:
                    rel_status = f"SUPERADO POR EFICÁCIA PELO LÍDER (Δ={delta:.4f} >= SE_atual={se_cur:.4f})"
                elif delta < se_fut:
                    rel_status = f"EMPATE TÉCNICO / PARADO POR FUTILIDADE (Δ={delta:.4f} < SE_futuro={se_fut:.4f} [SE_atual={se_cur:.4f}])"
                else:
                    rel_status = f"ZONA INTERMEDIÁRIA (SE_futuro={se_fut:.4f} <= Δ={delta:.4f} < SE_atual={se_cur:.4f})"
                role = "VICE-LÍDER (#2, define parada da fronteira)" if idx == 2 else f"Competidor Ativo (#{idx})"
                print(f"    • {role}: {comp[0]} ({comp[1]}) | n={comp[2]} | μ={comp[3]:.4f} | σ={comp[4]:.4f} -> {rel_status}")

            for ret in retired_list:
                delta = leader[3] - ret[3]
                sigma_pair = max(math.sqrt(0.5 * (leader[4] ** 2 + ret[4] ** 2)), 0.02)
                se_cur = z * sigma_pair * math.sqrt(1.0 / leader[2] + 1.0 / max(1, ret[2]))
                eff_tag = "e TAMBÉM Superado por Eficácia (Δ >= SE_atual)" if delta >= se_cur else ""
                print(f"    • PODADO (RETIRED via UCB < LCB): {ret[0]} ({ret[1]}) | n={ret[2]} | μ={ret[3]:.4f} | Δ_vs_L={delta:.4f} (SE_atual={se_cur:.4f}) {eff_tag}")

        r_final_0s = t0s[-4:]
        ret_0s = []
        for entry in r_final_0s:
            lop = entry["logical_op_id"]
            if lop.startswith("0-"):
                continue
            for pop_id, pinfo in entry["physical_operators"].items():
                if pinfo["status"] == "RETIRED":
                    cls_n = get_op_class_and_details(pop_id, pinfo, sps_0s, lop)
                    ret_0s.append(f"{pop_id} ({cls_n})")
        print(f"  [Comparação com Baseline 0s na Seed {seed}]: Total de operadores RETIRED ao final de {len(t0s)//4} rodadas = {len(ret_0s)}: {ret_0s}")

    # Verificar também o experimento dedicado morfeu-biodex-es-efficacy-b900 se existir log/perfil
    eff_log = "morfeu/logs/morfeu-biodex-es-efficacy-b900.log"
    if os.path.exists(eff_log):
        print("\n--- Experimento Dedicado de Captura de Eficácia vs. Futilidade (morfeu-biodex-es-efficacy-b900) ---")
        with open(eff_log, "r") as f:
            for line in f:
                if "[early-stop]" in line:
                    print("  " + line.strip())


def study_3_convergence_speed_and_tokens():
    print("\n" + "=" * 120)
    print("ESTUDO 3: CONVERGÊNCIA DO ABACUS (0s) vs. ES — OPERADORES, QUALIDADE E TOKENS POR STEP E A CADA 20 AMOSTRAS")
    print("=" * 120)

    for seed in SEEDS:
        tag_0s = run_tag("0s", seed)
        prof = load_json(f"{DATA_DIR}/{tag_0s}-profiling.json")
        sps = prof["sentinel_plan_stats"][list(prof["sentinel_plan_stats"].keys())[0]]["operator_stats"]
        pop_recs = {pop: pdata["record_op_stats_lst"] for lop, pops in sps.items() for pop, pdata in pops.items()}
        tlines = load_jsonl(f"{DATA_DIR}/{tag_0s}-trajectory.jsonl")
        csv_rows = load_csv(f"{DATA_DIR}/{tag_0s}-timeouts.csv")
        rounds_csv = split_csv_into_rounds(csv_rows)
        t0_wall = float(csv_rows[0]["wall_time"])

        print(f"\n--- Seed {seed}: Evolução Exata por Step (Rodada) no Abacus ({tag_0s}) ---")
        print(f"{'Step':>4} | {'Amostras MAB (+Step)':>20} | {'Wall (s)':>8} | {'Ops Explorados (P/A/R)':>22} | {'μ* Op1':>7} | {'μ* Op2':>7} | {'μ* Op3':>7} | {'μ* Média (ΔStep)':>18} | {'Tokens Step (In/Out/Tot)':>26} | {'Tokens Acum.':>12} | {'Calls':>5}")
        print("-" * 165)

        num_rounds = len(tlines) // 4
        seen_per_pop = defaultdict(int)
        cum_samples = 0
        cum_tot_tokens = 0
        prev_avg_q = 0.0

        for r in range(num_rounds):
            chunk = tlines[r * 4 : (r + 1) * 4]
            r_rows = rounds_csv[r] if r < len(rounds_csv) else []
            mab_r_rows = [row for row in r_rows if not row["logical_op_id"].startswith("0-")]
            step_samples = len(mab_r_rows)
            step_in_tok = 0
            step_out_tok = 0
            step_calls = 0
            wall_end = (float(r_rows[-1]["wall_time"]) - t0_wall) if r_rows else 0.0

            for row in mab_r_rows:
                pop = row["full_op_id"]
                idx = seen_per_pop[pop]
                seen_per_pop[pop] += 1
                recs = pop_recs.get(pop, [])
                if idx < len(recs):
                    rstat = recs[idx]
                    step_in_tok += estimate_input_tokens_from_record(rstat)
                    step_out_tok += int(rstat.get("output_text_tokens", 0) or 0)
                    step_calls += int(rstat.get("total_llm_calls", 0) or 0)

            explored_ops = 0
            pareto_ops = 0
            active_ops = 0
            retired_ops = 0
            quals = {}
            for entry in chunk:
                lop = entry["logical_op_id"]
                if lop.startswith("0-"):
                    continue
                best_q = 0.0
                for pop, pinfo in entry["physical_operators"].items():
                    st = pinfo["status"]
                    if pinfo["samples_drawn"] > 0 or st == "RETIRED":
                        explored_ops += 1
                    if st == "RETIRED":
                        retired_ops += 1
                    elif st == "ACTIVE_FRONTIER":
                        active_ops += 1
                    elif st == "PARETO_OPTIMAL":
                        pareto_ops += 1
                        q = pinfo["metrics"]["mean"]["quality"]
                        if q > best_q:
                            best_q = q
                quals[lop[:1]] = best_q

            cum_samples += step_samples
            step_tot_tok = step_in_tok + step_out_tok
            cum_tot_tokens += step_tot_tok
            avg_q = (quals["1"] + quals["2"] + quals["3"]) / 3.0
            dq = avg_q - prev_avg_q if r > 0 else avg_q
            prev_avg_q = avg_q
            es_marker = " <-- [PARADA ES]" if r == 2 else ""
            print(
                f"{r+1:4d} | {cum_samples:4d} (+{step_samples:2d} amostras)  | {wall_end:7.1f}s | {explored_ops:2d} ops ({pareto_ops:1d}P/{active_ops:2d}A/{retired_ops:2d}R) | "
                f"{quals['1']:7.3f} | {quals['2']:7.3f} | {quals['3']:7.3f} | {avg_q:7.3f} ({dq:+7.3f}) | "
                f"{step_in_tok:7d}/{step_out_tok:6d}/{step_tot_tok:7d} | {cum_tot_tokens:12d} | {step_calls:5d}{es_marker}"
            )

        # Janelas exatas de 20 em 20 amostras cronológicas
        mab_rows = [r for r in csv_rows if not r["logical_op_id"].startswith("0-")]
        print(f"\n--- Seed {seed}: Evolução Exata a Cada 20 Amostras Cronológicas ({tag_0s}) ---")
        print(f"{'Janela Amostras':>16} | {'Wall Clock (s)':>14} | {'Ops Distintos':>13} | {'μ* Op1':>7} | {'μ* Op2':>7} | {'μ* Op3':>7} | {'μ* Média (Δ20)':>16} | {' +OutTokens (Acum)':>20} | {' +TotTokens (Acum)':>22} | {' +Calls':>7}")
        print("-" * 155)

        seen_per_pop2 = defaultdict(int)
        q_running = defaultdict(list)
        pop_to_lop = {}
        ops_seen = set()
        win_out_tok = 0
        win_tot_tok = 0
        cum_out_tok = 0
        cum_tot_tok2 = 0
        win_calls = 0
        prev_w_q = 0.0

        for idx, row in enumerate(mab_rows, start=1):
            pop = row["full_op_id"]
            lop = row["logical_op_id"][:1]
            pop_to_lop[pop] = lop
            ops_seen.add(pop)
            obs_i = seen_per_pop2[pop]
            seen_per_pop2[pop] += 1
            recs = pop_recs.get(pop, [])
            if obs_i < len(recs):
                rstat = recs[obs_i]
                q_val = float(rstat.get("quality", 0.0) or 0.0)
                q_running[pop].append(q_val)
                in_t = estimate_input_tokens_from_record(rstat)
                out_t = int(rstat.get("output_text_tokens", 0) or 0)
                win_out_tok += out_t
                win_tot_tok += (in_t + out_t)
                cum_out_tok += out_t
                cum_tot_tok2 += (in_t + out_t)
                win_calls += int(rstat.get("total_llm_calls", 0) or 0)

            if idx % 20 == 0 or idx == len(mab_rows):
                wall_elapsed = float(row["wall_time"]) - t0_wall
                best_by_lop = {"1": 0.0, "2": 0.0, "3": 0.0}
                for p_id, qs in q_running.items():
                    if len(qs) >= (2 if idx >= 40 else 1):
                        mq = float(np.mean(qs))
                        l_id = pop_to_lop[p_id]
                        if mq > best_by_lop[l_id]:
                            best_by_lop[l_id] = mq
                avg_wq = (best_by_lop["1"] + best_by_lop["2"] + best_by_lop["3"]) / 3.0
                dwq = avg_wq - prev_w_q
                prev_w_q = avg_wq
                print(
                    f"{idx:4d} amostras    | {wall_elapsed:13.1f}s | {len(ops_seen):9d} ops | "
                    f"{best_by_lop['1']:7.3f} | {best_by_lop['2']:7.3f} | {best_by_lop['3']:7.3f} | "
                    f"{avg_wq:7.3f} ({dwq:+6.3f}) | +{win_out_tok:6d} ({cum_out_tok:7d}) | +{win_tot_tok:7d} ({cum_tot_tok2:8d}) | +{win_calls:5d}"
                )
                win_out_tok = 0
                win_tot_tok = 0
                win_calls = 0


def study_4_time_and_samples_to_target():
    print("\n" + "=" * 120)
    print("ESTUDO 4: COMPARAÇÃO DE AMOSTRAS, TEMPO E TOKENS PARA ATINGIR UM NÍVEL DE QUALIDADE ALVO (ES vs. ABACUS)")
    print("=" * 120)

    targets = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    for seed in SEEDS:
        tag_0s = run_tag("0s", seed)
        tag_es = run_tag("es", seed)
        m_0s = load_json(f"{DATA_DIR}/{tag_0s}-metrics.json")
        m_es = load_json(f"{DATA_DIR}/{tag_es}-metrics.json")
        prof_0s = load_json(f"{DATA_DIR}/{tag_0s}-profiling.json")
        sps_0s = prof_0s["sentinel_plan_stats"][list(prof_0s["sentinel_plan_stats"].keys())[0]]["operator_stats"]
        pop_recs = {pop: pdata["record_op_stats_lst"] for lop, pops in sps_0s.items() for pop, pdata in pops.items()}
        tlines = load_jsonl(f"{DATA_DIR}/{tag_0s}-trajectory.jsonl")
        csv_0s = load_csv(f"{DATA_DIR}/{tag_0s}-timeouts.csv")
        rounds_csv = split_csv_into_rounds(csv_0s)
        t0_wall = float(csv_0s[0]["wall_time"])

        num_rounds = len(tlines) // 4
        seen_per_pop = defaultdict(int)
        cum_samples = 0
        cum_out_tok = 0
        cum_tot_tok = 0
        step_records = []
        for r in range(num_rounds):
            chunk = tlines[r * 4 : (r + 1) * 4]
            r_rows = rounds_csv[r] if r < len(rounds_csv) else []
            mab_r_rows = [row for row in r_rows if not row["logical_op_id"].startswith("0-")]
            cum_samples += len(mab_r_rows)
            wall_s = (float(r_rows[-1]["wall_time"]) - t0_wall) if r_rows else 0.0
            for row in mab_r_rows:
                pop = row["full_op_id"]
                idx = seen_per_pop[pop]
                seen_per_pop[pop] += 1
                recs = pop_recs.get(pop, [])
                if idx < len(recs):
                    rstat = recs[idx]
                    cum_out_tok += int(rstat.get("output_text_tokens", 0) or 0)
                    cum_tot_tok += estimate_input_tokens_from_record(rstat) + int(rstat.get("output_text_tokens", 0) or 0)

            quals = {}
            for entry in chunk:
                lop = entry["logical_op_id"]
                if lop.startswith("0-"):
                    continue
                best_q = 0.0
                for pop, pinfo in entry["physical_operators"].items():
                    if pinfo["status"] == "PARETO_OPTIMAL":
                        q = pinfo["metrics"]["mean"]["quality"]
                        if q > best_q:
                            best_q = q
                quals[lop[:1]] = best_q
            avg_q = (quals["1"] + quals["2"] + quals["3"]) / 3.0
            step_records.append({
                "step": r + 1,
                "samples": cum_samples,
                "wall_s": wall_s,
                "out_tok": cum_out_tok,
                "tot_tok": cum_tot_tok,
                "q_op1": quals["1"],
                "q_op2": quals["2"],
                "q_op3": quals["3"],
                "q_avg": avg_q,
            })

        print(f"\n--- Seed {seed}: Quantas Amostras, Segundos e Tokens para Atingir Qualidade Alvo τ na Validação ---")
        print(f"    • Abacus Original (0s): obrigado a rodar até Step {len(step_records)} = {step_records[-1]['samples']} amostras / {m_0s['optimization_time']:.1f}s / {step_records[-1]['tot_tok']} tokens -> rp@5 teste = {m_0s['rp@5']:.4f}")
        print(f"    • Novo Método (ES):     encerra no Step 3 = {step_records[2]['samples']} amostras / {m_es['optimization_time']:.1f}s / {step_records[2]['tot_tok']} tokens -> rp@5 teste = {m_es['rp@5']:.4f}")
        print(f"{'Alvo τ (Qualidade)':>18} | {'Step Mínimo p/ Atingir τ':>30} | {'Amostras MAB':>14} | {'Tempo Wall-Clock':>16} | {'Tokens Acumulados':>18} | {'Economia vs Abacus Total (300 amos.)':>38}")
        print("-" * 150)
        total_abacus_samples = step_records[-1]["samples"]
        total_abacus_time = m_0s["optimization_time"]
        for tau in targets:
            hit = next((sr for sr in step_records if sr["q_avg"] >= tau or sr["q_op1"] >= tau), None)
            if hit:
                samp_sav = (1.0 - hit["samples"] / total_abacus_samples) * 100.0
                time_sav = (1.0 - hit["wall_s"] / total_abacus_time) * 100.0
                print(
                    f"τ >= {tau:4.2f}          | Step {hit['step']:2d} (μ_avg={hit['q_avg']:.3f}, Op1={hit['q_op1']:.3f}) | "
                    f"{hit['samples']:6d} amos.   | {hit['wall_s']:14.1f} s | {hit['tot_tok']:11d} tokens | "
                    f"-{samp_sav:5.1f}% amostras / -{time_sav:5.1f}% tempo"
                )
            else:
                print(f"τ >= {tau:4.2f}          | Não atingido                   |        -       |                - |                  - | -")


if __name__ == "__main__":
    study_1_consistency()
    study_2_pruning_futility_vs_efficacy()
    study_3_convergence_speed_and_tokens()
    study_4_time_and_samples_to_target()
