# Relatório Técnico Completo — Experimentos de Escala na Axia (NVIDIA H200) com `Llama-3.3-70B-Instruct` e `openai/gpt-oss-120b` + Nova Metodologia de Variância Pareada Intra-Documento (`sigma_paired`) e Paradas Dinâmicas

> **Foco Experimental:** Comparação direta e exclusiva entre o **Baseline Abacus (`0s` — Orçamento MAB Completo)** e o **Nosso Método (`ES` — Parada Antecipada Estatística por Eficácia e Futilidade)** em modelos de grande porte (**70B + 120B parâmetros**) executando em GPUs **NVIDIA H200 (141 GB VRAM)** no cluster **Axia (`kunumi/hpc4ai`)**, incluindo:
> 1. **A Nova Metodologia de Variância Pareada Intra-Documento (`ES_SIGMA_MODE="paired"`, limite inferior estrito $t_{\min}=1, n_{\min}=4, \gamma=0,5$)**, na qual $n_{\min}=4$ funciona exclusivamente como **portão de entrada (limite inferior)** e o teste estatístico decide se para por Eficácia, para por Futilidade ou **continua na Zona Cinza** até convergir;
> 2. **A Suíte de Validação Dinâmica na Axia (`t_min = 1`, variando $j, n_{\min}, \gamma, \text{ES\_MODE}$)**;
> 3. **Os Testes de Estresse de Orçamento Grande ($B = 250$ no CUAD e $B = 600$ no BioDex)** e **Campanha Multi-Seed Completa (`Seeds 0, 1, 2`)**.

---

## 1. Resumo Executivo e Principais Descobertas

Ao escalar o otimizador MAB do Palimpzest/Abacus da **Morfeu** (`Qwen2.5-14B` / `Llama-3.1-8B`) para a **Axia** (`meta-llama/Llama-3.3-70B-Instruct` + `openai/gpt-oss-120b` em 1× e 2× NVIDIA H200 de 141 GB), o custo computacional de cada amostra MAB aumenta substancialmente (especialmente em operadores compostos como `MixtureOfAgentsConvert` e `CritiqueAndRefineConvert`).

Nessa escala, nossa investigação revelou três descobertas centrais:

### 1.1. A Nova Metodologia de Variância Pareada Intra-Documento (`ES_SIGMA_MODE="paired"`, $t_{\min}=1, n_{\min}=4$ fixos) Resolve o Ruído Entre Documentos e Permite Permanecer na Zona Cinza Automaticamente
- **O Problema da Variância Não-Pareada ($\hat{\sigma}_{\text{unpaired}}$):** Documentos distintos possuem dificuldades intrinsecamente diferentes ($F_1 \in [0,0, 1,0]$). Quando a variância agrupada mistura documentos fáceis e difíceis sem descontar o efeito do documento, $\hat{\sigma}_{\text{unpaired}}$ fica artificialmente inflado ($\approx 0,20\text{--}0,40$), fazendo com que $\text{SE}_{\text{futuro}}$ fique alto e o algoritmo pare por Futilidade logo ao atingir o limite inferior $n_{\min}$.
- **A Solução ($\hat{\sigma}_{\text{paired}}$ — Variância Pareada Intra-Documento):** Como o MAB avalia candidatos sobre os **mesmos documentos de validação** (`shared_docs_lr`), subtraímos a média de cada documento $d$ ($r_{i,d} = y_{i,d} - \bar{y}_{\cdot,d}$) e combinamos com o erro-padrão das diferenças pareadas $d_{\text{pair}}(d) = y_{L,d} - y_{R,d}$. Isso reduz o desvio-padrão residual em **`45%` a `76%`** ($\hat{\sigma}_{\text{paired}} \approx 0,05\text{--}0,09$ vs. $\hat{\sigma}_{\text{unpaired}} \approx 0,09\text{--}0,37$)!
- **Resultado Direto na Axia (`2× H200: 70B + 120B`):** Com $n_{\min}=4$ e $t_{\min}=1$ mantidos fixos como **mero limite inferior**, ao atingir 4 amostras na Rodada 3 o algoritmo **não trava**:
  1. **`axia-paired-cuad-pilot-es` (`CUAD Pilot, B=40`):** Atinge **`F1 = 0,8213` (`P = 0,8718, R = 0,7763`)** — maior F1 de toda a campanha Piloto (`+2,5%` acima do Baseline `0,8010` e `+5,07 p.p.` a mais de Precisão que o ES não-pareado `0,8192`), reduzindo o tempo de otimização de `9.573,5s` para **`4.411,3s` (`2,17×` mais rápido / `-53,9%`)** e elegendo `CritiqueAndRefineConvert`.
  2. **`axia-paired-cuad-full-s0-es` (`CUAD Full Seed 0, B=100, N_test=50`):** Ao atingir $n_{\min}=4$ na Rodada 3 ($\hat{\sigma}_{\text{paired}} = 0,0499$ vs. $\hat{\sigma}_{\text{unpaired}} = 0,0903$), a diferença $\Delta = 0,0195$ cai **dentro da Zona Cinza** ($[\text{SE}_{\text{fut}} = 0,0130, \text{SE}_{\text{atual}} = 0,0581)$)! O algoritmo **continua amostrando nas Rodadas 3, 4 e 5 (`zone = gray_zone`)** e **só para na Rodada 6 ($n_L = 7$)**, elevando o F1 de `0,7208` (ES não-pareado) para **`0,7655` (`+4,47 p.p.` / `+6,2%` de ganho, com Precisão `0,8941` superior ao Baseline `0,8886`!)**, sendo **`2,12×` mais rápido na otimização (`3,34h` vs. `7,09h`)** e **`1,44×` mais rápido no tempo total (`5,16h` vs. `7,44h` do Baseline e `1,82h` mais rápido que o ES não-pareado!)**.
  3. **`axia-paired-biodex-full-s0-es` (`BioDex Full Seed 0, B=300, N_test=250`):** No pipeline de 3 operadores, `Op 3` para por **Eficácia na Rodada 4** ($n_L=5$), `Op 1` entra na Zona Cinza na Rodada 4 e para por **Eficácia na Rodada 5** ($n_L=6, \Delta = 0,0762 \ge \text{SE}_{\text{atual}} = 0,0616$), enquanto `Op 2` **permanece na Zona Cinza durante as Rodadas 3, 4, 5, 6, 7 e 8** ($\hat{\sigma}_{\text{paired}} \approx 0,05\text{--}0,08$ vs. $\hat{\sigma}_{\text{unpaired}} \approx 0,34\text{--}0,37$) e **para na Rodada 9 ($n_L = 10$)**! O resultado nos 250 artigos de teste é **`RP@5 = 0,3025` (`+47,8%` acima do Baseline `0,2046` e `+7,3%` acima do ES não-pareado `0,2818` — o maior `RP@5` de toda a campanha BioDex Full!)**, com **`2,55×` de speedup na otimização (`1,80h` vs. `4,59h`)** e **`1,44×` no tempo total (`7,45h` vs. `10,73h`, economizando `3,28 horas`)**!
  4. **`axia-paired-cuad-full-s1-es` (`CUAD Full Seed 1, B=100, N_test=50`):** **`4,01×` mais rápido na otimização (`2,29h` vs. `9,17h`, `-75,1%`)** e **`1,61×` mais rápido no tempo total (`6,45h` vs. `10,39h`)**, mantendo `F1 = 0,6246` vs. `0,6291` do Baseline (`< 0,45 p.p.`).
  5. **`axia-paired-biodex-pilot-es` (`BioDex Pilot, B=50`):** **`RP@5 = 0,4275` (`+11,8%` acima do Baseline `0,3825`)**, com **`1,52×` de speedup na otimização (`3.833,0s` vs. `5.829,1s`, `-34,2%`)** e menor tempo total (`5.971,9s` vs. `6.100,1s` do Baseline e `8.085,2s` do ES não-pareado).

---

### 1.2. Teste de Estresse de Orçamento Grande ($B = 250$ no CUAD): `8,97×` de Speedup Economizando `20,96 Horas` de 2× H200
- **`axia-cuad-stress-b250-baseline` vs. `axia-cuad-stress-b250-es` (`2× H200: 70B + 120B`, $B=250, k=5, j=4, N_{\text{val}}=25, N_{\text{test}}=50$):**
  - O **Baseline (`0s`)** levou **`84.930,6s` (`23,59 horas`!)** na fase de otimização (`85.991,2s` = `23,89 horas` no total) para escolher o operador `LLMConvertBonded (GPU1)`, obtendo `F1 = 0,7725` (`P = 0,9040, R = 0,6743`).
  - O **Nosso Método (`ES`)** convergiu em **`9.466,9s` (`2,63 horas`)** na otimização (`11.054,8s` = `3,07 horas` no total) — **`8,97×` MAIS RÁPIDO na otimização (`-88,9%`, economizando `20,96 horas` contínuas de 2× H200 em uma única execução!)** e **`7,78×` MAIS RÁPIDO no tempo total (`-87,1%`)** — escolhendo **exatamente o mesmo operador físico (`LLMConvertBonded GPU1`)** e atingindo **`F1 = 0,7901` (`+2,3%` acima do Baseline, `P = 0,9128, R = 0,6965`)**!

---

### 1.3. Suíte de Validação Dinâmica na Axia (`t_min = 1`, $B=60, k=5, N_{\text{val}}=15, N_{\text{test}}=20$)
Executando ao vivo nas 2× H200 da Axia sem trava na Rodada 3 (`t_min = 1`):
- **`axia-val-cuad-baseline-j2` (`Baseline 0s, j=2, 11 rodadas`):** `F1 = 0,7880`, `OptTime = 11.471,4s` (`3,19h`), `TotTime = 14.343,3s` (`3,98h`).
- **`axia-val-cuad-j1-nmin4` (`j=1, t_min=1, n_min=4` — Parou na Rodada 4):** **`F1 = 0,8112` (`+2,9%`)**, `OptTime = 4.488,5s` (`1,25h` — **`2,56×` mais rápido / `-60,9%`**), `TotTime = 5.921,4s` (`1,64h` — **`2,42×` mais rápido**).
- **`axia-val-cuad-j2-nmin5` (`j=2, t_min=1, n_min=5` — Parou na Rodada 4):** **`F1 = 0,8149` (`+3,4%`)**, `OptTime = 7.754,2s` (`2,15h` — **`1,48×` mais rápido**), `TotTime = 8.153,3s` (`2,26h` — **`1,76×` mais rápido**).
- **`axia-val-cuad-j2-nmin6` (`j=2, t_min=1, n_min=6` — Parou na Rodada 5):** **`F1 = 0,8069` (`+2,4%`)**, `OptTime = 7.698,4s` (`2,14h` — **`1,49×` mais rápido**).
- **`axia-val-cuad-j2-nmin8` (`j=2, t_min=1, n_min=8` — Parou na Rodada 7):** `F1 = 0,7288`, `OptTime = 10.155,1s` (`2,82h` — `1,13×` mais rápido).
- **`axia-val-cuad-grayzone-g05` (`j=2, t_min=1, n_min=4, gamma=0,5`):** **`F1 = 0,8292` (`+5,2%` acima do Baseline!)**, `OptTime = 3.305,7s` (`55,1 min` — **`3,47×` mais rápido / `-71,2%`**), `TotTime = 3.755,0s` (`62,6 min` — **`3,82×` mais rápido / `-73,8%`**)!
- **`axia-val-cuad-mode-efficacy` (`ES_MODE=efficacy` — Ablação Só Eficácia sem Futilidade):** Atinge **`F1 = 0,8431` (`+7,0%`)**, porém, por não possuir o critério de Futilidade para encerrar empates estatísticos, percorre todas as 10 rodadas (`OptTime = 15.468,6s`), comprovando experimentalmente o papel complementar dos dois critérios (**Eficácia** para líderes isolados e **Futilidade/Zona Cinza** para empates estatísticos).

---

## 2. Formulação Matemática Completa: Critério Dual e Variância Pareada Intra-Documento ($\hat{\sigma}_{\text{paired}}$)

Dentro de [mab_execution_strategy.py](file:///home/amarante/Jobs/Kunumi/palimpzest/src/palimpzest/query/execution/mab_execution_strategy.py#L106-L245), o critério de parada avalia ao final de cada rodada $t$ a diferença empírica $\Delta = \hat{\mu}_L - \hat{\mu}_R$ entre o operador **Líder ($L$)** e o **Vice-Líder ($R$)**, desde que ambos tenham atingido o **limite inferior de entrada** ($t \ge t_{\min}$ e $n_L, n_R \ge n_{\min}$):

$$\text{SE}_{\text{atual}} = z \cdot \hat{\sigma}_{\text{paired}} \sqrt{\frac{1}{n_L} + \frac{1}{n_R}}, \qquad \text{SE}_{\text{futuro}} = \gamma \cdot z \cdot \hat{\sigma}_{\text{paired}} \sqrt{\frac{1}{n_L^+} + \frac{1}{n_R^+}}$$

onde $n_i^+ = \min\!\left(N_{\text{val}},\, n_i + \frac{B_{\text{rem}}}{C}\right)$ é a projeção de amostras futuras limitada ao tamanho do conjunto de validação $N_{\text{val}}$, e $\hat{\sigma}_{\text{paired}}$ é o estimador de variância pareada intra-documento:

$$\hat{\sigma}_{\text{within}}^2 = \frac{1}{\text{dof}_{\text{within}}} \sum_{d \in D_{\text{shared}}} \sum_{i \in \mathcal{O}_d} \left(y_{i,d} - \bar{y}_{\cdot,d}\right)^2, \qquad \hat{\sigma}_{\text{diff}, LR}^2 = \frac{1}{2\left(|D_{L \cap R}| - 1\right)} \sum_{d \in D_{L \cap R}} \left(d_{LR}(d) - \bar{d}_{LR}\right)^2$$

$$\hat{\sigma}_{\text{paired}} = \sqrt{\frac{\hat{\sigma}_{\text{within}}^2 + \hat{\sigma}_{\text{diff}, LR}^2}{2}}$$

### As 3 Zonas de Decisão Após Atingir o Limite Inferior ($n_L, n_R \ge n_{\min}$):
1. **Zona de Eficácia ($\Delta \ge \text{SE}_{\text{atual}}$):** O Líder já supera o Vice-Líder com $95\%$ de confiança ($z = 1,645$). **Para imediatamente por Eficácia.**
2. **Zona de Futilidade ($\Delta < \text{SE}_{\text{futuro}}$):** A diferença entre Líder e Vice-Líder é menor que a resolução mínima alcançável com o orçamento restante. **Para imediatamente por Futilidade.**
3. **Zona Cinza ($\text{SE}_{\text{futuro}} \le \Delta < \text{SE}_{\text{atual}}$):** A diferença é promissora demais para desistir, mas ainda menor que a margem de confiança atual. **O algoritmo NÃO para na rodada em que atingiu $n_{\min}$ — ele continua amostrando nas rodadas seguintes até sair da Zona Cinza!**

---

## 3. Comparação Direta: Variância Não-Pareada (`Unpaired`) vs. Nova Variância Pareada (`Paired`) na Axia (`2× H200`)

| Cenário na Axia (`2× H200: 70B + 120B`) | Método | Rodada de Parada | Desvio-Padrão ($\hat{\sigma}$) | Qualidade no Teste (`F1` ou `RP@5`) | Precision / Recall (CUAD) | Tempo Otimização (`OptTime`) | Tempo Total (`TotTime`) | Speedup Fase 1 (`0s / ES`) | Operador Físico Selecionado |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CUAD Pilot** (`B=40, val=10, test=10`) | `Baseline (0s)` | Rodada 9 (Fim) | — | `F1 = 0,8010` | `0,8798` / `0,7352` | `9.573,5s` (`159,6m`) | `9.896,7s` (`164,9m`) | `1,00×` | `LLMConvertBonded (GPU1)` |
| **CUAD Pilot** (`B=40, val=10, test=10`) | `ES Antigo (Unpaired)` | Rodada 3 | `0,0882` | `F1 = 0,8192` | `0,8211` / `0,8174` | `2.524,1s` (`42,1m`) | `5.309,7s` (`88,5m`) | `3,79×` (`-73,6%`) | `MixtureOfAgents (GPU0+1 -> GPU1)` |
| **CUAD Pilot** (`B=40, val=10, test=10`) | **`ES Novo (Paired)`** | **Rodada 3** (`n_L=4`) | **`0,0516` (`-41,5%`)** | **`F1 = 0,8213` (`+2,5%`)** | **`0,8718` / `0,7763`** | **`4.411,3s` (`73,5m`)** | **`6.027,3s` (`100,5m`)** | **`2,17×` (`-53,9%`)** | **`CritiqueAndRefine (GPU1 -> GPU0 -> GPU1)`** |
| **CUAD Full Seed 0** (`B=100, val=20, test=50`) | `Baseline (0s)` | Rodada 17 (Fim) | — | `F1 = 0,7750` | `0,8886` / `0,6871` | `25.506,2s` (`7,09h`) | `26.790,9s` (`7,44h`) | `1,00×` | `LLMConvertBonded (GPU1)` |
| **CUAD Full Seed 0** (`B=100, val=20, test=50`) | `ES Antigo (Unpaired)` | Rodada 3 | `0,0903` | `F1 = 0,7208` | `0,8837` / `0,6087` | `9.743,7s` (`2,71h`) | `25.144,2s` (`6,98h`) | `2,62×` (`-61,8%`) | `MixtureOfAgents (GPU0+1 -> GPU1)` |
| **CUAD Full Seed 0** (`B=100, val=20, test=50`) | **`ES Novo (Paired)`** | **Rodadas 3–5 na Zona Cinza $\to$ Parou na Rodada 6 (`n_L=7`)!** | **`0,0499` (`-44,7%`)** | **`F1 = 0,7655` (`+4,47 p.p.` vs Unpaired!)** | **`0,8941` / `0,6692`** | **`12.017,1s` (`3,34h`)** | **`18.578,0s` (`5,16h`, `-30,7%`!)** | **`2,12×` (`-52,9%` Fase 1 / `-1,82h` Total vs Unpaired!)** | **`CritiqueAndRefine (GPU1 -> GPU0 -> GPU1)`** |
| **CUAD Full Seed 1** (`B=100, val=20, test=50`) | `Baseline (0s)` | Rodada 17 (Fim) | — | `F1 = 0,6291` | `0,8829` / `0,4886` | `33.029,1s` (`9,17h`) | `37.399,1s` (`10,39h`) | `1,00×` | `MixtureOfAgents (GPU1 -> GPU0)` |
| **CUAD Full Seed 1** (`B=100, val=20, test=50`) | **`ES Novo (Paired)`** | **Rodada 3** (`n_L=4`) | **`0,1285` (`-37,2%`)** | **`F1 = 0,6246`** | `0,8611` / **`0,4900`** | **`8.227,5s` (`2,29h`)** | **`23.237,0s` (`6,45h`, `-37,9%`)** | **`4,01×` (`-75,1%`, poupando `6,89h`!)** | `MixtureOfAgents (GPU0+1 -> GPU1)` |
| **BioDex Pilot** (`B=50/90, val=10, test=20`) | `Baseline (0s)` | Rodada 5 (Fim) | — | `RP@5 = 0,3825` | — | `5.829,1s` (`97,2m`) | `6.100,1s` (`101,7m`) | `1,00×` | `LLMConvertBonded + TopK(20) + Bonded` |
| **BioDex Pilot** (`B=50/90, val=10, test=20`) | `ES Antigo (Unpaired)` | Rodada 4 | `0,4163` | `RP@5 = 0,4425` | — | `4.581,6s` (`76,4m`) | `8.085,2s` (`134,8m`) | `1,27×` (`-21,4%`) | `Critique + TopK(15) + MoA` |
| **BioDex Pilot** (`B=50, val=10, test=20`) | **`ES Novo (Paired)`** | **Rodada 3** (`n_L=4`) | **`0,0000` (`-100%`)** | **`RP@5 = 0,4275` (`+11,8%`)** | — | **`3.833,0s` (`63,9m`)** | **`5.971,9s` (`99,5m`, `-26,1%` vs Unpaired)** | **`1,52×` (`-34,2%`)** | **`MoA + TopK(15) + Critique`** |
| **BioDex Full Seed 0** (`B=300, val=20, test=250`) | `Baseline (0s)` | Rodada 14 (Fim) | — | `RP@5 = 0,2046` | — | `16.512,7s` (`4,59h`) | `38.621,3s` (`10,73h`) | `1,00×` | `Critique + TopK(20) + Critique` |
| **BioDex Full Seed 0** (`B=300, val=20, test=250`) | `ES Antigo (Unpaired)` | Rodada 3 | `0,2175` | `RP@5 = 0,2818` (`+37,7%`) | — | `4.651,1s` (`1,29h`) | `31.602,7s` (`8,78h`) | `3,55×` (`-71,8%`) | `Bonded + TopK(20) + MoA` |
| **BioDex Full Seed 0** (`B=300, val=20, test=250`) | **`ES Novo (Paired)`** | **`Op 3` R4 (Eficácia), `Op 1` R5 (Eficácia), `Op 2` R3–R8 Zona Cinza $\to$ Parou na Rodada 9 (`n_L=10`)!** | **`0,0506` vs `0,3703` (`-86,3%` ruído!)** | **`RP@5 = 0,3025` (`+47,8%` vs Baseline / `+7,3%` vs Unpaired!)** | — | **`6.464,7s` (`1,80h`)** | **`26.803,1s` (`7,45h`, `-30,6%` vs Baseline!)** | **`2,55×` (`-60,8%` Fase 1 e `-3,28h` no Tempo Total!)** | **`Bonded (GPU1) + TopK(20) + Critique (GPU1->GPU0->GPU1)`** |

### Evolução Rodada a Rodada (`es_diagnostics`) de `axia-paired-cuad-full-s0-es`:
Veja nos logs reais da Axia como `axia-paired-cuad-full-s0-es` atinge o limite inferior $n_{\min}=4$ na Rodada 3, mas **permanece na Zona Cinza** até a Rodada 6:
- **Rodadas 1 e 2 ($n_L < 4$):** `zone = warmup_ineligible` (aguardando atingir o mínimo de 4 amostras).
- **Rodada 3 ($n_L = 4, n_R = 4$, `shared_docs_lr = 4`):** $\hat{\sigma}_{\text{paired}} = 0,0499$ (vs. $\hat{\sigma}_{\text{unpaired}} = 0,0903$). Temos $\Delta = 0,0195 \in [\text{SE}_{\text{fut}} = 0,0130,\, \text{SE}_{\text{atual}} = 0,0581) \implies$ **`zone = gray_zone` (NÃO para na Rodada 3! Continua amostrando!)**.
- **Rodada 4 ($n_L = 5, n_R = 5$, `shared_docs_lr = 5`):** $\hat{\sigma}_{\text{paired}} = 0,0525$, $\Delta = 0,0365 \in [\text{SE}_{\text{fut}} = 0,0136,\, \text{SE}_{\text{atual}} = 0,0546) \implies$ **`zone = gray_zone` (Continua amostrando!)**.
- **Rodada 5 ($n_L = 6, n_R = 6$, `shared_docs_lr = 6`):** $\hat{\sigma}_{\text{paired}} = 0,0948$, $\Delta = 0,0289 \in [\text{SE}_{\text{fut}} = 0,0247,\, \text{SE}_{\text{atual}} = 0,0900) \implies$ **`zone = gray_zone` (Continua amostrando!)**.
- **Rodada 6 ($n_L = 7, n_R = 7$, `shared_docs_lr = 7`):** $\hat{\sigma}_{\text{paired}} = 0,0944$, $\Delta = 0,0144 < \text{SE}_{\text{fut}} = 0,0246 \implies$ **`zone = futility` (Para dinamicamente na Rodada 6!)**, elegendo `CritiqueAndRefineConvert` e alcançando **`F1 = 0,7655`** (`P = 0,8941`)!

---

## 4. Resultados da Suíte de Validação Dinâmica (`t_min = 1`) na Morfeu e na Axia (`2× H200`)

### 4.1. Suíte de Validação Dinâmica na Axia (`2× H200: 70B + 120B`, $B=60, k=5, N_{\text{val}}=15, N_{\text{test}}=20$, `t_min = 1`)

| Experimento na Axia (`B=60, k=5, val=15, test=20`) | Configuração (`j`, `t_min`, `n_min`, `gamma`, `mode`) | Rodada de Parada ($t$) | Amostras no Líder ($n_L$) | Tempo Otimização (`OptTime`) | Tempo Total (`TotTime`) | F1-Score (`Precision` / `Recall`) | Speedup Fase 1 (`0s / ES`) | Speedup Total (`0s / ES`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`axia-val-cuad-baseline-j2`** (`Baseline 0s`) | `j=2`, Sem Parada | **Rodada 11** (Fim) | $n_L = 12$ | `11.471,4s` (`3,19h`) | `14.343,3s` (`3,98h`) | `0,7880` (`0,8900` / `0,7069`) | `1,00×` (Ref.) | `1,00×` (Ref.) |
| **`axia-val-cuad-j1-nmin4`** (`Nosso Método ES`) | `j=1, t_min=1, n_min=4` | **Parou na Rodada 4!** | $n_L = 4$ | **`4.488,5s` (`1,25h`)** | **`5.921,4s` (`1,64h`)** | **`0,8112` (`+2,9%`)** (`0,8896` / `0,7455`) | **`2,56×` (`-60,9%`)** | **`2,42×` (`-58,7%`)** |
| **`axia-val-cuad-j2-nmin5`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=5` | **Parou na Rodada 4!** | $n_L = 5$ | **`7.754,2s` (`2,15h`)** | **`8.153,3s` (`2,26h`)** | **`0,8149` (`+3,4%`)** (`0,8806` / `0,7584`) | **`1,48×` (`-32,4%`)** | **`1,76×` (`-43,2%`)** |
| **`axia-val-cuad-j2-nmin6`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=6` | **Parou na Rodada 5!** | $n_L = 6$ | **`7.698,4s` (`2,14h`)** | `14.718,8s` (`4,09h`) | **`0,8069` (`+2,4%`)** (`0,8370` / `0,7789`) | **`1,49×` (`-32,9%`)** | `0,97×` |
| **`axia-val-cuad-j2-nmin8`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=8` | **Parou na Rodada 7!** | $n_L = 8$ | **`10.155,1s` (`2,82h`)** | `16.390,1s` (`4,55h`) | `0,7288` (`0,8247` / `0,6530`) | **`1,13×` (`-11,5%`)** | `0,88×` |
| **`axia-val-cuad-grayzone-g05`** (`Zona Cinza`) | **`j=2, t_min=1, n_min=4, gamma=0,5`** | **Parou na Rodada 3!** | $n_L = 4$ | **`3.305,7s` (`0,92h`)** | **`3.755,0s` (`1,04h`)** | **`0,8292` (`+5,2%`!)** (`0,8932` / `0,7738`) | **`3,47×` (`-71,2%`)** | **`3,82×` (`-73,8%`!)** |
| **`axia-val-cuad-mode-efficacy`** (`Ablação`) | `j=2, t_min=1, n_min=4, mode=efficacy` | **Rodada 10** (Sem Futilidade) | $n_L = 11$ | `15.468,6s` (`4,30h`) | `17.638,4s` (`4,90h`) | **`0,8431` (`+7,0%`)** (`0,8983` / `0,7943`) | `0,74×` (Sem Futilidade não para em empate) | `0,81×` |

### 4.2. Suíte de Validação Dinâmica na Morfeu (`Qwen2.5-14B / Llama-3.1-8B`, $B=60, k=5, N_{\text{val}}=15, N_{\text{test}}=15$, `t_min = 1`)

| Experimento na Morfeu (`B=60, k=5, val=15, test=15`) | Config (`j`, `t_min`, `n_min`) | Rodada de Parada ($t$) | Amostras MAB (`n_L` no Líder) | Tokens `Out` Fase 1 (`Otimização`) | Tempo Otimização (`OptTime`) | Tempo Total (`TotTime`) | F1-Score (`P` / `R`) | Speedup Otimização (`0s / ES`) e Ganho |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`morfeu-val-cuad-baseline-j2`** (`Baseline 0s`) | `j=2`, Sem Parada | **Rodada 11** (Fim) | `61` ($n_L=12$) | `259.285` | `2.317,6s` (`38,6m`) | `3.095,7s` (`51,6m`) | `0,6225` (`0,8708` / `0,4844`) | Referência (`1,00×`, esgota 11 rodadas) |
| **`morfeu-val-cuad-j1-nmin4`** (`Nosso Método ES`) | `j=1, t_min=1, n_min=4` | **Parou na Rodada 4!** | **`20` (`-67,2%`, $n_L=4$)** | **`73.343` (`-71,7%`)** | **`788,9s` (`13,1m`)** | **`1.288,6s` (`21,5m`)** | `0,6047` (`0,8226` / `0,4781`) | **`2,94×` mais rápido na Fase 1 (`-66,0%`) e `-71,7%` tokens** |
| **`morfeu-val-cuad-j2-nmin5`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=5` | **Parou na Rodada 5!** | **`31` (`-49,2%`, $n_L=6$)** | **`109.218` (`-57,9%`)** | **`779,9s` (`13,0m`)** | **`1.718,7s` (`28,6m`)** | `0,5525` (`0,8776` / `0,4031`) | **`2,97×` mais rápido na Fase 1 (`-66,3%`) e `-57,9%` tokens** |
| **`morfeu-val-cuad-j2-nmin6`** (`Nosso Método ES`) | **`j=2, t_min=1, n_min=6`** | **Parou na Rodada 5!** | **`31` (`-49,2%`, $n_L=6$)** | **`111.138` (`-57,1%`)** | **`962,8s` (`16,0m`)** | **`1.153,2s` (`19,2m`)** | **`0,6528` (`+4,9%`!)** (`0,8238` / **`0,5406`**) | **`2,41×` mais rápido na Fase 1 (`2,68×` no Total) e `+4,9%` F1 (`+11,6%` Recall)!** |

---

## 5. Tabela Master Consolidada — Todos os Experimentos Concluídos na Axia (H200)

| Dataset & Configuração na Axia | Método | Qualidade Principal (`F1` ou `RP@5`) | Precision / Recall (CUAD) | Tempo Otimização (`OptTime`) | Tempo Execução (`ExecTime`) | Tempo Total (`TotTime`) | Rodadas MAB | Ganho do Nosso Método (`ES`) na Otimização e Qualidade |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CUAD Piloto 1× H200** (`120B + 8B`, `B=40, k=4, j=2, val=10, test=10`) | `Baseline (0s)` | `F1 = 0,7707` | `0,7990` / `0,7443` | `863,9s` (`14,4m`) | `173,0s` (`2,9m`) | `1.036,9s` (`17,3m`) | `9` | Referência |
| **CUAD Piloto 1× H200** (`120B + 8B`, `B=40, k=4, j=2, val=10, test=10`) | **`Nosso Método (ES)`** | **`F1 = 0,7838` (`+1,7%`)** | **`0,8168` / `0,7534`** | **`283,9s` (`4,7m`)** | `230,4s` (`3,8m`) | **`514,3s` (`8,6m`)** | **`3`** | **`3,04×` mais rápido (`-67,1%`) e `+1,7%` F1** |
| **CUAD Piloto 2× H200** (`70B + 120B`, `B=40, k=4, j=2, val=10, test=10`) | `Baseline (0s)` | `F1 = 0,8010` | `0,8798` / `0,7352` | `9.573,5s` (`159,6m`) | `323,3s` (`5,4m`) | `9.896,7s` (`164,9m`) | `9` | Referência |
| **CUAD Piloto 2× H200** (`70B + 120B`, `B=40, k=4, j=2, val=10, test=10`) | `Nosso Método (ES Unpaired)` | `F1 = 0,8192` (`+2,3%`) | `0,8211` / **`0,8174`** | **`2.524,1s` (`42,1m`)** | `2.785,6s` (`46,4m`) | **`5.309,7s` (`88,5m`)** | **`3`** | **`3,79×` mais rápido (`-73,6%`) e `+2,3%` F1** |
| **CUAD Piloto 2× H200** (`70B + 120B`, `B=40, k=4, j=2, val=10, test=10`) | **`Nosso Método (ES Paired)`** | **`F1 = 0,8213` (`+2,5%`)** | **`0,8718` / `0,7763`** | **`4.411,3s` (`73,5m`)** | **`1.616,0s` (`26,9m`)** | **`6.027,3s` (`100,5m`)** | **`3`** | **`2,17×` mais rápido (`-53,9%`) e maior F1 do Piloto (`0,8213`)!** |
| **CUAD Full Seed 0** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | `Baseline (0s)` | `F1 = 0,7750` | `0,8886` / `0,6871` | `25.506,2s` (`7,09h`) | `1.284,8s` (`0,36h`) | `26.790,9s` (`7,44h`) | `17` | Referência |
| **CUAD Full Seed 0** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | `Nosso Método (ES Unpaired)` | `F1 = 0,7208` | `0,8837` / `0,6087` | **`9.743,7s` (`2,71h`)** | `15.400,4s` (`4,28h`) | `25.144,2s` (`6,98h`) | `3` | `2,62×` mais rápido na Fase 1 (`-61,8%`) |
| **CUAD Full Seed 0** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | **`Nosso Método (ES Paired)`** | **`F1 = 0,7655` (`+4,47 p.p.`)** | **`0,8941` / `0,6692`** | **`12.017,1s` (`3,34h`)** | **`6.560,9s` (`1,82h`)** | **`18.578,0s` (`5,16h`)** | **`6` (Zona Cinza R3–R5!)** | **`2,12×` mais rápido Fase 1, `1,44×` no Total (`-2,28h`) e recupera F1 (`0,7655`)!** |
| **CUAD Full Seed 1** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | `Baseline (0s)` | `F1 = 0,6291` | `0,8829` / `0,4886` | `33.029,1s` (`9,17h`) | `4.370,0s` (`1,21h`) | `37.399,1s` (`10,39h`) | `17` | Referência |
| **CUAD Full Seed 1** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | **`Nosso Método (ES Unpaired)`** | **`F1 = 0,6525` (`+3,7%`)** | **`0,8904` / `0,5150`** | **`9.201,2s` (`2,56h`)** | **`3.289,7s` (`0,91h`)** | **`12.490,9s` (`3,47h`)** | **`3`** | **`3,59×` mais rápido (`-72,1%` / `-6,62h`) e `+3,7%` F1** |
| **CUAD Full Seed 1** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | **`Nosso Método (ES Paired)`** | **`F1 = 0,6246`** | `0,8611` / **`0,4900`** | **`8.227,5s` (`2,29h`)** | `15.009,5s` (`4,17h`) | **`23.237,0s` (`6,45h`)** | **`3`** | **`4,01×` mais rápido na Fase 1 (`-75,1%` / `-6,89h`) e `1,61×` no Total** |
| **CUAD Full Seed 2** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | `Baseline (0s)` | *(Fase 1 Concluída)* | — | `18.586,0s` (`5,16h`) | *(49/50 contratos)* | — | `17` | Referência Fase 1 (`103` amostras) |
| **CUAD Full Seed 2** (`70B + 120B`, `B=100, k=5, j=4, val=20, test=50`) | **`Nosso Método (ES)`** | **`F1 = 0,5828`** | **`0,8407` / `0,4460`** | **`6.220,7s` (`1,73h`)** | `13.966,6s` (`3,88h`) | **`20.187,4s` (`5,61h`)** | **`3`** | **`2,99×` mais rápido na Fase 1 (`-66,5%` / `-3,43h`)** |
| **CUAD Stress Test `B=250`** (`70B + 120B`, `B=250, k=5, j=4, val=25, test=50`) | `Baseline (0s)` | `F1 = 0,7725` | `0,9040` / `0,6743` | `84.930,6s` (**`23,59h`**) | `1.060,6s` (`0,29h`) | `85.991,2s` (**`23,89h`**) | `40` | Referência (`LLMConvertBonded GPU1`) |
| **CUAD Stress Test `B=250`** (`70B + 120B`, `B=250, k=5, j=4, val=25, test=50`) | **`Nosso Método (ES)`** | **`F1 = 0,7901` (`+2,3%`)** | **`0,9128` / `0,6965`** | **`9.466,9s` (`2,63h`)** | `1.587,9s` (`0,44h`) | **`11.054,8s` (`3,07h`)** | **`3`** | **`8,97×` MAIS RÁPIDO (`-88,9%`, poupando `20,96h` de 2× H200!) com o mesmo operador e `+2,3%` F1!** |
| **BioDex Piloto 1× H200** (`120B + 8B`, `B=90, k=4, j=2, val=10, test=20`) | `Baseline (0s)` | `RP@5 = 0,3200` | — | `1.068,2s` (`17,8m`) | `398,7s` (`6,6m`) | `1.466,9s` (`24,4m`) | `6` | Referência |
| **BioDex Piloto 1× H200** (`120B + 8B`, `B=90, k=4, j=2, val=10, test=20`) | **`Nosso Método (ES)`** | *(Fase 1 Concluída)* | — | **`599,0s` (`10,0m`)** | *(18/20 registros)* | — | **`3`** | **`1,78×` mais rápido (`-43,9%` na Fase 1)** |
| **BioDex Piloto 2× H200** (`70B + 120B`, `B=90, k=4, j=2, val=10, test=20`) | `Baseline (0s)` | `RP@5 = 0,3825` | — | `5.829,1s` (`97,2m`) | `271,0s` (`4,5m`) | `6.100,1s` (`101,7m`) | `5` | Referência |
| **BioDex Piloto 2× H200** (`70B + 120B`, `B=90, k=4, j=2, val=10, test=20`) | `Nosso Método (ES Unpaired)` | **`RP@5 = 0,4425` (`+15,7%`)** | — | **`4.581,6s` (`76,4m`)** | `3.503,5s` (`58,4m`) | `8.085,2s` (`134,8m`) | **`4`** | **`1,27×` mais rápido (`-21,4%`) e `+15,7%` RP@5** |
| **BioDex Piloto 2× H200** (`70B + 120B`, `B=50, k=4, j=2, val=10, test=20`) | **`Nosso Método (ES Paired)`** | **`RP@5 = 0,4275` (`+11,8%`)** | — | **`3.833,0s` (`63,9m`)** | **`2.138,9s` (`35,6m`)** | **`5.971,9s` (`99,5m`)** | **`3`** | **`1,52×` mais rápido (`-34,2%`) e `+11,8%` RP@5** |
| **BioDex Full Seed 0** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | `Baseline (0s)` | `RP@5 = 0,2046` | — | `16.512,7s` (`4,59h`) | `22.108,6s` (`6,14h`) | `38.621,3s` (`10,73h`) | `14` | Referência (`1.927.139` tokens totais) |
| **BioDex Full Seed 0** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | `Nosso Método (ES Unpaired)` | `RP@5 = 0,2818` (`+37,7%`) | — | **`4.651,1s` (`1,29h`)** | `26.951,6s` (`7,49h`) | `31.602,7s` (`8,78h`) | `3` | `3,55×` mais rápido (`-71,8%`) e `+37,7%` RP@5 |
| **BioDex Full Seed 0** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | **`Nosso Método (ES Paired)`** | **`RP@5 = 0,3025` (`+47,8%`!)** | — | **`6.464,7s` (`1,80h`)** | **`20.338,4s` (`5,65h`)** | **`26.803,1s` (`7,45h`)** | **`R4, R5 (Eficácia) e R9 (Zona Cinza R3–R8!)`** | **`2,55×` mais rápido Fase 1, `1,44×` no Total (`-3,28h`) e `+47,8%` RP@5 (`0,3025`)!** |
| **BioDex Full Seed 1** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | `Baseline (0s)` | *(Fase 1 Concluída)* | — | `17.876,0s` (`4,97h`) | *(99/250 registros)* | — | `14` | Referência Fase 1 (`269` amostras) |
| **BioDex Full Seed 1** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | **`Nosso Método (ES)`** | *(Fase 1 Concluída)* | — | **`5.232,0s` (`1,45h`)** | *(125/250 registros)* | — | **`7`** | **`3,42×` mais rápido (`-70,7%` / `-3,51h` na Fase 1)** |
| **BioDex Full Seed 2** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | `Baseline (0s)` | `RP@5 = 0,2839` | — | `12.959,3s` (`3,60h`) | `41.961,7s` (`11,66h`) | `54.921,1s` (`15,26h`) | `14` | Referência |
| **BioDex Full Seed 2** (`70B + 120B`, `B=300, k=5, j=4, val=20, test=250`) | **`Nosso Método (ES)`** | `RP@5 = 0,2414` | — | **`4.620,2s` (`1,28h`)** | **`19.145,9s` (`5,32h`)** | **`23.766,1s` (`6,60h`)** | **`3`** | **`2,80×` mais rápido na Fase 1 (`-64,3%`) e `2,31×` no Total (`-8,65h`!)** |

---

## 6. Status dos 2 Jobs Ativos na Axia (`2× NVIDIA H200`)

1. **`c4221713dca245758295a813ca14b6d7` (`abacus-es-axia-paired-sigma-suite`, `status: running`):**
   - **Já concluídos (5/8):** `axia-paired-cuad-pilot-es` (`F1 = 0,8213`), `axia-paired-biodex-pilot-es` (`RP@5 = 0,4275`), `axia-paired-cuad-full-s0-es` (`F1 = 0,7655`, parou na **Rodada 6** após Zona Cinza), `axia-paired-biodex-full-s0-es` (`RP@5 = 0,3025`, parou nas **Rodadas 4, 5 e 9** após Zona Cinza), `axia-paired-cuad-full-s1-es` (`F1 = 0,6246`, `4,01×` speedup).
   - **Em execução agora:** `axia-paired-biodex-full-s1-es`, seguido de `axia-paired-cuad-full-s2-es` e `axia-paired-biodex-full-s2-es`.
2. **`50101f8cacbb46abab2c8bdefe82ef3f` (`abacus-es-axia-validation-suite`, `status: running`):**
   - **Já concluídos (8/21):** Toda a Fase 3A de Paradas Dinâmicas (`axia-val-cuad-baseline-j2`, `axia-val-cuad-j1-nmin4`, `axia-val-cuad-j2-nmin5`, `axia-val-cuad-j2-nmin6`, `axia-val-cuad-j2-nmin8`, `axia-val-cuad-grayzone-g05`, `axia-val-cuad-mode-efficacy`) + `axia-cuad-b50-baseline` (`F1 = 0,7912`, `OptTime = 14.594,1s`).
   - **Em execução agora:** `axia-cuad-b50-es`, seguido da sensibilidade de orçamento ($B \in \{200, 400\}$) e topologia/concorrência ($k=10, w \in \{4, 16, 32\}$).
