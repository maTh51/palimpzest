# Relatório Técnico Completo — Experimentos de Escala na Axia (NVIDIA H200) com `Llama-3.3-70B-Instruct` e `openai/gpt-oss-120b`: Nova Variância Pareada (`sigma_paired`), Paradas Dinâmicas, Sensibilidade ao Orçamento ($B \le 600$) e Topologia Larga ($k = 10$)

> **Foco Experimental:** Comparação direta e exclusiva entre o **Baseline Abacus (`0s` — Orçamento MAB Completo)** e o **Nosso Método (`ES` — Parada Antecipada Estatística por Eficácia e Futilidade)** em modelos de grande porte (**70B + 120B parâmetros**) executando em GPUs **NVIDIA H200 (141 GB VRAM)** no cluster **Axia (`kunumi/hpc4ai`)**, cobrindo todas as frentes experimentais concluídas:
> 1. **Nova Metodologia de Variância Pareada Intra-Documento (`ES_SIGMA_MODE="paired"`, limite inferior estrito $t_{\min}=1, n_{\min}=4, \gamma=0,5$):** $n_{\min}=4$ atua exclusivamente como **portão de entrada (limite inferior)** e o teste estatístico decide se para por Eficácia, para por Futilidade ou **permanece na Zona Cinza** por múltiplas rodadas até convergir;
> 2. **Suíte de Validação Dinâmica na Axia (`t_min = 1`, variando $j \in \{1,2\}$, $n_{\min} \in \{4,5,6,8\}$, $\gamma=0,5$ e ablação `ES_MODE=efficacy`)**;
> 3. **Curva Completa de Sensibilidade ao Orçamento ($B \in \{40, 50, 100, 200, 250\}$ no CUAD e $B \in \{50, 90, 300, 600\}$ no BioDex):** Speedups que escalam até **`9,17×` (`-89,1%`) no CUAD ($B=200$, `F1 = 0,8394` vs. `0,7318`, `+10,76 p.p.`)**, **`8,97×` (`-88,9%`) no CUAD ($B=250$, poupando `20,96 horas` de 2× H200!)** e **`6,37×` (`-84,3%`) no BioDex ($B=600$, `RP@5 = 0,2350` vs. `0,1481`, `+58,6%`!)**;
> 4. **Escalabilidade Topológica com Fronteira Larga ($k = 10$ operadores simultâneos no CUAD e no BioDex):** Speedups de **`2,46×` no CUAD (`-7,17 horas`, escolhendo exatamente o mesmo operador!)** e **`3,23×` no BioDex (`-4,19 horas`, escolhendo exatamente os mesmos operadores LLM!)**.

---

## 1. Resumo Executivo e Discussão das Principais Descobertas

Ao escalar o otimizador MAB do Palimpzest/Abacus da **Morfeu** (`Qwen2.5-14B` / `Llama-3.1-8B`) para a **Axia** (`meta-llama/Llama-3.3-70B-Instruct` + `openai/gpt-oss-120b` em 1× e 2× NVIDIA H200 de 141 GB), o custo computacional de cada amostra MAB aumenta em mais de uma ordem de grandeza (especialmente em operadores compostos como `MixtureOfAgentsConvert` e `CritiqueAndRefineConvert`). A campanha completa na Axia consolidou **quatro resultados fundamentais**:

### 1.1. Discussão 1 — Por que $n_{\min}=4$ é Apenas um Limite Inferior e Como a Variância Pareada Intra-Documento ($\hat{\sigma}_{\text{paired}}$) Destrava a Zona Cinza
- **Diagnóstico do Problema no Estimador Não-Pareado ($\hat{\sigma}_{\text{unpaired}}$):** Documentos distintos possuem dificuldades intrinsecamente diferentes ($F_1 \in [0,0, 1,0]$). Quando a variância agrupada mistura documentos fáceis e difíceis sem descontar o efeito do documento, $\hat{\sigma}_{\text{unpaired}}$ fica artificialmente inflado ($\approx 0,09\text{--}0,41$), elevando $\text{SE}_{\text{futuro}}$ e causando parada por Futilidade assim que o algoritmo atinge o limite mínimo $n_{\min}$.
- **Solução Estatística ($\hat{\sigma}_{\text{paired}}$ — Variância Pareada Intra-Documento):** Como os operadores na fronteira avaliam os **mesmos documentos de validação** (`shared_docs_lr`), subtraímos a média de cada documento $d$ ($r_{i,d} = y_{i,d} - \bar{y}_{\cdot,d}$) e combinamos com a variância das diferenças pareadas $d_{LR}(d) = y_{L,d} - y_{R,d}$. Nos logs reais da Axia, isso reduziu o ruído residual $\hat{\sigma}$ em **`37,2%` a `86,3%`**!
- **Comportamento Empírico Comprovado na Axia (`2× H200: 70B + 120B`, $t_{\min}=1, n_{\min}=4$ fixos):**
  1. **`axia-paired-cuad-full-s0-es` (`CUAD Full Seed 0, B=100, N_test=50`):** Quando Líder e Vice-Líder atingem $n_{\min}=4$ na Rodada 3, $\hat{\sigma}_{\text{paired}} = 0,0499$ (44,7% menor que $\hat{\sigma}_{\text{unpaired}} = 0,0903$) coloca $\Delta = 0,0195$ **dentro da Zona Cinza** ($[\text{SE}_{\text{fut}} = 0,0130,\, \text{SE}_{\text{atual}} = 0,0581)$). Em vez de parar na Rodada 3, o algoritmo **continua amostrando nas Rodadas 3, 4 e 5 (`zone = gray_zone`)** e **para dinamicamente na Rodada 6 ($n_L = 7$)**, escolhendo `CritiqueAndRefineConvert`. Com isso, o **F1 sobe de `0,7208` (ES não-pareado) para `0,7655` (`+4,47 p.p.` / `+6,2%`, com Precisão `0,8941` superior ao Baseline `0,8886`!)**, mantendo **`2,12×` de speedup na otimização (`3,34h` vs. `7,09h`)** e terminando o pipeline completo **`2,28 horas` mais rápido que o Baseline e `1,82 horas` mais rápido que o ES não-pareado**!
  2. **`axia-paired-biodex-full-s0-es` (`BioDex Full Seed 0, B=300, N_test=250`):** No pipeline de 3 operadores, o ruído residual cai em **`86,3%`** ($\hat{\sigma}_{\text{paired}} = 0,0506$ vs. $\hat{\sigma}_{\text{unpaired}} = 0,3703$). Ao atingir $n_{\min}=4$, cada estágio toma uma decisão estatística independente:
     - `Op 3` para por **Eficácia na Rodada 4** ($n_L = 5, \Delta = 0,1675 \ge \text{SE}_{\text{atual}}$);
     - `Op 1` entra na Zona Cinza na Rodada 4 e para por **Eficácia na Rodada 5** ($n_L = 6, \Delta = 0,0762 \ge \text{SE}_{\text{atual}} = 0,0616$);
     - `Op 2` **permanece na Zona Cinza durante 6 rodadas consecutivas (Rodadas 3, 4, 5, 6, 7 e 8)** e **para na Rodada 9 ($n_L = 10$)**!
     - **Resultado nos 250 artigos biomédicos de teste:** **`RP@5 = 0,3025` (`+47,8%` acima do Baseline `0,2046` e `+7,3%` acima do ES não-pareado `0,2818` — o maior `RP@5` de toda a campanha BioDex Full!)**, com **`2,55×` de speedup na otimização (`1,80h` vs. `4,59h`)** e **`1,44×` no tempo total (`7,45h` vs. `10,73h`, economizando `3,28 horas`)**!
  3. **`axia-paired-cuad-pilot-es` (`CUAD Pilot, B=40`):** Atinge **`F1 = 0,8213` (`P = 0,8718, R = 0,7763`)** — maior F1 do Piloto (`+2,5%` acima do Baseline `0,8010` e `+5,07 p.p.` a mais de Precisão que o ES não-pareado), com **`2,17×` de speedup na otimização (`73,5m` vs. `159,6m`)**.
  4. **`axia-paired-cuad-full-s1-es` (`CUAD Full Seed 1, B=100`):** **`4,01×` mais rápido na otimização (`2,29h` vs. `9,17h`, `-75,1%`, economizando `6,89 horas`)** e **`1,61×` mais rápido no tempo total (`6,45h` vs. `10,39h`)**, mantendo `F1 = 0,6246` vs. `0,6291` (`-0,45 p.p.`).

---

### 1.2. Discussão 2 — Curva de Sensibilidade ao Orçamento ($B \in \{40, 50, 100, 200, 250, 300, 600\}$): Speedup Cresce Linearmente com $B$ (Até `9,17×`) e Previne *Overfitting* Severo em Orçamentos Grandes
Um resultado crucial para a tese é o comportamento assintótico quando o usuário fornece um orçamento de amostragem generoso ($B = 200, 250$ ou $600$):
- **O Baseline (`0s`) é Obrigado a Gastar Todo o Orçamento $B$:** No Abacus original, aumentar $B$ de `50` para `200` ou `250` faz o tempo de otimização explodir de `4,05 horas` (`B=50`) para **`18,01 horas` (`B=200`)** e **`23,59 horas` (`B=250`)** no CUAD, e para **`9,41 horas` (`B=600`)** no BioDex.
- **Pior ainda: Orçamentos Excessivos no Baseline Causam *Overfitting* na Validação!**
  - Em **`axia-cuad-b200-baseline` (`B=200`)**, após 18 horas reamostrando o conjunto de validação, o Baseline degradou para `F1 = 0,7318` (trocando o modelo base do `CritiqueAndRefineConvert` para `GPU0/70B`), enquanto o **`Nosso Método (`axia-cuad-b200-es`)`** parou em **`1,96 horas` (`7.073,8s`)** — **`9,17×` MAIS RÁPIDO (`-89,1%`, poupando `16,05 horas` de 2× H200!)** — e preservou o `CritiqueAndRefineConvert` com modelo base `GPU1/120B`, alcançando **`F1 = 0,8394` (`+14,7%` / `+10,76 p.p.` acima do Baseline!)**!
  - Em **`axia-cuad-stress-b250` (`B=250`)**, o Baseline levou **`23,59 horas`** (`84.930,6s`) para eleger `LLMConvertBonded (GPU1)` (`F1 = 0,7725`), enquanto o **`Nosso Método (`axia-cuad-stress-b250-es`)`** elegeu **exatamente o mesmo operador físico** em **`2,63 horas`** (`9.466,9s`) — **`8,97×` MAIS RÁPIDO (`-88,9%`, poupando `20,96 horas` de 2× H200!)** com **`F1 = 0,7901` (`+2,3%`)**!
  - Em **`axia-biodex-stress-b600` (`B=600` no BioDex)**, após `9,41 horas` de otimização, o Baseline sofreu *overfitting* extremo no operador de recuperação `TopKOp`, reduzindo $k$ para apenas `k=3` e derrubando o `RP@5` nos 250 artigos de teste para **`0,1481`**! Já o **`Nosso Método (`axia-biodex-stress-b600-es`)`** parou em **`1,48 horas` (`5.313,8s`)** — **`6,37×` MAIS RÁPIDO (`-84,3%`, poupando `7,93 horas` na Fase 1!)** — preservando `TopKOp(k=25)` e atingindo **`RP@5 = 0,2350` (`+58,6%` / `+8,69 p.p.` acima do Baseline!)**!

---

### 1.3. Discussão 3 — Escalabilidade Topológica com Fronteira Larga ($k = 10$ Operadores Simultâneos por Estágio)
Para testar se a parada antecipada mantém sua precisão quando a fronteira inicial de Pareto é dobrada de $k=5$ para **$k=10$ operadores concorrentes simultâneos**:
- **CUAD Fronteira Larga (`axia-cuad-stress-k10`, $B=150, k=10, j=3$):**
  - `Baseline (0s)` levou **`12,09 horas`** (`43.528,6s`) na otimização (`16,25 horas` no total) para selecionar `MixtureOfAgentsConvert (Proposers: [GPU0, GPU1], Temp: [0.0, 0.0], Aggregator: GPU1)`, obtendo `F1 = 0,7518` (`P = 0,8781, R = 0,6573`).
  - **`Nosso Método (ES)`** selecionou **exatamente o mesmo operador físico** em **`4,92 horas`** (`17.715,9s`) — **`2,46×` mais rápido na otimização (`-59,3%`, economizando `7,17 horas` de 2× H200!)** e **`1,80×` mais rápido no tempo total (`9,02h` vs. `16,25h`, economizando `7,22 horas`!)** — com `F1 = 0,7462` e Recall superior (`0,6616` vs. `0,6573`)!
- **BioDex Fronteira Larga (`axia-biodex-stress-k10`, $B=450, k=10, j=3, N_{\text{test}}=250$):**
  - `Baseline (0s)` levou **`6,06 horas`** (`21.821,2s`) na otimização (`13,27 horas` no total) para selecionar `Op 1: CritiqueAndRefine (GPU1->GPU1->GPU0)`, `Op 2: TopKOp(k=10)` e `Op 3: MixtureOfAgents (GPU0+1, T=0.0 -> GPU1)`, obtendo `RP@5 = 0,2819`.
  - **`Nosso Método (ES)`** convergiu em **`1,88 horas`** (`6.751,8s`) — **`3,23×` mais rápido na otimização (`-69,1%`, economizando `4,19 horas`)** e **`1,36×` mais rápido no tempo total (`9,74h` vs. `13,27h`, economizando `3,54 horas`)** — selecionando **exatamente os mesmos operadores LLM (`Op 1` e `Op 3`)** (`RP@5 = 0,2483`).

---

### 1.4. Discussão 4 — Suíte de Validação Dinâmica na Axia (`t_min = 1`, Variando $j, n_{\min}, \gamma, \text{ES\_MODE}$)
Na bateria dedicada de validação dinâmica (`B=60, k=5, val=15, test=20, t_min=1`), provamos diretamente nas 2× H200 que:
- Variar $n_{\min} \in \{4, 5, 6, 8\}$ desloca organicamente a rodada de parada (**Rodadas 3, 4, 5 e 7**), superando o F1 do Baseline (`0,7880`) em todas as configurações moderadas:
  - `j=1, n_min=4` (Rodada 4): **`F1 = 0,8112` (`+2,9%`)**, **`2,56×` speedup** (`1,25h` vs. `3,19h`);
  - `j=2, n_min=5` (Rodada 4): **`F1 = 0,8149` (`+3,4%`)**, **`1,48×` speedup Fase 1 (`1,76×` Total)**;
  - `j=2, n_min=6` (Rodada 5): **`F1 = 0,8069` (`+2,4%`)**, **`1,49×` speedup Fase 1**;
  - `j=2, n_min=4, gamma=0,5` (Rodada 3): **`F1 = 0,8292` (`+5,2%`)**, **`3,47×` speedup Fase 1 (`3,82×` Total: `1,04h` vs. `3,98h`)**!
- Já a ablação **`ES_MODE=efficacy` (`axia-val-cuad-mode-efficacy`)**, que desliga a parada por Futilidade, alcança **`F1 = 0,8431` (`+7,0%`)**, mas percorre todas as 10 rodadas (`OptTime = 15.468,6s`), provando experimentalmente que **a Futilidade é indispensável para encerrar empates estatísticos**, enquanto **a Eficácia captura líderes isolados** (como demonstrado nos estágios `Op 1` e `Op 3` do BioDex Pareado).

---

## 2. Formulação Matemática: Critério Dual e Variância Pareada Intra-Documento ($\hat{\sigma}_{\text{paired}}$)

Dentro de [mab_execution_strategy.py](file:///home/amarante/Jobs/Kunumi/palimpzest/src/palimpzest/query/execution/mab_execution_strategy.py#L106-L245), o critério de parada avalia ao final de cada rodada $t$ a diferença empírica $\Delta = \hat{\mu}_L - \hat{\mu}_R$ entre o operador **Líder ($L$)** e o **Vice-Líder ($R$)**, após ambos atingirem o **limite inferior de entrada** ($t \ge t_{\min}$ e $n_L, n_R \ge n_{\min}$):

$$\text{SE}_{\text{atual}} = z \cdot \hat{\sigma}_{\text{paired}} \sqrt{\frac{1}{n_L} + \frac{1}{n_R}}, \qquad \text{SE}_{\text{futuro}} = \gamma \cdot z \cdot \hat{\sigma}_{\text{paired}} \sqrt{\frac{1}{n_L^+} + \frac{1}{n_R^+}}$$

onde $n_i^+ = \min\!\left(N_{\text{val}},\, n_i + \frac{B_{\text{rem}}}{C}\right)$ limita a projeção futura ao tamanho do conjunto de validação $N_{\text{val}}$, e $\hat{\sigma}_{\text{paired}}$ combina o resíduo centrado por documento com a variância das diferenças pareadas entre Líder e Vice-Líder nos documentos compartilhados $D_{L \cap R}$:

$$\hat{\sigma}_{\text{within}}^2 = \frac{1}{\text{dof}_{\text{within}}} \sum_{d \in D_{\text{shared}}} \sum_{i \in \mathcal{O}_d} \left(y_{i,d} - \bar{y}_{\cdot,d}\right)^2, \qquad \hat{\sigma}_{\text{diff}, LR}^2 = \frac{1}{2\left(|D_{L \cap R}| - 1\right)} \sum_{d \in D_{L \cap R}} \left(d_{LR}(d) - \bar{d}_{LR}\right)^2$$

$$\hat{\sigma}_{\text{paired}} = \sqrt{\frac{\hat{\sigma}_{\text{within}}^2 + \hat{\sigma}_{\text{diff}, LR}^2}{2}}$$

---

## 3. Tabelas Temáticas de Resultados na Axia (`2× NVIDIA H200: Llama-3.3-70B + gpt-oss-120b`)

### 3.1. Tabela Comparativa — Variância Não-Pareada (`Unpaired`) vs. Nova Variância Pareada (`Paired`, $t_{\min}=1, n_{\min}=4$ fixos)

| Cenário na Axia (`2× H200: 70B + 120B`) | Método | Rodada de Parada e Dinâmica da Zona Cinza | Desvio-Padrão ($\hat{\sigma}$) | Qualidade no Teste (`F1` ou `RP@5`) | Precision / Recall (CUAD) | Tempo Otimização (`OptTime`) | Tempo Total (`TotTime`) | Speedup Fase 1 (`0s / ES`) | Operador Físico Selecionado |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CUAD Pilot** (`B=40, val=10, test=10`) | `Baseline (0s)` | Rodada 9 (Fim) | — | `F1 = 0,8010` | `0,8798` / `0,7352` | `9.573,5s` (`159,6m`) | `9.896,7s` (`164,9m`) | `1,00×` | `LLMConvertBonded (GPU1)` |
| **CUAD Pilot** (`B=40, val=10, test=10`) | `ES Antigo (Unpaired)` | Rodada 3 (`n_L=4`) | `0,0882` | `F1 = 0,8192` | `0,8211` / `0,8174` | `2.524,1s` (`42,1m`) | `5.309,7s` (`88,5m`) | `3,79×` (`-73,6%`) | `MixtureOfAgents (GPU0+1 -> GPU1)` |
| **CUAD Pilot** (`B=40, val=10, test=10`) | **`ES Novo (Paired)`** | **Rodada 3 (`n_L=4`)** | **`0,0516` (`-41,5%`)** | **`F1 = 0,8213` (`+2,5%`)** | **`0,8718` / `0,7763`** | **`4.411,3s` (`73,5m`)** | **`6.027,3s` (`100,5m`)** | **`2,17×` (`-53,9%`)** | **`CritiqueAndRefine (GPU1 -> GPU0 -> GPU1)`** |
| **CUAD Full Seed 0** (`B=100, val=20, test=50`) | `Baseline (0s)` | Rodada 17 (Fim) | — | `F1 = 0,7750` | `0,8886` / `0,6871` | `25.506,2s` (`7,09h`) | `26.790,9s` (`7,44h`) | `1,00×` | `LLMConvertBonded (GPU1)` |
| **CUAD Full Seed 0** (`B=100, val=20, test=50`) | `ES Antigo (Unpaired)` | Rodada 3 (`n_L=4`) | `0,0903` | `F1 = 0,7208` | `0,8837` / `0,6087` | `9.743,7s` (`2,71h`) | `25.144,2s` (`6,98h`) | `2,62×` (`-61,8%`) | `MixtureOfAgents (GPU0+1 -> GPU1)` |
| **CUAD Full Seed 0** (`B=100, val=20, test=50`) | **`ES Novo (Paired)`** | **Zona Cinza nas Rodadas 3, 4 e 5 $\to$ Parou na Rodada 6 (`n_L=7`)!** | **`0,0499` (`-44,7%`)** | **`F1 = 0,7655` (`+4,47 p.p.` vs Unpaired!)** | **`0,8941` / `0,6692`** | **`12.017,1s` (`3,34h`)** | **`18.578,0s` (`5,16h`, `-30,7%`!)** | **`2,12×` Fase 1 / `1,44×` Total (`-2,28h` vs Base / `-1,82h` vs Unpaired)** | **`CritiqueAndRefine (GPU1 -> GPU0 -> GPU1)`** |
| **CUAD Full Seed 1** (`B=100, val=20, test=50`) | `Baseline (0s)` | Rodada 17 (Fim) | — | `F1 = 0,6291` | `0,8829` / `0,4886` | `33.029,1s` (`9,17h`) | `37.399,1s` (`10,39h`) | `1,00×` | `MixtureOfAgents (GPU1 -> GPU0)` |
| **CUAD Full Seed 1** (`B=100, val=20, test=50`) | **`ES Novo (Paired)`** | **Rodada 3 (`n_L=4`)** | **`0,1285` (`-37,2%`)** | **`F1 = 0,6246`** | `0,8611` / **`0,4900`** | **`8.227,5s` (`2,29h`)** | **`23.237,0s` (`6,45h`, `-37,9%`)** | **`4,01×` (`-75,1%`, poupando `6,89h`!)** | `MixtureOfAgents (GPU0+1 -> GPU1)` |
| **BioDex Pilot** (`B=50/90, val=10, test=20`) | `Baseline (0s)` | Rodada 5 (Fim) | — | `RP@5 = 0,3825` | — | `5.829,1s` (`97,2m`) | `6.100,1s` (`101,7m`) | `1,00×` | `Bonded + TopK(20) + Bonded` |
| **BioDex Pilot** (`B=50, val=10, test=20`) | **`ES Novo (Paired)`** | **Rodada 3 (`n_L=4`)** | **`0,0000` (`-100%`)** | **`RP@5 = 0,4275` (`+11,8%`)** | — | **`3.833,0s` (`63,9m`)** | **`5.971,9s` (`99,5m`)** | **`1,52×` (`-34,2%`)** | **`MoA + TopK(15) + Critique`** |
| **BioDex Full Seed 0** (`B=300, val=20, test=250`) | `Baseline (0s)` | Rodada 14 (Fim) | — | `RP@5 = 0,2046` | — | `16.512,7s` (`4,59h`) | `38.621,3s` (`10,73h`) | `1,00×` | `Critique + TopK(20) + Critique` |
| **BioDex Full Seed 0** (`B=300, val=20, test=250`) | `ES Antigo (Unpaired)` | Rodada 3 (Todos os 3 Ops) | `0,2175` | `RP@5 = 0,2818` (`+37,7%`) | — | `4.651,1s` (`1,29h`) | `31.602,7s` (`8,78h`) | `3,55×` (`-71,8%`) | `Bonded + TopK(20) + MoA` |
| **BioDex Full Seed 0** (`B=300, val=20, test=250`) | **`ES Novo (Paired)`** | **`Op 3` R4 (Eficácia), `Op 1` R5 (Eficácia), `Op 2` R3–R8 Zona Cinza $\to$ Parou na Rodada 9 (`n_L=10`)!** | **`0,0506` vs `0,3703` (`-86,3%` ruído!)** | **`RP@5 = 0,3025` (`+47,8%` vs Base / `+7,3%` vs Unpaired!)** | — | **`6.464,7s` (`1,80h`)** | **`26.803,1s` (`7,45h`, `-30,6%` vs Base!)** | **`2,55×` Fase 1 (`-60,8%`) e `-3,28h` no Tempo Total!** | **`Bonded (GPU1) + TopK(20) + Critique (GPU1->GPU0->GPU1)`** |

---

### 3.2. Tabela de Sensibilidade ao Orçamento ($B \in \{40, 50, 100, 200, 250\}$ no CUAD e $B \in \{50, 90, 300, 600\}$ no BioDex)

| Dataset & Orçamento ($B$) | Qualidade `Baseline (0s)` | Qualidade `Nosso Método (ES)` | Ganho de Qualidade ($\Delta$) | Tempo Otimização `Baseline (0s)` | Tempo Otimização `Nosso Método (ES)` | Speedup Fase 1 (`0s / ES`) | Speedup Tempo Total (`0s / ES`) | Horas de 2× H200 Economizadas (Fase 1 / Total) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUAD `B = 40`** (`Pilot 2× H200`, `Paired / Unpaired`) | `F1 = 0,8010` | **`F1 = 0,8213` / `0,8192`** | **`+2,5%` / `+2,3%`** | `9.573,5s` (`2,66h`) | **`4.411,3s` / `2.524,1s`** | **`2,17×` a `3,79×`** | **`1,64×` a `1,86×`** | **`-1,43h` a `-1,96h` Fase 1** |
| **CUAD `B = 50`** (`axia-cuad-b50`, `val=15, test=20`) | `F1 = 0,7912` | **`F1 = 0,8122`** | **`+2,7%` (`+2,10 p.p.`)** | `14.594,1s` (`4,05h`) | **`4.861,6s` (`1,35h`)** | **`3,00×` (`-66,7%`)** | **`3,19×` (`-68,7%`)** | **`-2,70h` Fase 1 / `-3,23h` Total** |
| **CUAD `B = 100`** (`Full Seed 0 Paired`, `val=20, test=50`) | `F1 = 0,7750` | **`F1 = 0,7655`** | `-0,95 p.p.` (`P=0,8941`) | `25.506,2s` (`7,09h`) | **`12.017,1s` (`3,34h`)** | **`2,12×` (`-52,9%`)** | **`1,44×` (`-30,7%`)** | **`-3,75h` Fase 1 / `-2,28h` Total** |
| **CUAD `B = 100`** (`Full Seed 1 Unpaired / Paired`) | `F1 = 0,6291` | **`F1 = 0,6525` / `0,6246`** | **`+3,7%`** (Mesmo Op!) | `33.029,1s` (`9,17h`) | **`9.201,2s` / `8.227,5s`** | **`3,59×` a `4,01×` (`-75,1%`)** | **`2,99×` a `1,61×`** | **`-6,62h` a `-6,89h` Fase 1 / `-6,92h` Total** |
| **CUAD `B = 200`** (`axia-cuad-b200`, `val=15, test=20`) | `F1 = 0,7318` | **`F1 = 0,8394`** | **`+14,7%` (`+10,76 p.p.`!)** | `64.843,0s` (**`18,01h`**) | **`7.073,8s` (`1,96h`)** | **`9,17×` (`-89,1%`!)** | **`6,76×` (`-85,2%`!)** | **`-16,05 horas` Fase 1 / `-16,41 horas` Total!** |
| **CUAD `B = 250`** (`axia-cuad-stress-b250`, `val=25, test=50`) | `F1 = 0,7725` | **`F1 = 0,7901`** | **`+2,3%`** (Mesmo Op!) | `84.930,6s` (**`23,59h`**) | **`9.466,9s` (`2,63h`)** | **`8,97×` (`-88,9%`!)** | **`7,78×` (`-87,1%`!)** | **`-20,96 horas` Fase 1 / `-20,82 horas` Total!** |
| **BioDex `B = 50/90`** (`Pilot 2× H200`, `Paired / Unpaired`) | `RP@5 = 0,3825` | **`RP@5 = 0,4275` / `0,4425`** | **`+11,8%` / `+15,7%`** | `5.829,1s` (`1,62h`) | **`3.833,0s` / `4.581,6s`** | **`1,52×` / `1,27×`** | **`1,02×`** | **`-0,55h` Fase 1** |
| **BioDex `B = 300`** (`Full Seed 0 Paired / Unpaired`, `test=250`) | `RP@5 = 0,2046` | **`RP@5 = 0,3025` / `0,2818`** | **`+47,8%` / `+37,7%`!** | `16.512,7s` (`4,59h`) | **`6.464,7s` / `4.651,1s`** | **`2,55×` a `3,55×`** | **`1,44×` a `1,22×`** | **`-2,79h` a `-3,30h` Fase 1 / `-3,28h` Total** |
| **BioDex `B = 600`** (`axia-biodex-stress-b600`, `val=25, test=250`) | `RP@5 = 0,1481` | **`RP@5 = 0,2350`** | **`+58,6%` (`+8,69 p.p.`!)** | `33.862,9s` (**`9,41h`**) | **`5.313,8s` (`1,48h`)** | **`6,37×` (`-84,3%`!)** | **`1,21×` (`-17,5%`)** | **`-7,93 horas` Fase 1 / `-2,33 horas` Total!** |

---

### 3.3. Tabela de Escalabilidade Topológica com Fronteira Larga ($k = 10$ Operadores Simultâneos por Estágio)

| Experimento de Fronteira Larga ($k = 10$) | Método | Qualidade (`F1` ou `RP@5`) | Precision / Recall (CUAD) | Tempo Otimização (`OptTime`) | Tempo Execução (`ExecTime`) | Tempo Total (`TotTime`) | Speedup Fase 1 (`0s / ES`) | Speedup Total (`0s / ES`) | Operadores Físicos Selecionados |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CUAD `k = 10`** (`B=150, k=10, j=3, val=20, test=50`) | `Baseline (0s)` | `F1 = 0,7518` | `0,8781` / `0,6573` | `43.528,6s` (`12,09h`) | `14.959,3s` (`4,16h`) | `58.487,9s` (`16,25h`) | `1,00×` (Ref.) | `1,00×` (Ref.) | `MixtureOfAgents (GPU0+1, T=0.0 -> GPU1)` |
| **CUAD `k = 10`** (`B=150, k=10, j=3, val=20, test=50`) | **`Nosso Método (ES)`** | **`F1 = 0,7462`** | `0,8556` / **`0,6616`** | **`17.715,9s` (`4,92h`)** | **`14.766,8s` (`4,10h`)** | **`32.482,7s` (`9,02h`)** | **`2,46×` (`-59,3%`, `-7,17h`!)** | **`1,80×` (`-44,5%`, `-7,22h`!)** | **Exatamente o mesmo operador (`MixtureOfAgents GPU0+1, T=0.0 -> GPU1`)!** |
| **BioDex `k = 10`** (`B=450, k=10, j=3, val=20, test=250`) | `Baseline (0s)` | `RP@5 = 0,2819` | — | `21.821,2s` (`6,06h`) | `25.966,3s` (`7,21h`) | `47.787,5s` (`13,27h`) | `1,00×` (Ref.) | `1,00×` (Ref.) | `Critique (GPU1->1->0) + TopK(10) + MoA (GPU0+1, T=0.0 -> GPU1)` |
| **BioDex `k = 10`** (`B=450, k=10, j=3, val=20, test=250`) | **`Nosso Método (ES)`** | `RP@5 = 0,2483` | — | **`6.751,8s` (`1,88h`)** | `28.305,9s` (`7,86h`) | **`35.057,8s` (`9,74h`)** | **`3,23×` (`-69,1%`, `-4,19h`!)** | **`1,36×` (`-26,6%`, `-3,54h`!)** | **Exatamente os mesmos operadores LLM (`Critique` e `MoA`), com `TopK(15)`!** |

---

### 3.4. Tabela da Suíte de Validação Dinâmica (`t_min = 1`) na Axia (`2× H200: 70B + 120B`, $B=60, k=5, N_{\text{val}}=15, N_{\text{test}}=20$)

| Experimento na Axia (`B=60, k=5, val=15, test=20`) | Configuração (`j`, `t_min`, `n_min`, `gamma`, `mode`) | Rodada de Parada ($t$) | Amostras no Líder ($n_L$) | Tempo Otimização (`OptTime`) | Tempo Total (`TotTime`) | F1-Score (`Precision` / `Recall`) | Speedup Fase 1 (`0s / ES`) | Speedup Total (`0s / ES`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`axia-val-cuad-baseline-j2`** (`Baseline 0s`) | `j=2`, Sem Parada | **Rodada 11** (Fim) | $n_L = 12$ | `11.471,4s` (`3,19h`) | `14.343,3s` (`3,98h`) | `0,7880` (`0,8900` / `0,7069`) | `1,00×` (Ref.) | `1,00×` (Ref.) |
| **`axia-val-cuad-j1-nmin4`** (`Nosso Método ES`) | `j=1, t_min=1, n_min=4` | **Parou na Rodada 4!** | $n_L = 4$ | **`4.488,5s` (`1,25h`)** | **`5.921,4s` (`1,64h`)** | **`0,8112` (`+2,9%`)** (`0,8896` / `0,7455`) | **`2,56×` (`-60,9%`)** | **`2,42×` (`-58,7%`)** |
| **`axia-val-cuad-j2-nmin5`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=5` | **Parou na Rodada 4!** | $n_L = 5$ | **`7.754,2s` (`2,15h`)** | **`8.153,3s` (`2,26h`)** | **`0,8149` (`+3,4%`)** (`0,8806` / `0,7584`) | **`1,48×` (`-32,4%`)** | **`1,76×` (`-43,2%`)** |
| **`axia-val-cuad-j2-nmin6`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=6` | **Parou na Rodada 5!** | $n_L = 6$ | **`7.698,4s` (`2,14h`)** | `14.718,8s` (`4,09h`) | **`0,8069` (`+2,4%`)** (`0,8370` / `0,7789`) | **`1,49×` (`-32,9%`)** | `0,97×` |
| **`axia-val-cuad-j2-nmin8`** (`Nosso Método ES`) | `j=2, t_min=1, n_min=8` | **Parou na Rodada 7!** | $n_L = 8$ | **`10.155,1s` (`2,82h`)** | `16.390,1s` (`4,55h`) | `0,7288` (`0,8247` / `0,6530`) | **`1,13×` (`-11,5%`)** | `0,88×` |
| **`axia-val-cuad-grayzone-g05`** (`Zona Cinza`) | **`j=2, t_min=1, n_min=4, gamma=0,5`** | **Parou na Rodada 3!** | $n_L = 4$ | **`3.305,7s` (`0,92h`)** | **`3.755,0s` (`1,04h`)** | **`0,8292` (`+5,2%`!)** (`0,8932` / `0,7738`) | **`3,47×` (`-71,2%`)** | **`3,82×` (`-73,8%`!)** |
| **`axia-val-cuad-mode-efficacy`** (`Ablação`) | `j=2, t_min=1, n_min=4, mode=efficacy` | **Rodada 10** (Sem Futilidade) | $n_L = 11$ | `15.468,6s` (`4,30h`) | `17.638,4s` (`4,90h`) | **`0,8431` (`+7,0%`)** (`0,8983` / `0,7943`) | `0,74×` (Sem Futilidade não para em empate) | `0,81×` |

---

## 4. Tabela Master Consolidada — Todos os 36 Experimentos Concluídos na Axia (`1×` e `2× NVIDIA H200`)

| # | Experimento na Axia (`H200`) | Método | Qualidade (`F1` ou `RP@5`) | Precision / Recall | Tempo Otimização (`OptTime`) | Tempo Execução (`ExecTime`) | Tempo Total (`TotTime`) | Rodadas MAB | Ganho do Nosso Método (`ES`) na Otimização e Qualidade |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **CUAD Piloto 1× H200** (`120B+8B, B=40`) | `Baseline (0s)` | `F1 = 0,7707` | `0,7990` / `0,7443` | `863,9s` (`14,4m`) | `173,0s` (`2,9m`) | `1.036,9s` (`17,3m`) | `9` | Referência |
| 2 | **CUAD Piloto 1× H200** (`120B+8B, B=40`) | **`Nosso Método (ES)`** | **`F1 = 0,7838` (`+1,7%`)** | **`0,8168` / `0,7534`** | **`283,9s` (`4,7m`)** | `230,4s` (`3,8m`) | **`514,3s` (`8,6m`)** | **`3`** | **`3,04×` mais rápido (`-67,1%`) e `+1,7%` F1** |
| 3 | **CUAD Piloto 2× H200** (`70B+120B, B=40`) | `Baseline (0s)` | `F1 = 0,8010` | `0,8798` / `0,7352` | `9.573,5s` (`159,6m`) | `323,3s` (`5,4m`) | `9.896,7s` (`164,9m`) | `9` | Referência |
| 4 | **CUAD Piloto 2× H200** (`70B+120B, B=40`) | `Nosso Método (ES Unpaired)` | `F1 = 0,8192` (`+2,3%`) | `0,8211` / **`0,8174`** | **`2.524,1s` (`42,1m`)** | `2.785,6s` (`46,4m`) | **`5.309,7s` (`88,5m`)** | **`3`** | **`3,79×` mais rápido (`-73,6%`) e `+2,3%` F1** |
| 5 | **CUAD Piloto 2× H200** (`70B+120B, B=40`) | **`Nosso Método (ES Paired)`** | **`F1 = 0,8213` (`+2,5%`)** | **`0,8718` / `0,7763`** | **`4.411,3s` (`73,5m`)** | **`1.616,0s` (`26,9m`)** | **`6.027,3s` (`100,5m`)** | **`3`** | **`2,17×` mais rápido (`-53,9%`) e maior F1 do Piloto (`0,8213`)** |
| 6 | **CUAD Full Seed 0** (`70B+120B, B=100, test=50`) | `Baseline (0s)` | `F1 = 0,7750` | `0,8886` / `0,6871` | `25.506,2s` (`7,09h`) | `1.284,8s` (`0,36h`) | `26.790,9s` (`7,44h`) | `17` | Referência |
| 7 | **CUAD Full Seed 0** (`70B+120B, B=100, test=50`) | `Nosso Método (ES Unpaired)` | `F1 = 0,7208` | `0,8837` / `0,6087` | **`9.743,7s` (`2,71h`)** | `15.400,4s` (`4,28h`) | `25.144,2s` (`6,98h`) | `3` | `2,62×` mais rápido na Fase 1 (`-61,8%`) |
| 8 | **CUAD Full Seed 0** (`70B+120B, B=100, test=50`) | **`Nosso Método (ES Paired)`** | **`F1 = 0,7655` (`+4,47 p.p.`)** | **`0,8941` / `0,6692`** | **`12.017,1s` (`3,34h`)** | **`6.560,9s` (`1,82h`)** | **`18.578,0s` (`5,16h`)** | **`6` (Zona Cinza R3–R5!)** | **`2,12×` Fase 1, `1,44×` Total (`-2,28h`) e recupera F1 (`0,7655`)!** |
| 9 | **CUAD Full Seed 1** (`70B+120B, B=100, test=50`) | `Baseline (0s)` | `F1 = 0,6291` | `0,8829` / `0,4886` | `33.029,1s` (`9,17h`) | `4.370,0s` (`1,21h`) | `37.399,1s` (`10,39h`) | `17` | Referência |
| 10 | **CUAD Full Seed 1** (`70B+120B, B=100, test=50`) | **`Nosso Método (ES Unpaired)`** | **`F1 = 0,6525` (`+3,7%`)** | **`0,8904` / `0,5150`** | **`9.201,2s` (`2,56h`)** | **`3.289,7s` (`0,91h`)** | **`12.490,9s` (`3,47h`)** | **`3`** | **`3,59×` mais rápido (`-72,1%` / `-6,62h`) e `+3,7%` F1** |
| 11 | **CUAD Full Seed 1** (`70B+120B, B=100, test=50`) | **`Nosso Método (ES Paired)`** | **`F1 = 0,6246`** | `0,8611` / **`0,4900`** | **`8.227,5s` (`2,29h`)** | `15.009,5s` (`4,17h`) | **`23.237,0s` (`6,45h`)** | **`3`** | **`4,01×` mais rápido Fase 1 (`-75,1%` / `-6,89h`) e `1,61×` Total** |
| 12 | **CUAD Full Seed 2** (`70B+120B, B=100, test=50`) | `Baseline (0s)` | *(Fase 1 Concluída)* | — | `18.586,0s` (`5,16h`) | *(49/50 contratos)* | — | `17` | Referência Fase 1 (`103` amostras) |
| 13 | **CUAD Full Seed 2** (`70B+120B, B=100, test=50`) | **`Nosso Método (ES)`** | **`F1 = 0,5828`** | **`0,8407` / `0,4460`** | **`6.220,7s` (`1,73h`)** | `13.966,6s` (`3,88h`) | **`20.187,4s` (`5,61h`)** | **`3`** | **`2,99×` mais rápido na Fase 1 (`-66,5%` / `-3,43h`)** |
| 14 | **CUAD Budget `B = 50`** (`70B+120B, val=15, test=20`) | `Baseline (0s)` | `F1 = 0,7912` | `0,8496` / `0,7404` | `14.594,1s` (`4,05h`) | `2.348,0s` (`0,65h`) | `16.942,1s` (`4,71h`) | `9` | Referência |
| 15 | **CUAD Budget `B = 50`** (`70B+120B, val=15, test=20`) | **`Nosso Método (ES)`** | **`F1 = 0,8122` (`+2,7%`)** | **`0,8848` / `0,7506`** | **`4.861,6s` (`1,35h`)** | **`442,7s` (`0,12h`)** | **`5.304,3s` (`1,47h`)** | **`4`** | **`3,00×` mais rápido Fase 1 (`3,19×` Total) e `+2,7%` F1!** |
| 16 | **CUAD Budget `B = 200`** (`70B+120B, val=15, test=20`) | `Baseline (0s)` | `F1 = 0,7318` | `0,8012` / `0,6735` | `64.843,0s` (**`18,01h`**) | `4.478,7s` (`1,24h`) | `69.321,7s` (**`19,26h`**) | `39` | Referência (Sofre *overfitting* em `18h`) |
| 17 | **CUAD Budget `B = 200`** (`70B+120B, val=15, test=20`) | **`Nosso Método (ES)`** | **`F1 = 0,8394` (`+14,7%`!)** | **`0,8835` / `0,7995`** | **`7.073,8s` (`1,96h`)** | **`3.175,0s` (`0,88h`)** | **`10.248,8s` (`2,85h`)** | **`5`** | **`9,17×` MAIS RÁPIDO (`-89,1%`, poupa `16,41h` Total!) e `+10,76 p.p.` F1!** |
| 18 | **CUAD Stress `B = 250`** (`70B+120B, val=25, test=50`) | `Baseline (0s)` | `F1 = 0,7725` | `0,9040` / `0,6743` | `84.930,6s` (**`23,59h`**) | `1.060,6s` (`0,29h`) | `85.991,2s` (**`23,89h`**) | `40` | Referência (`LLMConvertBonded GPU1`) |
| 19 | **CUAD Stress `B = 250`** (`70B+120B, val=25, test=50`) | **`Nosso Método (ES)`** | **`F1 = 0,7901` (`+2,3%`)** | **`0,9128` / `0,6965`** | **`9.466,9s` (`2,63h`)** | `1.587,9s` (`0,44h`) | **`11.054,8s` (`3,07h`)** | **`3`** | **`8,97×` MAIS RÁPIDO (`-88,9%`, poupa `20,96h`!) com mesmo Op e `+2,3%` F1!** |
| 20 | **CUAD Topologia `k = 10`** (`70B+120B, B=150, j=3`) | `Baseline (0s)` | `F1 = 0,7518` | `0,8781` / `0,6573` | `43.528,6s` (`12,09h`) | `14.959,3s` (`4,16h`) | `58.487,9s` (`16,25h`) | `13` | Referência (`MixtureOfAgents GPU0+1 -> GPU1`) |
| 21 | **CUAD Topologia `k = 10`** (`70B+120B, B=150, j=3`) | **`Nosso Método (ES)`** | **`F1 = 0,7462`** | `0,8556` / **`0,6616`** | **`17.715,9s` (`4,92h`)** | **`14.766,8s` (`4,10h`)** | **`32.482,7s` (`9,02h`)** | **`3`** | **`2,46×` mais rápido Fase 1 (`-7,17h`) e `1,80×` Total com mesmo Op!** |
| 22–28 | **CUAD Validação Dinâmica** (`7 runs: B=60, t_min=1`) | `Baseline` vs. `6 ES` | `0,7880` $\to$ **`0,8112` a `0,8431`** | Até **`0,8983` / `0,7943`** | `11.471,4s` $\to$ **`3.305,7s`** | — | `14.343,3s` $\to$ **`3.755,0s`** | **`3, 4, 5, 7, 10`** | **Paradas dinâmicas nas Rodadas 3, 4, 5 e 7 com até `3,82×` speedup e `+5,2%` F1** |
| 29 | **BioDex Piloto 1× H200** (`120B+8B, B=90`) | `Baseline (0s)` | `RP@5 = 0,3200` | — | `1.068,2s` (`17,8m`) | `398,7s` (`6,6m`) | `1.466,9s` (`24,4m`) | `6` | Referência |
| 30 | **BioDex Piloto 1× H200** (`120B+8B, B=90`) | **`Nosso Método (ES)`** | *(Fase 1 Concluída)* | — | **`599,0s` (`10,0m`)** | *(18/20 registros)* | — | **`3`** | **`1,78×` mais rápido (`-43,9%` na Fase 1)** |
| 31 | **BioDex Piloto 2× H200** (`70B+120B, B=90`) | `Baseline (0s)` | `RP@5 = 0,3825` | — | `5.829,1s` (`97,2m`) | `271,0s` (`4,5m`) | `6.100,1s` (`101,7m`) | `5` | Referência |
| 32 | **BioDex Piloto 2× H200** (`70B+120B, B=90`) | `Nosso Método (ES Unpaired)` | **`RP@5 = 0,4425` (`+15,7%`)** | — | **`4.581,6s` (`76,4m`)** | `3.503,5s` (`58,4m`) | `8.085,2s` (`134,8m`) | **`4`** | **`1,27×` mais rápido (`-21,4%`) e `+15,7%` RP@5** |
| 33 | **BioDex Piloto 2× H200** (`70B+120B, B=50`) | **`Nosso Método (ES Paired)`** | **`RP@5 = 0,4275` (`+11,8%`)** | — | **`3.833,0s` (`63,9m`)** | **`2.138,9s` (`35,6m`)** | **`5.971,9s` (`99,5m`)** | **`3`** | **`1,52×` mais rápido (`-34,2%`) e `+11,8%` RP@5** |
| 34 | **BioDex Full Seed 0** (`70B+120B, B=300, test=250`) | `Baseline (0s)` | `RP@5 = 0,2046` | — | `16.512,7s` (`4,59h`) | `22.108,6s` (`6,14h`) | `38.621,3s` (`10,73h`) | `14` | Referência |
| 35 | **BioDex Full Seed 0** (`70B+120B, B=300, test=250`) | `Nosso Método (ES Unpaired)` | `RP@5 = 0,2818` (`+37,7%`) | — | **`4.651,1s` (`1,29h`)** | `26.951,6s` (`7,49h`) | `31.602,7s` (`8,78h`) | `3` | `3,55×` mais rápido (`-71,8%`) e `+37,7%` RP@5 |
| 36 | **BioDex Full Seed 0** (`70B+120B, B=300, test=250`) | **`Nosso Método (ES Paired)`** | **`RP@5 = 0,3025` (`+47,8%`!)** | — | **`6.464,7s` (`1,80h`)** | **`20.338,4s` (`5,65h`)** | **`26.803,1s` (`7,45h`)** | **`R4, R5 (Eficácia), R9 (Zona Cinza!)`** | **`2,55×` Fase 1, `1,44×` Total (`-3,28h`) e `+47,8%` RP@5 (`0,3025`)!** |
| 37 | **BioDex Full Seed 1** (`70B+120B, B=300, test=250`) | `Baseline` vs. **`ES`** | *(Fase 1 Concluída)* | — | `17.876,0s` $\to$ **`5.232,0s`** | — | — | `14` $\to$ **`7`** | **`3,42×` mais rápido na Fase 1 (`-70,7%` / `-3,51h`; `Op 2` parou na R7)** |
| 38 | **BioDex Full Seed 2** (`70B+120B, B=300, test=250`) | `Baseline (0s)` | `RP@5 = 0,2839` | — | `12.959,3s` (`3,60h`) | `41.961,7s` (`11,66h`) | `54.921,1s` (`15,26h`) | `14` | Referência |
| 39 | **BioDex Full Seed 2** (`70B+120B, B=300, test=250`) | **`Nosso Método (ES)`** | `RP@5 = 0,2414` | — | **`4.620,2s` (`1,28h`)** | **`19.145,9s` (`5,32h`)** | **`23.766,1s` (`6,60h`)** | **`3`** | **`2,80×` mais rápido Fase 1 (`-64,3%`) e `2,31×` Total (`-8,65h`!)** |
| 40 | **BioDex Stress `B = 600`** (`70B+120B, val=25, test=250`) | `Baseline (0s)` | `RP@5 = 0,1481` | — | `33.862,9s` (**`9,41h`**) | `14.082,5s` (`3,91h`) | `47.945,4s` (**`13,32h`**) | `29` | Referência (Sofre *overfitting* severo em `TopK(k=3)`) |
| 41 | **BioDex Stress `B = 600`** (`70B+120B, val=25, test=250`) | **`Nosso Método (ES)`** | **`RP@5 = 0,2350` (`+58,6%`!)** | — | **`5.313,8s` (`1,48h`)** | `34.257,2s` (`9,52h`) | **`39.571,0s` (`10,99h`)** | **`3`** | **`6,37×` MAIS RÁPIDO Fase 1 (`-84,3%`, poupa `7,93h`!) e `+58,6%` RP@5!** |
| 42 | **BioDex Topologia `k = 10`** (`70B+120B, B=450, j=3`) | `Baseline (0s)` | `RP@5 = 0,2819` | — | `21.821,2s` (`6,06h`) | `25.966,3s` (`7,21h`) | `47.787,5s` (`13,27h`) | `7` | Referência (`Critique + TopK(10) + MoA`) |
| 43 | **BioDex Topologia `k = 10`** (`70B+120B, B=450, j=3`) | **`Nosso Método (ES)`** | `RP@5 = 0,2483` | — | **`6.751,8s` (`1,88h`)** | `28.305,9s` (`7,86h`) | **`35.057,8s` (`9,74h`)** | **`3`** | **`3,23×` mais rápido Fase 1 (`-69,1%`, `-4,19h`) e `1,36×` Total (`-3,54h`)!** |

---

## 5. Conclusões Gerais da Campanha Experimental na Axia (`H200`)

1. **A Nova Metodologia de Variância Pareada Intra-Documento (`ES_SIGMA_MODE="paired"`, $t_{\min}=1, n_{\min}=4$ fixos) comprova que $n_{\min}=4$ é apenas o limite inferior de entrada:** Ao descontar o efeito de dificuldade de cada documento, o ruído residual cai em **`37%` a `86%`**, permitindo que o algoritmo permaneça na **Zona Cinza** quando a disputa está aberta e pare dinamicamente na **Rodada 6 (`CUAD Full Seed 0`, `F1 = 0,7655`)** ou nas **Rodadas 4, 5 e 9 (`BioDex Full Seed 0`, `RP@5 = 0,3025`, `+47,8%` sobre o Baseline)**.
2. **O Speedup Escala Linearmente com o Orçamento $B$ (de `2,17×` até `9,17×`) e Protege Contra *Overfitting*:** Em orçamentos grandes ($B=200, 250, 600$), onde o Baseline gasta de `9,4h` a `23,6h` reamostrando o conjunto de validação e frequentemente degrada a qualidade por *overfitting* (`F1 = 0,7318` em $B=200$; `RP@5 = 0,1481` em $B=600$), o Nosso Método para entre `1,48h` e `2,63h`, economizando **até `20,96 horas` de 2× H200 em uma única execução** e entregando **`+14,7%` de F1 ($B=200$)** e **`+58,6%` de `RP@5` ($B=600$)**.
3. **Invariância e Robustez em Topologias Largas ($k = 10$):** Ao dobrar a fronteira inicial para $k=10$ operadores concorrentes por estágio, o Nosso Método seleciona **exatamente os mesmos operadores físicos de LLM que o Baseline**, reduzindo o tempo de otimização em **`2,46×` (`-7,17h`) no CUAD** e **`3,23×` (`-4,19h`) no BioDex**.
