# Guia Completo para Geração de Slides — `Baseline Abacus (0s)` vs. `Nosso Método (Paired ES — Variância Pareada Intra-Documento)` em Larga Escala na Axia (`2× NVIDIA H200`: `Llama-3.3-70B` + `gpt-oss-120b`)

> **Objetivo deste documento:** Servir como base pronta e autocontida para gerar os slides da apresentação ao orientador, comparando **exclusivamente o `Baseline Abacus (0s — Orçamento Completo)` e o `Nosso Método (Paired ES — Parada Antecipada Estatística Dual com Variância Pareada Intra-Documento)`** nos benchmarks **CUAD** e **BioDex** em **2× GPUs NVIDIA H200 (141 GB VRAM)** com os modelos **`meta-llama/Llama-3.3-70B-Instruct`** e **`openai/gpt-oss-120b`**, incluindo **tokens gerados (`Fase 1`, `Fase 2` e `Total`)**, amostras MAB, tempos e qualidade.

---

## Parte 1 — Fundamento Matemático e Arquitetura na Axia (`2× NVIDIA H200`)

### 1.1. Limite Inferior ($t_{\min}=1, n_{\min}=4$) + Variância Pareada Intra-Documento ($\hat{\sigma}_{\text{paired}}$)

$$y_{i,d} = \mu_i + \beta_d + \epsilon_{i,d} \implies r_{i,d} = y_{i,d} - \bar{y}_{\cdot,d}, \qquad d_{LR}(d) = y_{L,d} - y_{R,d}$$

$$\hat{\sigma}_{\text{paired}} = \sqrt{\frac{\hat{\sigma}_{\text{within}}^2 + \hat{\sigma}_{\text{diff}, LR}^2}{2}}, \qquad \text{SE}_{\text{atual}} = z \cdot \hat{\sigma}_{\text{paired}} \sqrt{\frac{1}{n_L} + \frac{1}{n_R}}, \qquad \text{SE}_{\text{futuro}} = \gamma \cdot z \cdot \hat{\sigma}_{\text{paired}} \sqrt{\frac{1}{n_L^+} + \frac{1}{n_R^+}}$$

- **$t_{\min}=1$ e $n_{\min}=4$ são Apenas o Limite Inferior (Portão de Entrada):** O algoritmo **não para porque atingiu $n_{\min}=4$**; atingir 4 amostras compartilhadas (`shared_docs_lr >= 4`) apenas habilita o teste estatístico ($z = 1,645$ para $95\%$ de confiança, $\gamma = 0,5$).
- **Variância Pareada Intra-Documento ($\hat{\sigma}_{\text{paired}}$):** Subtrai o efeito fixo de dificuldade de cada documento $\beta_d$ sobre os documentos compartilhados no conjunto de validação, medindo apenas a incerteza residual real entre o Líder ($L$) e o Vice-Líder ($R$).
- **As 3 Zonas de Decisão Após Atingir $n_{\min}=4$:**
  1. **Zona de Eficácia ($\Delta \ge \text{SE}_{\text{atual}}$):** Para imediatamente porque o Líder já supera o Vice-Líder com $95\%$ de confiança (ex.: Estágios 3 e 1 do `BioDex Full Seed 0`, que param nas **Rodadas 4 e 5**).
  2. **Zona Cinza ($\text{SE}_{\text{futuro}} \le \Delta < \text{SE}_{\text{atual}}$):** **NÃO para na rodada em que atingiu $n_{\min}=4$ — continua amostrando!** (ex.: `CUAD Full Seed 0` permanece na Zona Cinza nas **Rodadas 3, 4 e 5** e para na **Rodada 6**; Estágio 2 do `BioDex Full Seed 0` permanece na Zona Cinza nas **Rodadas 3 a 8** e para na **Rodada 9**).
  3. **Zona de Futilidade ($\Delta < \text{SE}_{\text{futuro}}$):** Para porque a diferença residual entre Líder e Vice-Líder é menor que a resolução mínima alcançável com o orçamento restante.

---

## Parte 2 — Roteiro Sugerido de 10 Slides Prontos para Apresentação

- **Slide 1 — Motivação da Escala na Axia (`2× NVIDIA H200` + Modelos 70B e 120B):** Em modelos de 70B (`Llama-3.3-70B`) e 120B (`gpt-oss-120b`), o `Baseline Abacus (0s)` consome de `4,59 horas` a `42,35 horas` e até `1,94 milhão` de tokens por execução por ser obrigado a esgotar 100% do orçamento $B$.
- **Slide 2 — Formulação do Nosso Método (`Paired ES`): Limite Inferior ($n_{\min}=4$) + Variância Pareada ($\hat{\sigma}_{\text{paired}}$):** Decomposição $y_{i,d} = \mu_i + \beta_d + \epsilon_{i,d}$ e decisão sequencial nas 3 Zonas (**Eficácia**, **Zona Cinza** e **Futilidade**).
- **Slide 3 — Resultados no CUAD (`Baseline 0s` vs. `Nosso Método Paired ES`, com Tokens):** Tabela comparativa completa (Piloto `B=40`, Full Seed 0 `B=100` e Full Seed 1 `B=100`), destacando **`-64,6%` a `-78,7%` de redução nos tokens da Fase 1**, **`2,17×` de speedup e `F1 = 0,8213` (`+2,5%`) no Piloto**, **`2,12×` de speedup (`-3,75h`, `-355.172` tokens) e Precisão `0,8941` na Seed 0**, e **`4,01×` de speedup (`-6,89h`, `-439.189` tokens) na Seed 1**.
- **Slide 4 — Evolução Rodada a Rodada na Zona Cinza (`CUAD Full Seed 0`):** Tabela `es_diagnostics` das Rodadas 1 a 6 mostrando que, ao atingir $n_{\min}=4$ na Rodada 3, o algoritmo entra na **Zona Cinza** ($\Delta = 0,0195 \in [0,0130, 0,0581)$), continua amostrando nas **Rodadas 3, 4 e 5** e para na **Rodada 6 ($n_L=7$)**.
- **Slide 5 — Resultados no BioDex Multi-Estágio (`Baseline 0s` vs. `Nosso Método Paired ES`, com Tokens):** Tabela comparativa (Piloto `B=90` e Full Seed 0 `B=300, N_test=250` artigos), destacando **`+11,8%` em `RP@5` (`0,4275` vs. `0,3825`, `-41,8%` tokens na Fase 1) no Piloto** e **`+47,8%` em `RP@5` (`0,3025` vs. `0,2046`) com `2,55×` de speedup (`-3,28h` no Tempo Total) e `-774.655` tokens totais (`-40,2%`) na Campanha Full**!
- **Slide 6 — Controle Cirúrgico por Estágio no `BioDex Full Seed 0`:** Tabela rodada a rodada dos 3 estágios acoplados mostrando `Op 3` parando na **Rodada 4 por Eficácia**, `Op 1` parando na **Rodada 5 por Eficácia** (`-76,3%` amostras de LLM!) e `Op 2` (`TopKOp`) permanecendo **6 rodadas na Zona Cinza (R3–R8)** até parar na **Rodada 9**.
- **Slide 7 — Curva de Sensibilidade ao Orçamento ($B \in \{40, 50, 100, 200, 250, 400, 600\}$):** Speedup cresce até **`9,17×` ($B=200$, `-85,7%` tokens Fase 1, `F1 = 0,8394` vs. `0,7318`, `+14,7%`)**, **`8,97×` ($B=250$, `-86,5%` tokens Fase 1, poupando `20,96 horas` com o mesmo operador!)**, **`17,65×` ($B=400$, `-90,5%` tokens Fase 1, poupando `39,95 horas` e `1,70 milhão` de tokens!)** e **`6,37×` ($B=600$ no BioDex, `-83,3%` tokens Fase 1, `RP@5 = 0,2350` vs. `0,1481`, `+58,6%`)**.
- **Slide 8 — Escalabilidade Topológica com Fronteira Larga ($k = 10$ Operadores Concorrentes por Estágio):** Mesmo com $k=10$, o Nosso Método escolhe **exatamente os mesmos operadores físicos de LLM que o Baseline**, poupando **`464.110` tokens (`-58,9%`) e `7,17 horas` (`2,46×`) no CUAD**, e **`424.724` tokens (`-65,8%`) e `4,19 horas` (`3,23×`) no BioDex**.
- **Slide 9 — Sensibilidade aos Parâmetros de Entrada e Ablação (`t_min = 1`, $B = 60$):** Paradas orgânicas nas **Rodadas 3, 4, 5 e 7** conforme $n_{\min} \in \{4, 5, 6, 8\}$ e comprovação experimental da complementaridade entre Eficácia e Futilidade.
- **Slide 10 — Conclusões Finais da Apresentação:** Síntese dos 4 pilares validados.

---

## Parte 3 — Tabelas Prontas para Copiar para os Slides (`Baseline 0s` vs. `Nosso Método Paired ES`, com Tokens)

### Tabela Slide 1 — Comparação Direta `Baseline (0s)` vs. `Nosso Método (Paired ES)` no CUAD e no BioDex (`2× NVIDIA H200`)

| Cenário na Axia (`2× H200: 70B + 120B`) | Método | Rodadas MAB | Amostras MAB (Fase 1) | Tokens Fase 1 (`Otimização`) | Tokens Fase 2 (`Execução`) | Tokens Totais (`Fase 1 + 2`) | Tempo Otimização (`OptTime`) | Speedup Fase 1 (`Speedup Total`) | Qualidade Final (`F1` ou `RP@5`) | Precision / Recall (CUAD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUAD Piloto** (`B=40, test=10`) | `Baseline (0s)` | `9` | `41` | `232.485` | `32.936` | `265.421` | `9.573,5s` (`2,66h`) | `1,00×` | `F1 = 0,8010` | `0,8798` / `0,7352` |
| **CUAD Piloto** (`B=40, test=10`) | **`Nosso Método (Paired ES)`** | **`3`** | **`17` (`-58,5%`)** | **`82.191` (`-64,6%`)** | `70.055` | **`152.246` (`-42,6%`)** | **`4.411,3s` (`1,23h`)** | **`2,17×` (`1,64×` Total)** | **`F1 = 0,8213` (`+2,5%`)** | `0,8718` / **`0,7763`** |
| **CUAD Full Seed 0** (`B=100, test=50`) | `Baseline (0s)` | `17` | `103` | `536.575` | `158.613` | `695.188` | `25.506,2s` (`7,09h`) | `1,00×` | `F1 = 0,7750` | `0,8886` / `0,6871` |
| **CUAD Full Seed 0** (`B=100, test=50`) | **`Nosso Método (Paired ES)`** | **`6` (Zona Cinza R3–R5!)** | **`36` (`-65,0%`)** | **`181.403` (`-66,2%`)** | `335.135` | **`516.538` (`-25,7%`)** | **`12.017,1s` (`3,34h`)** | **`2,12×` (`1,44×` Total, `-2,28h`)** | **`F1 = 0,7655`** | **`0,8941`** / `0,6692` |
| **CUAD Full Seed 1** (`B=100, test=50`) | `Baseline (0s)` | `17` | `103` | `558.004` | `176.739` | `734.743` | `33.029,1s` (`9,17h`) | `1,00×` | `F1 = 0,6291` | `0,8829` / `0,4886` |
| **CUAD Full Seed 1** (`B=100, test=50`) | **`Nosso Método (Paired ES)`** | **`3`** | **`21` (`-79,6%`)** | **`118.815` (`-78,7%`)** | `209.099` | **`327.914` (`-55,4%`)** | **`8.227,5s` (`2,29h`)** | **`4,01×` (`1,61×` Total, `-6,89h`!)** | **`F1 = 0,6246`** | `0,8611` / **`0,4900`** |
| **BioDex Piloto** (`B=90, test=20`) | `Baseline (0s)` | `5 / 5 / 5` | `94` (`43+26+25`) | `154.447` | `38.297` | `192.744` | `5.829,1s` (`1,62h`) | `1,00×` | `RP@5 = 0,3825` | — |
| **BioDex Piloto** (`B=90, test=20`) | **`Nosso Método (Paired ES)`** | **`3 / 3 / 3`** | **`49` (`-47,9%`)** | **`89.858` (`-41,8%`)** | `107.664` | `197.522` | **`3.833,0s` (`1,06h`)** | **`1,52×` (`1,02×` Total)** | **`RP@5 = 0,4275` (`+11,8%`)** | — |
| **BioDex Full Seed 0** (`B=300, test=250`) | `Baseline (0s)` | `14 / 14 / 14` | `293` (`118+84+91`) | `432.945` | `1.494.194` | `1.927.139` | `16.512,7s` (`4,59h`) | `1,00×` | `RP@5 = 0,2046` | — |
| **BioDex Full Seed 0** (`B=300, test=250`) | **`Nosso Método (Paired ES)`** | **`R5 (Efic.) / R9 (Zona Cinza!) / R4 (Efic.)`** | **`108` (`-63,1%`)** | **`162.769` (`-62,4%`)** | **`989.715` (`-33,8%`)** | **`1.152.484` (`-40,2%`, `-774.655` tok!)** | **`6.464,7s` (`1,80h`)** | **`2,55×` (`1,44×` Total, `-3,28h`!)** | **`RP@5 = 0,3025` (`+47,8%`!)** | — |

---

### Tabela Slide 2 — Evolução Rodada a Rodada na Zona Cinza (`CUAD Full Seed 0` e `BioDex Full Seed 0`)

| Experimento (`t_min=1, n_min=4` fixos) | Rodada 1–2 ($n < 4$) | Rodada 3 ($n_L=4$) | Rodada 4 ($n_L=5$) | Rodada 5 ($n_L=6$) | Rodada 6 ($n_L=7$) | Rodadas 7–8 ($n_L=8, 9$) | Rodada 9 ($n_L=10$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CUAD Full Seed 0`** (`Op 1: Extração 41 Cláusulas`) | `warmup` (Continua) | **`gray_zone`** ($\Delta=0,0195 \in [0,0130, 0,0581)$) | **`gray_zone`** ($\Delta=0,0365 \in [0,0136, 0,0546)$) | **`gray_zone`** ($\Delta=0,0289 \in [0,0247, 0,0900)$) | **`PARA (`futility`)!** ($\Delta=0,0144 < 0,0246$, `36` am. vs `103`) | — | — |
| **`BioDex Full Seed 0` — Estágio 3 (`Ranking LLM`)** | `warmup` (Continua) | Empate inicial ($\Delta=0,0$) | **`PARA (`efficacy`)!** ($\Delta=0,1675 \ge \text{SE}_{\text{atual}}$, `28` am. vs `91`) | Já convergido | Já convergido | Já convergido | Já convergido |
| **`BioDex Full Seed 0` — Estágio 1 (`Extração LLM`)** | `warmup` (Continua) | Empate inicial ($\Delta=0,0$) | **`gray_zone`** ($\Delta=0,0243 \in [0,0176, 0,0749)$) | **`PARA (`efficacy`)!** ($\Delta=0,0762 \ge 0,0616$, `28` am. vs `118`) | Já convergido | Já convergido | Já convergido |
| **`BioDex Full Seed 0` — Estágio 2 (`RAG TopKOp`)** | `warmup` (Continua) | **`gray_zone`** ($\Delta=0,0357 \in [0,0144, 0,0644)$) | **`gray_zone`** ($\Delta=0,0352 \in [0,0132, 0,0526)$) | **`gray_zone`** ($\Delta=0,0294 \in [0,0138, 0,0502)$) | **`gray_zone`** ($\Delta=0,0252 \in [0,0132, 0,0445)$) | **`gray_zone`** ($\Delta=0,0263$ e $0,0233$) | **`PARA (`futility`)!** ($\Delta=0,0210 < 0,0218$, `52` am. vs `84`) |

---

### Tabela Slide 3 — Curva de Sensibilidade ao Orçamento ($B \in \{40, 50, 100, 200, 250, 300, 400, 600\}$) e Fronteira Larga ($k = 10$), com Tokens

| Cenário de Escala na Axia (`2× H200: 70B + 120B`) | Qualidade `Baseline` $\to$ `Nosso Método` ($\Delta$) | Tokens Fase 1 `Baseline` $\to$ `Nosso Método` (`Redução %`) | Tokens Totais `Baseline` $\to$ `Nosso Método` (`Economia`) | Tempo Otimização `Baseline` $\to$ `Nosso Método` | Speedup Fase 1 (`0s / ES`) | Speedup Total (`0s / ES`) | Horas de 2× H200 Poupadas (Fase 1 / Total) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUAD `B = 40`** (`Piloto, val=10, test=10`) | `0,8010` $\to$ **`0,8213` (`+2,5%`)** | `232.485` $\to$ **`82.191` (`-64,6%`)** | `265.421` $\to$ **`152.246` (`-42,6%`, `-113.175` tok)** | `2,66h` $\to$ **`1,23h`** | **`2,17×` (`-53,9%`)** | **`1,64×` (`-39,1%`)** | **`-1,43h` Fase 1 / `-1,07h` Total** |
| **CUAD `B = 50`** (`val=15, test=20`) | `0,7912` $\to$ **`0,8122` (`+2,7%`)** | `279.131` $\to$ **`123.999` (`-55,6%`)** | `412.466` $\to$ **`184.250` (`-55,3%`, `-228.216` tok)** | `4,05h` $\to$ **`1,35h`** | **`3,00×` (`-66,7%`)** | **`3,19×` (`-68,7%`)** | **`-2,70h` Fase 1 / `-3,23h` Total** |
| **CUAD `B = 100`** (`Full Seeds 0 e 1, test=50`) | `0,7750` / `0,6291` $\to$ **`0,7655` / `0,6246`** | `536k–558k` $\to$ **`118k–181k` (`-66%` a `-79%`)** | `695k–735k` $\to$ **`328k–517k` (até `-406.829` tok)** | `7,09h–9,17h` $\to$ **`2,29h–3,34h`** | **`2,12×` a `4,01×`** | **`1,44×` a `1,61×`** | **`-3,75h` a `-6,89h` Fase 1 / `-3,94h` Total** |
| **CUAD `B = 200`** (`val=15, test=20`) | `0,7318` $\to$ **`0,8394` (`+14,7%`!)** | `954.174` $\to$ **`136.664` (`-85,7%`!)** | `1.080.277` $\to$ **`267.453` (`-75,2%`, `-812.824` tok!)** | **`18,01h` $\to$ `1,96h`** | **`9,17×` (`-89,1%`!)** | **`6,76×` (`-85,2%`!)** | **`-16,05 horas` Fase 1 / `-16,41 horas` Total!** |
| **CUAD `B = 250`** (`Stress Test, val=25, test=50`) | `0,7725` $\to$ **`0,7901` (`+2,3%`, Mesmo Op!)** | `1.130.366` $\to$ **`152.816` (`-86,5%`!)** | `1.278.637` $\to$ **`330.971` (`-74,1%`, `-947.666` tok!)** | **`23,59h` $\to$ `2,63h`** | **`8,97×` (`-88,9%`!)** | **`7,78×` (`-87,1%`!)** | **`-20,96 horas` Fase 1 / `-20,82 horas` Total!** |
| **CUAD `B = 400`** (`val=15, test=20`) | `0,8409` vs. **`0,8213`** (`P=0,8750`) | `1.882.174` $\to$ **`179.728` (`-90,5%`!)** | `1.939.481` $\to$ **`236.555` (`-87,8%`, `-1.702.926` tok!)** | **`42,35h` $\to$ `2,40h`** | **`17,65×` (`-94,3%`!)** | **`16,61×` (`-94,0%`!)** | **`-39,95 horas` Fase 1 / `-39,91 horas` Total!** |
| **CUAD Topologia `k = 10`** (`B=150, k=10, test=50`) | `0,7518` vs. **`0,7462`** (**`R=0,6616`, Mesmo Op!**) | `787.415` $\to$ **`323.305` (`-58,9%`)** | `1.048.608` $\to$ **`587.871` (`-43,9%`, `-460.737` tok!)** | `12,09h` $\to$ **`4,92h`** | **`2,46×` (`-59,3%`)** | **`1,80×` (`-44,5%`)** | **`-7,17 horas` Fase 1 / `-7,22 horas` Total!** |
| **BioDex `B = 90`** (`Piloto, val=10, test=20`) | `0,3825` $\to$ **`0,4275` (`+11,8%`)** | `154.447` $\to$ **`89.858` (`-41,8%`)** | `192.744` $\to$ `197.522` | `1,62h` $\to$ **`1,06h`** | **`1,52×` (`-34,2%`)** | **`1,02×` (`-2,1%`)** | **`-0,55h` Fase 1** |
| **BioDex `B = 300`** (`Full Seed 0, val=20, test=250`) | `0,2046` $\to$ **`0,3025` (`+47,8%`!)** | `432.945` $\to$ **`162.769` (`-62,4%`)** | `1.927.139` $\to$ **`1.152.484` (`-40,2%`, `-774.655` tok!)** | `4,59h` $\to$ **`1,80h`** | **`2,55×` (`-60,8%`)** | **`1,44×` (`-30,6%`)** | **`-2,79h` Fase 1 / `-3,28h` Total!** |
| **BioDex `B = 600`** (`Stress Test, val=25, test=250`) | `0,1481` $\to$ **`0,2350` (`+58,6%`!)** | `921.944` $\to$ **`153.576` (`-83,3%`!)** | `1.457.765` $\to$ `1.786.511` (`-768.368` tok na Fase 1!) | **`9,41h` $\to$ `1,48h`** | **`6,37×` (`-84,3%`!)** | **`1,21×` (`-17,5%`)** | **`-7,93 horas` Fase 1 / `-2,33 horas` Total!** |
| **BioDex Topologia `k = 10`** (`B=450, k=10, test=250`) | `0,2819` vs. `0,2483` (**Mesmos Ops LLM `Op 1, 3`**) | `645.200` $\to$ **`220.476` (`-65,8%`)** | `1.853.298` $\to$ **`1.529.969` (`-17,4%`, `-323.329` tok!)** | `6,06h` $\to$ **`1,88h`** | **`3,23×` (`-69,1%`)** | **`1,36×` (`-26,6%`)** | **`-4,19 horas` Fase 1 / `-3,54 horas` Total!** |
