# Cuad & Biodex – Setup & Run Guide

## Overview
This repository contains two end‑to‑end demonstration pipelines built on **Palimpzest/Abacus**:
- **Cuad** – a question‑answering workload built on the *Cuad* dataset.
- **Biodex** – a multi‑modal retrieval/QA workload using the *Biodex* benchmark.

Both demos showcase the Multi‑Armed Bandit (MAB) optimizer, Pareto‑front tracking, and the visualisation scripts you have generated.

---

## Prerequisites
1. **Linux/macOS** (tested on Ubuntu 22.04).  
2. **Conda** installed – we provide an environment named `kunumi`.
3. Python **3.12** (the `kunumi` env uses this version).
4. Access to the LLM APIs you plan to use (OpenAI, Anthropic, etc.).

---

## 1️⃣ Install the repository
```bash
# Clone the repo (if you haven't already)
git clone https://github.com/your-org/abacus-research.git
cd abacus-research
```

## 2️⃣ Activate the Conda environment
```bash
# The environment is located at /home/amarante/.conda/envs/kunumi
conda activate /home/amarante/.conda/envs/kunumi

# Verify the Python version
python --version   # → 3.12.x
```
If the environment does not exist, create it from the supplied `environment.yml` (or install the requirements manually):
```bash
conda env create -f environment.yml -n kunumi
conda activate kunumi
```

## 3️⃣ Install Python dependencies
```bash
# Inside the repository root
pip install -r requirements.txt
```
(If a `requirements.txt` file is missing, install the core Palimpzest package:
```bash
pip install -e .
```
)

---

## 📂 Data
The profiling data for both workloads lives in `opt-profiling-data/`.  It is automatically used by the demo scripts, so you **do not need** to download anything else.

---

## 4️⃣ Run the **Cuad** demo
```bash
# From the repository root
python abacus-research/cuad-demo.py
```
This script will:
- Load the Cuad dataset.
- Execute the MAB search with the three belief configurations (no prior, random prior, LLM prior).
- Produce the GIFs/PNG assets under `presentation_figures/`.

If you prefer the shell wrapper, use:
```bash
bash abacus-research/run_exp.sh cuad
```
(The `run_exp.sh` helper accepts a single argument – `cuad` or `biodex` – and runs the corresponding demo.)

---

## 5️⃣ Run the **Biodex** demo
```bash
python abacus-research/biodex-demo.py
```
Or via the helper script:
```bash
bash abacus-research/run_exp.sh biodex
```
The Biodex demo performs the same MAB search on the Biodex benchmark and emits the visualisations.

---

## 6️⃣ Additional utilities
- **Generate all GIFs / static images**: `python generate_presentation_summaries.py`
- **Run a full experiment batch**: `./run_all_polyaxon.sh` (requires a Polyaxon cluster).
- **Inspect intermediate logs**: look at `abacus-research/*.log` files.

---

## 🎨 Visualisation assets
All produced figures are stored under:
```
presentation_figures/
```
You will find:
- Clean frontier GIFs (`clean_frontier_gifs_fixed/`)
- Presentation‑ready assets (`presentation_final/`)
- Summary PNGs for slides.

---

## 🛠️ Common issues
| Symptom | Fix |
|---|---|
| `conda: command not found` | Install Miniconda/Anaconda and make sure `conda` is on your `$PATH`. |
| Missing LLM API key | Export `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc., before running the demos. |
| Permission denied on scripts | `chmod +x abacus-research/*.sh` |
| Python packages missing | Re‑run `pip install -r requirements.txt` inside the `kunumi` env. |

---

## 📚 Further reading
- **MAB pruning** – See `patched_src/palimpzest/query/execution/mab_execution_strategy.py` for the new probability‑of‑improvement logic.
- **Top‑K operator** – Updated in `patched_src/palimpzest/query/operators/topk.py` to respect the new pruning thresholds.
- **Presentation guide** – `presentation_figures/README.md` explains how to embed the generated assets into slide decks.

---

Happy experimenting! 🎉
