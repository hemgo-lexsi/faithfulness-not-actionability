# Faithfulness Is Not Actionability — reproduction package

Code and data to reproduce every number, table, and figure in
**"Faithfulness Is Not Actionability: Extrinsic Validation of Circuit
Selectors Under Structured Pruning."**

The paper asks whether patch-faithfulness (**patchF**) of a discovered circuit
predicts how much accuracy survives when its low-scored components are actually
removed (**accRet**). Across 10 selectors, two models, and three tasks (five
model·task cells) the Spearman correlation between the two flips sign by
task — significantly negative on Llama·BoolQ, significantly positive on
Llama·SST-2, null on the other three cells — so patchF is not a transferable
keep/discard ranking.

Everything here runs on the **public** `circuitkit` package (PyPI) — no
internal code required.

---

## TL;DR — reproduce the paper's statistics in 30 seconds, no GPU

```bash
pip install numpy scipy pandas
python analysis/analyze.py --results-dir results/reference --self-test
```

This ingests the shipped per-selector seed-42 values (`results/reference/`) and
recomputes **every** statistic in the paper — the five per-cell Spearman ρ,
leave-one-family-out, the coherence-confound partial correlation, the cross-swap
asymmetry, multiplicity, pooled correlation — asserting each against the
published value. All checks should print `PASS`.

The GPU experiments below regenerate those `results/reference/` values from
scratch.

---

## What reproduces what

| Paper item | Command |
|---|---|
| Table 1 (5 cells) | `drivers/run_grid.py --seeds 42` |
| Tables 3, 5, 6 — per-cell selector detail | same run as above |
| Table 2 — selector list / scoring rules (static, not a rerun) | `configs/grid.py`: `ALL_SELECTORS`, `SELECTOR_FAMILY` |
| Table 4 — leave-one-family-out | computed by `analysis/analyze.py` (no rerun) |
| Coherence confound (partial ρ \| log PPL) | computed by `analysis/analyze.py` |
| Appendix C — 50% budget | `drivers/run_grid.py --cells llama:boolq --seeds 42 --sparsities 0.50` |
| §4.3, Fig. 4, Table 9 — cross-swap | `drivers/run_crossswap.py --seeds 42` |
| §4.2, Table 8 — quantization, LoRA healing | `drivers/run_interventions.py --which {quantization,lora-heal}` |
| Multi-seed check on the two significant cells — **not yet run**, see "The seed question" below | `drivers/run_grid.py --cells llama:boolq llama:sst2 --seeds 42 123 456` |
| All tables/figures at once | `bash run_all.sh` |

Every driver appends to `results/master_seed{SEED}.csv`; `analysis/analyze.py`
turns those into the paper's tables. Nothing is hard-coded — the analysis reads
only the CSVs.

---

## Install (for the GPU experiments)

Install the library under test (public, source-available) in one call and
let its own dependency graph resolve — don't hand-pin
`transformers`/`huggingface_hub`/`datasets`/`lm-eval` yourself. circuitkit's
`benchmarks` extra pulls a `lm-eval` build tested against the same
`huggingface-hub`/`transformers` range circuitkit itself requires; pinning
any one of those packages independently (as an earlier version of this
package's `requirements.txt` did) desyncs the graph and produces exactly the
kind of `ImportError`/`HfUriError` that's hard to diagnose from the outside.

```bash
# 1. The library under test — one consolidated install from PyPI.
#    (A100/H100/H200 + CUDA 12.x assumed; swap gpu-cu126 for your build.)
pip install "circuitkit[benchmarks,gpu-cu126]" --extra-index-url https://download.pytorch.org/whl/cu126

# 2. This reproduction package's one extra dependency (not a circuitkit dep):
pip install "peft>=0.12"

# 3. Structural pruner (public; not on PyPI)
git clone https://github.com/horseee/LLM-Pruner
export LLM_PRUNER_PATH=$PWD/LLM-Pruner

# 4. Gated models: accept the licenses on the Hub, then
huggingface-cli login
#   meta-llama/Llama-3.2-3B-Instruct   (Llama 3.2 Community License)
#   google/gemma-3-4b-it               (Gemma Terms of Use)
```

If circuitkit's own hardcoded LLM-Pruner path expectation trips you up (it
looks for `LLM-Pruner/` as a sibling of `src/` inside its own repo root, not
wherever `LLM_PRUNER_PATH` points), symlink it there:
`ln -s $LLM_PRUNER_PATH ./LLM-Pruner` from the circuitkit repo root.

On a shared/pre-provisioned box where other tooling already needs a newer
`transformers`/`huggingface_hub` than circuitkit's range, do the install
above inside a venv created with `python -m venv --system-site-packages`
(reuses the box's existing CUDA-matched torch, shadows only what circuitkit
needs) rather than pinning individual packages into the system environment.

## Reproduce

```bash
export LLM_PRUNER_PATH=/path/to/LLM-Pruner
bash run_all.sh                              # full grid, ~1–2 GPU-days
# or one slice:
python drivers/run_grid.py --cells llama:boolq --selectors eap-ig --seeds 42
python analysis/analyze.py --results-dir results
```

### The seed question (important)

Every published seed-42 number in the paper rests on a single draw, with one
exception: Gemma·BoolQ, which is also checked at seed 123 and reproduces the
same ordering (folded into its per-cell table as a parenthetical). Multi-seed
replication of the two *significant* cells (Llama·BoolQ, Llama·SST-2) has not
been run yet — the paper states this plainly as the first item of future work.

The internal harness and `configs/grid.py`'s `SEEDS` default to
**{42, 123, 456}** for exactly this purpose:
`drivers/run_grid.py --cells llama:boolq llama:sst2 --seeds 42 123 456` fills
that gap. The seed propagates into the 128-example discovery draw
(`discover_circuit(cfg["discovery"]["seed"]`) and `data_params["seed"]`), the
Taylor/Wanda calibration split, and the random-control graph, so each seed is a
genuinely independent draw — which is what the replication is meant to stress.

---

## Compute budget (single A100 80 GB)

| Stage | Runs | ≈ time |
|---|---|---|
| Discovery (5 circuit selectors × 5 cells, seed 42) | 25 | 3–5 h (IBCircuit/EAP-IG dominate) |
| Prune + accRet + PPL (10 selectors × 5 cells) | 50 | 6–9 h |
| **Multi-seed BoolQ+SST-2 × seeds 123, 456** | 40 | **~15–25 GPU-h** |
| Cross-swap (3 pairs × 3 conditions) | 9 | 1–2 h |
| Quantization + LoRA healing | — | 2–4 h |

Peak memory stays flat because every cell reloads a fresh model and frees it
(`common.free`). If you OOM: lower `configs.grid.BATCH_SIZE`, keep IBCircuit at
`IB_BATCH_SIZE=1`, or `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Discovery and prune/eval checkpoint scores to disk between phases, so you can
shard them into separate processes.

---

## Layout

```
configs/grid.py         all experiment knobs (models, tasks, selectors, seeds, budgets)
common.py               seeds, OOM-safe model loading, WikiText PPL, master-CSV writer
drivers/
  lib.py                scoring · keep-set faithfulness · GQA-aware prune · accRet
  run_grid.py           patchF↔accRet grid  (Tables 1, 3, 5, 6)
  run_crossswap.py      component-level cross-swap causal test  (§4.3)
  run_interventions.py  quantization + LoRA healing  (§4.2)
analysis/analyze.py     every statistic; --self-test asserts published values
results/reference/      the paper's seed-42 values (so analysis runs GPU-free)
run_all.sh              one-shot reproduction
```

---

## Sharing / anonymity (double-blind)

This package is built to be reviewer-runnable **without** deanonymizing the
authors: it depends only on the public `circuitkit` package as a third-party
dependency and contains no author names or internal URLs.

For a double-blind submission, mirror it anonymously rather than linking a
named GitHub repo — [**anonymous.4open.science**](https://anonymous.4open.science)
serves an anonymized copy of a GitHub repo with the owner and any identifying
strings replaced. Push this tree to a private repo, register it there, fill in
every identifying string, and verify the resulting mirror in a
private/incognito window before citing it anywhere.

## License

The reproduction scripts here are released for review and replication. The
`circuitkit` library is under its own source-available license (LSAL-1.1);
LLM-Pruner and lm-evaluation-harness under their respective (MIT / Apache-2.0)
licenses. Model weights are governed by the Llama 3.2 Community License and the
Gemma Terms of Use.

## Citation

See the paper. (Author block omitted here for double-blind.)
