# Faithfulness Is Not Actionability: reproduction package

Code and data for **"Faithfulness Is Not Actionability: Extrinsic Validation
of Circuit Selectors Under Structured Pruning"** (Hem Gosalia, Pratinav Seth,
Aditya Kasliwal, and Vinay Kumar Sankarapu; Lexsi Labs), NeurIPS 2026 Workshop
on XAI4Science.

The paper asks whether the patch faithfulness (**patchF**) of a discovered
circuit predicts how much accuracy survives when the components a selector
ranks lowest are actually removed (**accRet**). Across 10 selectors, two
models, and three tasks (five model·task cells), the Spearman correlation
between the two changes sign with the task. It is significantly negative on
Llama·BoolQ, significantly positive on Llama·SST-2, and null on the other
three cells, so patchF is not a transferable keep/discard ranking.

Everything here runs on the public
[CircuitKIT](https://github.com/Lexsi-Labs/circuitkit) library
(`pip install circuitkit`); no internal code is required.

---

## Reproduce the paper's statistics in 30 seconds, no GPU

```bash
pip install numpy scipy pandas
python analysis/analyze.py --results-dir results/reference --self-test
```

This reads the shipped per-selector seed-42 values (`results/reference/`) and
recomputes every statistic in the paper: the five per-cell Spearman ρ, the
leave-one-family-out check, the coherence-confound partial correlation, the
cross-swap asymmetry, the multiplicity correction, and the pooled correlation.
Each is asserted against the published value, and every check should print
`PASS`.

The GPU experiments below regenerate the `results/reference/` values from
scratch.

---

## What reproduces what

Table and figure numbers follow the camera-ready version of the paper.

| Paper item | Command |
|---|---|
| Table 1 (five cells) | `drivers/run_grid.py --seeds 42` |
| Tables 3, 5, and 6 (per-cell selector detail) | same run as above |
| Table 2 (selector list and scoring rules; static) | `configs/grid.py`: `ALL_SELECTORS`, `SELECTOR_FAMILY` |
| Table 4 (leave-one-family-out) | `analysis/analyze.py` (no rerun) |
| Coherence confound (partial ρ given log PPL) | `analysis/analyze.py` (no rerun) |
| Appendix C, Table 7 (50% budget; rows go to `results/sparsity_0.50/`) | `drivers/run_grid.py --cells llama:boolq --seeds 42 --sparsities 0.50` |
| §4.3, Figure 4, Table 9 (cross-swap) | `drivers/run_crossswap.py --seeds 42` |
| §4.2, Tables 8 and 17 (LoRA healing, quantization) | `drivers/run_interventions.py --which {quantization,lora-heal}` |
| Multi-seed replication of the two significant cells (future work; see "Seeds" below) | `drivers/run_grid.py --cells llama:boolq llama:sst2 --seeds 42 123 456` |
| Everything at once | `bash run_all.sh` |

### Protocol

- **patchF** scores the top-30% circuit (`configs/grid.py`:
  `PATCHF_CIRCUIT_FRAC`) at every pruning budget, as in the paper.
- **accRet** uses the paper's intervention by default (`--pruner zero`). Each
  selector scores every attention head and MLP block
  (`circuitkit.selection.get_selector`), the components are ranked on one
  global list, and the bottom `round(N × sparsity)` are zeroed in an exported
  checkpoint (`circuitkit.evaluation.hf_checkpoint.save_pruned_checkpoint`):
  210 of 700 components on Llama and 92 of 306 on Gemma at 30%. Accuracy is
  lm-eval on that checkpoint, with the chat template, on the full validation
  split. Each run saves its scores and discard set to `results/scores/`.
- `--pruner llm-pruner` is an alternative that removes 30% of every layer with
  LLM-Pruner, in whole KV-head groups and single MLP channels. It is not the
  paper's protocol (it is closer to the per-layer variant of Appendix G), and
  its rows go to `results/llm_pruner/` so they never mix with the paper's.
- The OBS row (Llama·BoolQ only) and the quantization and LoRA-healing runs
  (`drivers/run_interventions.py`, which call CircuitKIT's own example
  scripts) use LLM-Pruner.
- Rediscovering the selectors does not return the grid's exact discard sets
  for five of the ten selectors (Appendix O), so fresh runs reproduce the
  protocol and the qualitative pattern, not every digit of Table 1.

Every driver appends to `results/master_seed{SEED}.csv`, and
`analysis/analyze.py` turns those files into the paper's tables. Nothing is
hard-coded: the analysis reads only the CSVs.

---

## Install (for the GPU experiments)

Install the library in one call and let its own dependency graph resolve.
Do not pin `transformers`, `huggingface_hub`, `datasets`, or `lm-eval` by
hand. CircuitKIT's `benchmarks` extra pulls an `lm-eval` build tested against
the `huggingface-hub` and `transformers` range CircuitKIT itself requires, and
pinning any one of them separately (as an earlier `requirements.txt` here did)
desyncs the graph. The result is an `ImportError` or `HfUriError` that is hard
to diagnose.

```bash
# 1. The library, from PyPI (A100/H100/H200 with CUDA 12.x assumed;
#    swap gpu-cu126 for your build).
pip install "circuitkit[benchmarks,gpu-cu126]" --extra-index-url https://download.pytorch.org/whl/cu126

# 2. This package's one extra dependency.
pip install "peft>=0.12"

# 3. The structural pruner (public, not on PyPI). Needed only for
#    --pruner llm-pruner, the OBS row, and run_interventions.py.
git clone https://github.com/horseee/LLM-Pruner
export LLM_PRUNER_PATH=$PWD/LLM-Pruner

# 4. Gated models: accept the licenses on the Hub, then log in.
huggingface-cli login
#   meta-llama/Llama-3.2-3B-Instruct   (Llama 3.2 Community License)
#   google/gemma-3-4b-it               (Gemma Terms of Use)
```

CircuitKIT expects `LLM-Pruner/` next to `src/` in its own repository root,
not wherever `LLM_PRUNER_PATH` points. If that check fails, symlink it from
the CircuitKIT root: `ln -s $LLM_PRUNER_PATH ./LLM-Pruner`.

On a shared machine where other tools already need a newer `transformers` or
`huggingface_hub` than CircuitKIT allows, install inside a venv created with
`python -m venv --system-site-packages`. It reuses the machine's CUDA-matched
torch and shadows only what CircuitKIT needs, so you never pin packages into
the system environment.

## Reproduce

```bash
export LLM_PRUNER_PATH=/path/to/LLM-Pruner    # for steps that use LLM-Pruner
bash run_all.sh                              # full grid, about 1 to 2 GPU-days
# or one slice:
python drivers/run_grid.py --cells llama:boolq --selectors eap-ig --seeds 42
python analysis/analyze.py --results-dir results
```

### Seeds

Every grid number in the paper comes from a single seed-42 draw. Gemma·BoolQ
was also run at seed 123 and keeps the same qualitative accRet ordering,
though not the same patchF values. For the camera-ready we rediscovered the
Llama·BoolQ selectors at seeds 42 and 43 with a separate audit harness. The
correlation stayed negative at both seeds (ρ = −0.61, p = 0.068; ρ = −0.91,
p = 0.0006; Appendix O). Rediscovery does not return the grid's exact sets
for five selectors, so this checks the protocol rather than the grid itself.

Full multi-seed replication of the two significant cells (Llama·BoolQ and
Llama·SST-2) remains future work. `configs/grid.py` sets `SEEDS` to
**{42, 123, 456}** for that purpose, and
`drivers/run_grid.py --cells llama:boolq llama:sst2 --seeds 42 123 456` runs
it. The seed reaches the 128-example discovery draw
(`discover_circuit(cfg["discovery"]["seed"])` and `data_params["seed"]`), the
Taylor and Wanda calibration split, and the random-control graph, so each seed
is an independent draw.

---

## Compute budget (single A100 80 GB)

| Stage | Runs | Approximate time |
|---|---|---|
| Discovery (5 circuit selectors × 5 cells, seed 42) | 25 | 3 to 5 h (IBCircuit and EAP-IG dominate) |
| Prune, accRet, and PPL (10 selectors × 5 cells) | 50 | 6 to 9 h |
| **Multi-seed BoolQ and SST-2 at seeds 123 and 456** | 40 | **about 15 to 25 GPU-h** |
| Cross-swap (3 pairs × 3 conditions) | 9 | 1 to 2 h |
| Quantization and LoRA healing | n/a | 2 to 4 h |

Peak memory stays flat because every cell reloads a fresh model and frees it
(`common.free`). If you run out of memory, lower `configs.grid.BATCH_SIZE`,
keep IBCircuit at `IB_BATCH_SIZE=1`, or set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The zeroing path writes one
pruned checkpoint at a time (about 7 to 9 GB) under `results/checkpoints/` and
deletes it after scoring unless you pass `--keep-checkpoints`. Discovery and
prune/evaluate save scores to disk between phases, so you can run them as
separate processes.

---

## Layout

```
configs/grid.py         all experiment settings (models, tasks, selectors, seeds, budgets)
common.py               seeds, OOM-safe model loading, WikiText PPL, master-CSV writer
drivers/
  lib.py                scoring, top-30% circuit faithfulness, zeroing and LLM-Pruner paths, accRet
  run_grid.py           patchF vs. accRet grid (Tables 1, 3, 5, and 6)
  run_crossswap.py      component-level cross-swap test (§4.3)
  run_interventions.py  quantization and LoRA healing (§4.2)
analysis/analyze.py     every statistic; --self-test asserts the published values
results/reference/      the paper's seed-42 values, so the analysis runs without a GPU
run_all.sh              one-shot reproduction
```

---

## License

The code in this repository is released under the MIT License (see
`LICENSE`). The CircuitKIT library has its own source-available license
(LSAL-1.1), and LLM-Pruner and lm-evaluation-harness are under their own
licenses (MIT and Apache-2.0). Model weights are governed by the Llama 3.2
Community License and the Gemma Terms of Use.

## Citation

```bibtex
@inproceedings{gosalia2026faithfulness,
  title     = {Faithfulness Is Not Actionability: Extrinsic Validation of Circuit Selectors Under Structured Pruning},
  author    = {Gosalia, Hem and Seth, Pratinav and Kasliwal, Aditya and Sankarapu, Vinay Kumar},
  booktitle = {NeurIPS 2026 Workshop on XAI4Science: Knowledge Discovery and Trust through Interpretable Foundation Models},
  year      = {2026}
}
```
