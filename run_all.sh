#!/usr/bin/env bash
# Reproduce every experiment in "Faithfulness Is Not Actionability".
# Assumes an A100 (80 GB), the install steps in README.md, and LLM_PRUNER_PATH
# set (used by the OBS row and step 5; the other steps zero components directly).
# Total compute is ~1-2 GPU-days for the full grid; see README for a breakdown.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-results}

echo "== 0. sanity: analysis reproduces the shipped seed-42 reference =="
python analysis/analyze.py --results-dir results/reference --self-test

echo "== 1. headline 5-cell x 10-selector grid @ 30% sparsity, seed 42 (Tables 1, 3, 5, 6) =="
python drivers/run_grid.py --seeds 42 --out-dir "$OUT"
# OBS has no node-level selector in circuitkit.selection, so it runs on the
# LLM-Pruner path (rows go to $OUT/llm_pruner/).
python drivers/run_grid.py --cells llama:boolq --selectors obs --seeds 42 --pruner llm-pruner --out-dir "$OUT"

echo "== 2. multi-seed replication of the two significant cells (not yet run in the paper) =="
python drivers/run_grid.py --cells llama:boolq llama:sst2 --seeds 42 123 456 --out-dir "$OUT"

echo "== 3. budget robustness: 50% sparsity on Llama.BoolQ (Appendix C; -> $OUT/sparsity_0.50) =="
python drivers/run_grid.py --cells llama:boolq --seeds 42 --sparsities 0.50 --out-dir "$OUT"

echo "== 4. cross-swap causal test (Section 4.3, Figure 4) =="
python drivers/run_crossswap.py --seeds 42 --out-dir "$OUT"

echo "== 5. quantization + LoRA healing (Section 4.2) =="
python drivers/run_interventions.py --which quantization --seed 42 --out-dir "$OUT"
python drivers/run_interventions.py --which lora-heal   --seed 42 --out-dir "$OUT"

echo "== 6. compute every statistic + fill every table =="
python analysis/analyze.py --results-dir "$OUT"

echo "done. Figures regenerate from the CSVs via ../figures (see paper repo)."
