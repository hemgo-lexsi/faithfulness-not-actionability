#!/usr/bin/env python3
"""
run_interventions.py -- Section 4.2 robustness interventions that corroborate
the pruning result: higher-sparsity budgets, mixed-precision quantization, and
post-hoc LoRA healing (all on Llama.BoolQ).

These reuse the SHIPPED CircuitKit example scripts rather than re-implementing
them, so the code path is identical to the library's documented usage. This
driver just pins the paper's configuration and collects outputs.

  # 50% budget-robustness slice (Appendix C) -- this is just run_grid:
  python run_grid.py --cells llama:boolq --seeds 42 --sparsities 0.50

  # Mixed-precision quantization (3-bit W / 8-bit A, 5% float fallback):
  python run_interventions.py --which quantization --seed 42

  # LoRA healing of the pruned circuit (rank 16, 1k Alpaca examples, 2 epochs):
  python run_interventions.py --which lora-heal --seed 42

The quantization and healing example scripts live in the installed package at
  <circuitkit>/applications/quantization/examples/quantize_llama.py
  <circuitkit>/applications/pruning/examples/prune_llama.py   (--heal path)
This wrapper resolves those paths and calls them with the paper's flags.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import grid as CFG          # noqa: E402


def _example_path(*parts) -> str:
    import circuitkit
    root = os.path.dirname(circuitkit.__file__)
    return os.path.join(root, "applications", *parts)


def run_quantization(seed: int, out_dir: str):
    script = _example_path("quantization", "examples", "quantize_llama.py")
    q = CFG.QUANT
    cmd = [sys.executable, script,
           "--base-model", CFG.MODELS["llama"],
           "--task", "boolq",
           "--seed", str(seed),
           "--weight-bits", str(q["weight_bits"]),
           "--act-bits", str(q["act_bits"]),
           "--float-fallback", str(q["float_fallback"]),
           "--selectors", *CFG.ALL_SELECTORS,
           "--output-dir", os.path.join(out_dir, f"quant_seed{seed}")]
    print("[quantization]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_lora_heal(seed: int, out_dir: str):
    script = _example_path("pruning", "examples", "prune_llama.py")
    lora = CFG.LORA
    cmd = [sys.executable, script,
           "--base-model", CFG.MODELS["llama"],
           "--task", "boolq",
           "--seed", str(seed),
           "--attn-sparsity", str(CFG.HEADLINE_SPARSITY),
           "--mlp-sparsity", str(CFG.HEADLINE_SPARSITY),
           "--heal",
           "--ft-data", lora["data"],
           "--ft-examples", str(lora["ft_examples"]),
           "--ft-epochs", str(lora["epochs"]),
           "--output-dir", os.path.join(out_dir, f"loraheal_seed{seed}")]
    print("[lora-heal]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--which", required=True, choices=["quantization", "lora-heal"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(here, "..", "results"))
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    if args.which == "quantization":
        run_quantization(args.seed, out)
    else:
        run_lora_heal(args.seed, out)


if __name__ == "__main__":
    main()
