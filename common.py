"""
Shared utilities for the faithfulness-audit reproduction harness.

Seed handling, OOM-safe model loading, WikiText perplexity, and the
append-only master CSV that analysis/analyze.py consumes.

A100 (80 GB) notes
------------------
* Models are 3-4B; loaded in bfloat16 they occupy ~7-9 GB. The headroom is
  spent on discovery activation caches (EAP-IG integrated-gradient steps and
  IBCircuit's per-neuron mask stats), which is why IBCircuit runs at bs=1.
* Every cell reloads a FRESH model (pruning mutates weights in place) and
  frees it before the next cell -- see `free()`. This keeps peak memory flat
  across a long grid instead of leaking across selectors.
* If you still OOM: lower configs.grid.BATCH_SIZE, set
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, or shard discovery and
  prune/eval into separate processes (the drivers already checkpoint scores
  to disk between the two phases).
"""
from __future__ import annotations

import csv
import gc
import json
import os
import random
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches.

    The discovery data sampler also receives `seed` explicitly through the
    discover_circuit config (configs/grid + drivers/01_discover), so different
    seeds yield different 128-example discovery draws -- which is exactly what
    the multi-seed replication is meant to stress.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading / freeing
# ---------------------------------------------------------------------------
def load_hf_model(model_id: str, precision: str = "bfloat16", device: str = "cuda"):
    """Fresh HF causal-LM + tokenizer. Call once per cell; pruning is in-place."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[precision]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def free(*objs) -> None:
    """Drop references + empty the CUDA allocator between cells."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# WikiText-2 perplexity (coherence / confound measurement, paper Tables 3, 6)
# ---------------------------------------------------------------------------
def wikitext_perplexity(model, tokenizer, dataset: str = "wikitext-2-raw-v1",
                        max_samples: int = 256, stride: int = 512,
                        device: str = "cuda") -> float:
    """Sliding-window WikiText-2 PPL. Deterministic; used to flag selectors
    whose pruning breaks general language modeling (the confound the paper
    rules out with a partial correlation)."""
    import torch
    from datasets import load_dataset

    data = load_dataset("wikitext", dataset, split="test")
    text = "\n\n".join(t for t in data["text"] if t.strip())[: max_samples * stride]
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids.to(device)
    seq_len = ids.size(1)
    win = min(stride, getattr(model.config, "max_position_embeddings", 2048))

    nlls, n_tok = [], 0
    prev = 0
    model.eval()
    with torch.no_grad():
        for begin in range(0, seq_len, win):
            end = min(begin + win, seq_len)
            trg_len = end - prev
            chunk = ids[:, begin:end]
            tgt = chunk.clone()
            tgt[:, :-trg_len] = -100
            out = model(chunk, labels=tgt)
            nlls.append(out.loss.float() * trg_len)
            n_tok += trg_len
            prev = end
            if end == seq_len:
                break
    return float(torch.exp(torch.stack(nlls).sum() / max(n_tok, 1)).item())


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
MASTER_FIELDS = ["model", "task", "seed", "selector", "family",
                 "is_control", "patchF", "accRet", "ablF", "ppl"]


def append_master(results_dir: str, seed: int, row: dict) -> None:
    """Append one selector's result to master_seed{SEED}.csv (create + header
    on first write). analyze.py reads exactly these columns."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(results_dir, f"master_seed{seed}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MASTER_FIELDS})


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
