#!/usr/bin/env python3
"""
run_crossswap.py -- component-level cross-swap causal test (paper Section 4.3,
Figure 4, Table 1 "Cross-swap asymmetry" row).

For each pair (HIGH-patchF selector, LOW-patchF selector) on Llama.BoolQ:

  keep_H / discard_H = HIGH selector's keep/discard split at the 30% budget
  keep_L / discard_L = LOW  selector's partition
  disagreement       = components the two rank on opposite sides of the cut

  three size-matched interventions, each measured as accuracy harm
  (base_acc - pruned_acc):
    * low-patchF discards  : prune only (discard_L intersect disagreement)
    * high-patchF discards  : prune only (discard_H intersect disagreement)
    * random baseline       : prune a random draw of the same size from the
                              set BOTH selectors KEEP (keep_H intersect keep_L)

Paper result: low-patchF discards are ~free, high-patchF discards are as
harmful as the random draw -> the damage sits exactly where patchF points the
wrong way. Emits crossswap_seed{SEED}.csv in the schema analyze.py reads.

    python run_crossswap.py --seeds 42
    python run_crossswap.py --seeds 42 --pruner llm-pruner   # -> results/llm_pruner/

Pruners: "zero" (default) is the paper's protocol: node-level scores, the
global bottom-k split over heads and MLP blocks, and each target set zeroed in
an exported checkpoint. "llm-pruner" removes the targets structurally.
    python ../analysis/analyze.py --results-dir ../results   # picks up the CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import grid as CFG          # noqa: E402
import common                             # noqa: E402
import lib                                # noqa: E402


def _keep_discard(spec, model, tok, disco_dir):
    """Component score dict -> (keep_set, discard_set) at spec.sparsity.
    keep = top (1 - sparsity) by score; discard = bottom sparsity."""
    importance, _cfg, _artifact, _scores_path = lib.scores_for_selector(
        spec, model, tok, disco_dir)
    scores = importance.scores if hasattr(importance, "scores") else importance
    items = sorted(scores.items(), key=lambda kv: float(kv[1]))
    n_discard = int(round(spec.sparsity * len(items)))
    discard = {k for k, _ in items[:n_discard]}
    keep = {k for k, _ in items[n_discard:]}
    return keep, discard


class ForcedDiscardImportance:
    """LLM-Pruner importance that removes EXACTLY a given component set:
    the target components get -inf importance, everything else +inf, so
    MetaPruner at the matched sparsity prunes precisely the target."""
    def __init__(self, target: set, all_scores: dict):
        import math
        self.scores = {k: (-math.inf if k in target else math.inf) for k in all_scores}

    def __call__(self, group, **_):
        import torch
        # delegate to CircuitKitImportance semantics via the score dict
        from circuitkit.applications.pruning.importance import CircuitKitImportance
        return CircuitKitImportance(self.scores)(group)


def _harm(spec, target: set, all_scores: dict, base_acc: float):
    """Accuracy harm from pruning exactly `target` (size-matched)."""
    model, tok = common.load_hf_model(spec.model_id, CFG.PRECISION)
    # sparsity that removes |target| components across the pruneable scope
    sp = len(target) / max(len(all_scores), 1)
    forced = ForcedDiscardImportance(target, all_scores)
    lib._apply_llm_pruner(model, forced, sp, tok)
    acc = lib.eval_task_accuracy(model, tok, spec.task)
    common.free(model, tok)
    return base_acc - acc


def _zero_split(spec):
    """Node-level scores and the grid's global keep/discard split."""
    scores, tl_model = lib.node_scores(spec)
    discard = set(lib.discard_set(scores, spec.sparsity))
    return tl_model, set(scores) - discard, discard


def run_pair_zero(high_sel, low_sel, seed, out_dir, rng, keep_checkpoints=False):
    """The paper's protocol: zero each size-matched target set."""
    m, t = CFG.CROSSSWAP_CELL
    hs = CFG.CellSpec(m, t, high_sel, seed)
    ls = CFG.CellSpec(m, t, low_sel, seed)
    model, tok = common.load_hf_model(hs.model_id, CFG.PRECISION)
    common.set_seed(seed)
    base_acc = lib.eval_task_accuracy(model, tok, t)
    del model, tok
    common.free()

    tl_model, keep_L, disc_L = _zero_split(ls)
    del tl_model
    common.free()
    tl_model, keep_H, disc_H = _zero_split(hs)

    disagree = disc_H ^ disc_L                        # symmetric difference
    targets = {"low": disc_L & disagree, "high": disc_H & disagree}
    both_keep = sorted(keep_H & keep_L)
    k = max(len(targets["high"]), 1)
    targets["random"] = set(rng.sample(both_keep, min(k, len(both_keep))))

    work = os.path.join(out_dir, "checkpoints")
    paths = {name: lib.export_zeroed(hs, tl_model, tgt, work,
                                     name=f"crossswap_{high_sel}_{low_sel}_s{seed}_{name}")
             for name, tgt in targets.items()}
    del tl_model
    common.free()
    harm = {name: base_acc - lib.eval_checkpoint(hs, path, keep=keep_checkpoints,
                                                 with_ppl=False)["acc_pruned"]
            for name, path in paths.items()}
    print(f"    |disagreement| = {len(disagree)} "
          f"(low {len(targets['low'])}, high {len(targets['high'])})")
    return {"low_patchf_harm": harm["low"], "high_patchf_harm": harm["high"],
            "random_harm": harm["random"]}


def run_pair(high_sel, low_sel, seed, out_dir, disco_dir, rng):
    """LLM-Pruner variant (--pruner llm-pruner)."""
    import random
    model, tok = common.load_hf_model(CFG.MODELS[CFG.CROSSSWAP_CELL[0]], CFG.PRECISION)
    common.set_seed(seed)
    m, t = CFG.CROSSSWAP_CELL
    base_acc = lib.eval_task_accuracy(model, tok, t)

    hs = CFG.CellSpec(m, t, high_sel, seed)
    ls = CFG.CellSpec(m, t, low_sel, seed)
    keep_H, disc_H = _keep_discard(hs, model, tok, disco_dir)
    keep_L, disc_L = _keep_discard(ls, model, tok, disco_dir)
    all_scores = {k: 0.0 for k in keep_H | disc_H}
    common.free(model, tok)

    disagree = (disc_H ^ disc_L)                      # symmetric difference
    tgt_low = disc_L & disagree
    tgt_high = disc_H & disagree
    both_keep = sorted(keep_H & keep_L)
    k = max(len(tgt_high), 1)
    tgt_rand = set(rng.sample(both_keep, min(k, len(both_keep))))

    return {
        "low_patchf_harm": _harm(ls, tgt_low, all_scores, base_acc),
        "high_patchf_harm": _harm(hs, tgt_high, all_scores, base_acc),
        "random_harm": _harm(hs, tgt_rand, all_scores, base_acc),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out-dir", default=os.path.join(here, "..", "results"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--pruner", choices=CFG.PRUNERS, default=CFG.DEFAULT_PRUNER)
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    import random
    out_dir = os.path.abspath(args.out_dir)
    disco_dir = os.path.join(out_dir, "discovery")
    if args.pruner == "llm-pruner":
        out_dir = os.path.join(out_dir, "llm_pruner")
        os.makedirs(out_dir, exist_ok=True)

    for seed in args.seeds:
        rng = random.Random(seed)
        rows = []
        for i, (high_sel, low_sel) in enumerate(CFG.CROSSSWAP_PAIRS, 1):
            print(f"[seed {seed}] Pair {i}: high={high_sel} vs low={low_sel}")
            if args.pruner == "zero":
                r = run_pair_zero(high_sel, low_sel, seed, out_dir, rng,
                                  keep_checkpoints=args.keep_checkpoints)
            else:
                r = run_pair(high_sel, low_sel, seed, out_dir, disco_dir, rng)
            rows.append({"pair": i, "high_patchf_selector": high_sel,
                         "low_patchf_selector": low_sel, **r})
            print(f"    low={r['low_patchf_harm']:+.3f} high={r['high_patchf_harm']:+.3f} "
                  f"random={r['random_harm']:+.3f}")
        path = os.path.join(out_dir, f"crossswap_seed{seed}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pair", "high_patchf_selector",
                                              "low_patchf_selector", "low_patchf_harm",
                                              "high_patchf_harm", "random_harm"])
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
