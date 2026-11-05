#!/usr/bin/env python3
"""
run_grid.py -- end-to-end patchF vs accRet grid (paper Tables 1, 3, 5, 6).

For every (model, task, selector, seed, sparsity) it produces one master-CSV
row: patchF, ablF (faithfulness of the top-30% circuit) and accRet, ppl
(post-prune). Base accuracy is measured once per (model, task, seed) and cached.

Examples
--------
# Headline 5-cell x 10-selector grid at 30% sparsity, seed 42 (paper Tables 1, 3, 5, 6):
python run_grid.py --seeds 42

# Multi-seed replication of the two significant cells (not yet run in the
# paper -- see README "seed question"; no dedicated table):
python run_grid.py --cells llama:boolq llama:sst2 --seeds 42 123 456

# Single cell / selector for debugging:
python run_grid.py --cells llama:boolq --selectors eap-ig --seeds 42

# 50% budget-robustness slice on Llama.BoolQ (Appendix C; rows go to
# <out-dir>/sparsity_0.50/):
python run_grid.py --cells llama:boolq --seeds 42 --sparsities 0.50

# LLM-Pruner variant (per-layer structural removal; not the paper's protocol).
# Writes to <out-dir>/llm_pruner/ so it never mixes with the zeroing rows:
python run_grid.py --cells llama:boolq --pruner llm-pruner

Pruners: "zero" (default) is the paper's protocol. Each selector's node-level
scores are saved to <out-dir>/scores/<tag>.json together with the discard set,
and the zeroed checkpoint is deleted after scoring (--keep-checkpoints keeps it).

Resume-safe: a (model,task,selector,seed,sparsity) already present in the
master CSV is skipped. Then run:  python ../analysis/analyze.py --results-dir <out>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import grid as CFG          # noqa: E402
import common                             # noqa: E402
import lib                                # noqa: E402

import pandas as pd                       # noqa: E402


def _already_done(master_path: str, spec: CFG.CellSpec) -> bool:
    if not os.path.exists(master_path):
        return False
    df = pd.read_csv(master_path)
    m = ((df.model == spec.model) & (df.task == spec.task)
         & (df.seed == spec.seed) & (df.selector == spec.selector))
    return bool(m.any())


def _parse_cells(vals):
    if not vals:
        return list(CFG.CELLS)
    out = []
    for v in vals:
        model, task = v.split(":")
        out.append((model, task))
    return out


def _base_accuracy(spec: CFG.CellSpec, base_cache: dict, model=None, tok=None) -> float:
    """Unpruned accuracy, once per (model, task, seed), scored like the pruned
    models (lm-eval, chat template)."""
    bkey = (spec.model, spec.task, spec.seed)
    if bkey not in base_cache:
        own = model is None
        if own:
            model, tok = common.load_hf_model(spec.model_id, CFG.PRECISION)
        base_cache[bkey] = lib.eval_task_accuracy(model, tok, spec.task)
        if own:
            del model, tok
            common.free()
        print(f"    [base] {bkey} acc = {base_cache[bkey]:.4f}")
    return base_cache[bkey]


def _row(spec, faith, accret, ppl):
    return {
        "model": spec.model, "task": spec.task, "seed": spec.seed,
        "selector": spec.selector, "family": spec.family,
        "is_control": int(spec.family == "control"),
        "patchF": ("" if faith["patchF"] != faith["patchF"] else round(faith["patchF"], 4)),
        "accRet": round(accret, 4),
        "ablF": ("" if faith["ablF"] != faith["ablF"] else round(faith["ablF"], 4)),
        "ppl": round(ppl, 1),
    }


def run_cell_zero(spec: CFG.CellSpec, out_dir: str, disco_dir: str, base_cache: dict,
                  keep_checkpoints: bool = False):
    """One selector on one cell with the paper's zeroing protocol."""
    base_acc = _base_accuracy(spec, base_cache)

    # faithfulness of the top-30% circuit (patchF/ablF). NaN only for
    # magnitude/taylor/wanda/multi_granular/random, whose patchF ships in
    # results/reference (see lib.scores_for_selector).
    cfg, artifact = lib.faithfulness_handle(spec, disco_dir)
    if artifact:
        faith = lib.circuit_faithfulness(spec, cfg, artifact=artifact)
    else:
        faith = {"patchF": float("nan"), "ablF": float("nan")}

    # node-level scores -> global bottom-k discard set -> zeroed checkpoint
    scores, tl_model = lib.node_scores(spec)
    discard = lib.discard_set(scores, spec.sparsity)
    common.save_json({"selector": spec.selector, "model": spec.model_id,
                      "task": spec.task, "seed": spec.seed,
                      "sparsity": spec.sparsity, "pruner": "zero",
                      "scores": scores, "discard": discard},
                     os.path.join(out_dir, "scores", spec.tag() + ".json"))
    ckpt = lib.export_zeroed(spec, tl_model, discard,
                             os.path.join(out_dir, "checkpoints"))
    del tl_model
    common.free()
    pr = lib.eval_checkpoint(spec, ckpt, keep=keep_checkpoints)
    accret = pr["acc_pruned"] / base_acc if base_acc else float("nan")
    print(f"    discarded {len(discard)} of {len(scores)} components")
    return _row(spec, faith, accret, pr["ppl"])


def run_cell(spec: CFG.CellSpec, out_dir: str, disco_dir: str, base_cache: dict):
    """One selector on one cell with LLM-Pruner (--pruner llm-pruner)."""
    model, tok = common.load_hf_model(spec.model_id, CFG.PRECISION)
    common.set_seed(spec.seed)

    base_acc = _base_accuracy(spec, base_cache, model, tok)

    # score -> importance (+ faithfulness handle: artifact or saved scores)
    importance, cfg, artifact, scores_path = lib.scores_for_selector(
        spec, model, tok, disco_dir)

    # faithfulness of the top-30% circuit (patchF/ablF). NaN only for
    # magnitude/taylor/random, whose patchF ships in results/reference
    # (see scores_for_selector note).
    if artifact or scores_path:
        faith = lib.circuit_faithfulness(spec, cfg, artifact=artifact,
                                         scores_path=scores_path)
    else:
        faith = {"patchF": float("nan"), "ablF": float("nan")}

    # prune + accRet + ppl (mutates model in place)
    pr = lib.prune_and_accret(spec, model, tok, importance)
    accret = pr["acc_pruned"] / base_acc if base_acc else float("nan")

    row = _row(spec, faith, accret, pr["ppl"])
    common.free(model, tok, importance)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out-dir", default=os.path.join(here, "..", "results"))
    ap.add_argument("--cells", nargs="+", default=None,
                    help="model:task pairs, e.g. llama:boolq (default: all 5).")
    ap.add_argument("--selectors", nargs="+", default=list(CFG.ALL_SELECTORS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--sparsities", nargs="+", type=float, default=[CFG.HEADLINE_SPARSITY])
    ap.add_argument("--pruner", choices=CFG.PRUNERS, default=CFG.DEFAULT_PRUNER,
                    help="zero (default): the paper's protocol; llm-pruner: per-layer "
                         "structural removal, written to <out-dir>/llm_pruner/.")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep the zeroed checkpoints (about 7-9 GB each).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    disco_dir = os.path.join(out_dir, "discovery")
    if args.pruner == "llm-pruner":
        out_dir = os.path.join(out_dir, "llm_pruner")
    os.makedirs(disco_dir, exist_ok=True)

    specs = []
    for model, task in _parse_cells(args.cells):
        for sel in args.selectors:
            if sel == "obs" and (model, task) != CFG.OBS_CELL:
                continue
            for seed in args.seeds:
                for sp in args.sparsities:
                    specs.append(CFG.CellSpec(model, task, sel, seed, sp))

    if args.pruner == "zero" and "obs" in {s.selector for s in specs}:
        print("  note: obs needs --pruner llm-pruner; skipping it here")
        specs = [s for s in specs if s.selector != "obs"]

    print(f"=== run_grid: {len(specs)} cells, pruner={args.pruner} ===")
    for s in specs:
        print(f"  {s.tag()}")
    if args.dry_run:
        return

    base_cache: dict = {}
    ok = fail = 0
    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        # The master CSV has no sparsity column, so each non-headline budget
        # gets its own folder (e.g. results/sparsity_0.50/) and never collides
        # with the 30% rows.
        cell_out = (out_dir if abs(spec.sparsity - CFG.HEADLINE_SPARSITY) < 1e-9
                    else os.path.join(out_dir, f"sparsity_{spec.sparsity:.2f}"))
        master = os.path.join(cell_out, f"master_seed{spec.seed}.csv")
        if not args.force and _already_done(master, spec):
            print(f"[{i}/{len(specs)}] skip {spec.tag()} (in master)")
            continue
        print(f"[{i}/{len(specs)}] {spec.tag()}")
        try:
            if args.pruner == "zero":
                row = run_cell_zero(spec, cell_out, disco_dir, base_cache,
                                    keep_checkpoints=args.keep_checkpoints)
            else:
                row = run_cell(spec, cell_out, disco_dir, base_cache)
            common.append_master(cell_out, spec.seed, row)
            print(f"    patchF={row['patchF']} accRet={row['accRet']} ppl={row['ppl']}")
            ok += 1
        except Exception as e:
            print(f"    FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
    print(f"\n=== done: {ok} ok, {fail} failed, {time.time()-t0:.0f}s ===")
    print(f"Analyze:  python {os.path.join(here, '..', 'analysis', 'analyze.py')} "
          f"--results-dir {out_dir}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
