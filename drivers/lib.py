"""
Core experiment primitives, built on the PUBLIC CircuitKit API.

One cell = one (model, task, selector, seed, sparsity). For every selector we:

  1. SCORE the model's components
       - circuit selectors (eap/eap-ig/eap-gp/ibcircuit/cdt): discover_circuit
       - baselines (magnitude/taylor/wanda/multi_granular/random): importance
         estimators evaluated on the same clean data
  2. Take the top-30% of components as "the circuit"
     (CFG.PATCHF_CIRCUIT_FRAC), whatever the pruning budget.
  3. patchF         = faithfulness of that circuit. On the zeroing path it is
                      computed node-level from the same scores
                      (node_patchf, the paper's protocol); on the LLM-Pruner
                      path it comes from neuron-level discover_circuit
                      artifacts, with ablF (circuit_faithfulness).
  4. accRet         = lm-eval accuracy after removing the discard set / base
                      accuracy. Two pruners (CFG.PRUNERS):
       - "zero" (default, the paper's protocol): node-level scores from
         circuitkit.selection; zero the bottom round(N * sparsity) heads and
         MLP blocks, ranked on one global list, in an exported checkpoint.
       - "llm-pruner": LLM-Pruner removes `sparsity` of every layer, in whole
         KV-head groups and MLP channels. Not the paper's headline protocol.
  5. ppl            = WikiText-2 perplexity of the pruned model (confound check).

It is wired to the public package so reviewers need only:

    pip install circuitkit            # public PyPI package
    pip install lm-eval               # EleutherAI/lm-evaluation-harness
    git clone https://github.com/horseee/LLM-Pruner   # only for --pruner llm-pruner

and set LLM_PRUNER_PATH (see README) for the LLM-Pruner path. Functions here
raise informative errors if an optional dependency is missing rather than
failing cryptically.

NOTE ON EXECUTION: these routines require a GPU + the 3-4B models and cannot be
run on a CPU box. The statistics layer (analysis/analyze.py) is fully validated
offline against the shipped seed-42 reference values; this layer reproduces the
raw (patchF, accRet, ablF, ppl) rows those statistics consume.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

# configs/ is a sibling of drivers/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import grid as CFG          # noqa: E402
import common                             # noqa: E402


def _require_llm_pruner() -> None:
    """LLM-Pruner is a separate public repo; put it on sys.path via env var."""
    p = os.environ.get("LLM_PRUNER_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)
    try:
        import LLMPruner.torch_pruning  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "LLM-Pruner not importable. Clone https://github.com/horseee/LLM-Pruner "
            "and set LLM_PRUNER_PATH to its root. Original error: %r" % e)


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------
def build_config(spec: CFG.CellSpec, out_dir: str) -> tuple[dict, str]:
    """The discover_circuit / evaluate_circuit config dict for a cell, plus the
    artifact path. Verified against circuitkit.api (discover_circuit at api.py,
    legacy q2 discover_all). Seed rides in both discovery["seed"] and
    data_params["seed"] so each seed is an independent 128-example draw."""
    artifact = os.path.join(out_dir, spec.tag() + ".pt")
    discovery = {
        "algorithm": spec.selector if spec.selector in CFG.CIRCUIT_SELECTORS else "eap-ig",
        "task": spec.task,
        "level": CFG.LEVEL,
        "mlp_hook": CFG.MLP_HOOK,
        "batch_size": CFG.IB_BATCH_SIZE if spec.selector == "ibcircuit" else CFG.BATCH_SIZE,
        "model_name": spec.model_id,
        "seed": spec.seed,
        "data_params": {"num_examples": CFG.DISCOVERY_EXAMPLES, "seed": spec.seed},
    }
    if spec.selector == "eap-ig":
        discovery["ig_steps"] = CFG.IG_STEPS
        discovery["method"] = "EAP-IG-inputs"
    elif spec.selector == "ibcircuit":
        discovery.update(CFG.IB)
    cfg = {
        "model": {"name": spec.model_id, "precision": CFG.PRECISION},
        "discovery": discovery,
        # This block sizes only the circuit CircuitKIT discovers and scores for
        # patchF: the top PATCHF_CIRCUIT_FRAC of components at every budget.
        # IBCircuit fixes that size in its discovery artifact, so it must be
        # set here, not just at evaluation. The accRet prune reads
        # spec.sparsity separately (prune_and_accret).
        "pruning": {"target_sparsity": 1.0 - CFG.PATCHF_CIRCUIT_FRAC,
                    "scope": CFG.SCOPE},
        "output_path": artifact,
        # We only read report.patching_score / .ablation_score (see
        # circuit_faithfulness). Without this, evaluate_circuit()'s
        # eval_cfg.get("pillars") comes back None, which it treats as "run
        # all six" -- stability alone re-runs discovery n_stability_runs=5
        # times internally, plus robustness builds extra corruption
        # dataloaders and generalization loads a second task. On a 60-cell
        # grid that is the difference between "fits on an H200" and OOM.
        "eval": {"pillars": ["patching", "ablation"]},
    }
    return cfg, artifact


def discover_scores(spec: CFG.CellSpec, out_dir: str) -> str:
    """Run circuit discovery for a circuit selector; return the scores sidecar
    path (circuitkit.api.discover_circuit auto-saves `*_scores.pt`)."""
    from circuitkit.api import discover_circuit
    cfg, artifact = build_config(spec, out_dir)
    sidecar = artifact.replace(".pt", "_scores.pt")
    if os.path.exists(sidecar):
        return sidecar
    common.set_seed(spec.seed)
    discover_circuit(cfg)
    # discover_circuit() loads its own HookedTransformer internally (on top
    # of the HF model this driver already holds for pruning/eval) and
    # doesn't free it before returning -- nudge Python + the CUDA allocator
    # so it doesn't linger into the next call in this same cell.
    common.free()
    return sidecar


def baseline_scores(spec: CFG.CellSpec, model, tokenizer):
    """Per-component importance for a non-circuit selector, as an
    LLM-Pruner importance object ready for MetaPruner.

    magnitude  : LLaMA-aware ||W_q||+||W_k||+||W_v||+||W_o|| (LLM-Pruner)
    taylor     : gradient*weight saliency, calibrated on the task's public
                 HF dataset (same convention the internal harness used for
                 MMLU Taylor calibration: pull raw examples directly rather
                 than through circuitkit's TransformerLens-only dataloader).
    random     : seeded RandomImportance (the control)

    wanda / multi_granular: NOT wired here. Both are registered in
    circuitkit.selection as HookedTransformer-only selectors (they read
    model.cfg / model.blocks / run_with_cache), so they cannot run against
    the plain HF model this driver prunes. Bridging them requires loading a
    second TransformerLens copy of the model and mapping their per-component
    scores back onto the HF module tree (the same aggregate_to_kv_heads +
    build_importance_dict bridge circuit selectors already use, but their
    score shape needs a small adapter first). Left as an explicit
    NotImplementedError rather than silently feeding the wrong model type.
    Their reference values ship in results/reference/master_seed42.csv.
    """
    _require_llm_pruner()          # must run before importing anything that
    import LLMPruner.torch_pruning as tp  # transitively imports LLMPruner
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner

    sel = spec.selector
    if sel == "magnitude":
        return llama_pruner.MagnitudeImportance(p=2)
    if sel == "random":
        common.set_seed(spec.seed)          # the control depends on the seed
        return tp.importance.RandomImportance()
    if sel == "taylor":
        _calibrate_taylor_grads(spec, model, tokenizer)
        return llama_pruner.TaylorImportance(taylor="param_first")
    if sel in ("wanda", "multi_granular"):
        raise NotImplementedError(
            f"{sel!r} is a HookedTransformer-only CircuitKit selector "
            "(circuitkit.selection) and needs a TransformerLens bridge not "
            "yet wired in this driver -- see baseline_scores() docstring. "
            "Skipping this cell; its reference value ships in "
            "results/reference/master_seed42.csv."
        )
    raise ValueError(f"unknown baseline selector {sel!r}")


# Public HF dataset id + a simple prompt template per task, used ONLY for
# Taylor calibration (need raw text on the clean distribution, no corruption
# pairing). Mirrors the internal harness's MMLU-via-datasets.load_dataset
# pattern rather than routing through circuitkit's task registry, which only
# exposes a TransformerLens-dependent build_dataloader() -- see
# tasks/builtins/boolq.py (no raw-text accessor).
_TAYLOR_CALIB_DATASET = {
    "boolq": ("google/boolq", "train",
             lambda ex: f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"),
    "sst2": ("glue", "train", "sst2",
            lambda ex: f"Review: {ex['sentence']}\nSentiment:"),
    "winogrande": ("winogrande", "train", "winogrande_xl",
                  lambda ex: ex["sentence"]),
}


def _clean_prompts(spec, n: int) -> list[str]:
    """Raw text from the task's public HF dataset (train split, so it never
    overlaps the eval split lm-eval scores accRet on)."""
    from datasets import load_dataset
    entry = _TAYLOR_CALIB_DATASET[spec.task]
    if len(entry) == 3:
        name, split, fmt = entry
        ds = load_dataset(name, split=split)
    else:
        name, split, config, fmt = entry
        ds = load_dataset(name, config, split=split)
    ds = ds.shuffle(seed=spec.seed).select(range(min(n, len(ds))))
    return [fmt(ds[i]) for i in range(len(ds))]


def _calibrate_taylor_grads(spec, model, tokenizer, device: str = "cuda"):
    """Populate weight.grad on the clean split so Taylor's grad*weight is
    defined -- same data distribution discovery saw, no corruption pairs."""
    import torch
    prompts = _clean_prompts(spec, n=CFG.DISCOVERY_EXAMPLES)
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()
    model.train(False)
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        model(**ids, labels=ids["input_ids"]).loss.backward()
    for p in model.parameters():            # backfill so Taylor never sees None
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)


# ---------------------------------------------------------------------------
# 2-3. Circuit faithfulness (patchF / ablF)
# ---------------------------------------------------------------------------
def circuit_faithfulness(spec: CFG.CellSpec, cfg: dict, artifact: str = None,
                         scores_path: str = None) -> dict:
    """patchF + ablF of the top-30% circuit (CFG.PATCHF_CIRCUIT_FRAC, set in
    build_config), via the verified public entry point
    circuitkit.api.evaluate_circuit -> FaithfulnessReport. The circuit does
    not depend on the pruning budget, matching the paper's protocol.

      report.patching_score  = Pillar 1, causal patching  = patchF
      report.ablation_score  = Pillar 2, ablation          = ablF

    Circuit selectors pass the discovery artifact; baseline selectors pass a
    saved per-component scores file (evaluate_circuit rebuilds the circuit
    graph from either). Restrict to the two cheap pillars for speed.
    """
    from circuitkit.api import evaluate_circuit
    kw = {"pruned_artifact_path": artifact} if artifact else {"scores_path": scores_path}
    report = evaluate_circuit(cfg, **kw)
    ablf = getattr(report, "ablation_score", float("nan"))
    result = {"patchF": float(report.patching_score),
             "ablF": float(ablf) if ablf is not None else float("nan")}
    # evaluate_circuit() always loads its own fresh HookedTransformer
    # (the `_model` reuse parameter is internal-only, per its docstring) --
    # same leak risk as discover_circuit() above.
    del report
    common.free()
    return result


# ---------------------------------------------------------------------------
# 4a. Zeroing prune + accRet (default, --pruner zero): the paper's protocol
# ---------------------------------------------------------------------------
def task_spec(name: str):
    """The registered CircuitKIT task, with a clear error for tasks the public
    package does not ship (the paper's authored SST-2 contrastive pairs)."""
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    try:
        return get_task(name)
    except Exception:             # the registry starts empty
        _bootstrap_builtin_tasks()
    try:
        return get_task(name)
    except Exception as e:
        raise NotImplementedError(
            f"task {name!r} is not registered in the installed circuitkit. The "
            "Llama-SST-2 cell uses the paper's authored label-flipping "
            "contrastive pairs, whose task module is not in public CircuitKIT "
            "yet (see README, Protocol).") from e


def node_scores(spec: CFG.CellSpec):
    """Signed score for every attention head ("A{l}.{h}") and MLP block
    ("MLP {l}"), computed as the paper's grid did:
    circuitkit.selection.get_selector(selector) on a HookedTransformer with
    TransformerLens's default weight processing and the grid's discovery
    settings (CFG.SELECTOR_CFG).

    Returns (scores, tl_model). The caller passes tl_model to export_zeroed()
    and then drops its reference.
    """
    if spec.selector == "obs":
        raise NotImplementedError(
            "obs has no node-level selector in circuitkit.selection; "
            "run it with --pruner llm-pruner")
    import circuitkit as ck
    from circuitkit.selection import get_selector   # registers all ten selectors
    task_spec(spec.task)                            # fail before loading a model
    common.set_seed(spec.seed)
    model = ck.load_model(spec.model_id, dtype=CFG.PRECISION, device="cuda",
                          **CFG.TL_WEIGHT_PROCESSING)
    model.cfg.model_name = spec.model_id   # the task-data cache keys on the full id
    model.eval()
    cfg = dict(CFG.SELECTOR_CFG, seed=spec.seed, model_name=spec.model_id)
    scores = get_selector(spec.selector)(model, spec.task, cfg)
    return {k: float(v) for k, v in scores.items()}, model


def discard_set(scores: dict, sparsity: float) -> list:
    """The grid's global rule: the bottom round(N * sparsity) components by
    signed score, attention heads and MLP blocks ranked on one list."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    n = max(1, int(round(len(ordered) * sparsity)))
    return [name for name, _ in ordered[:n]]


def export_zeroed(spec: CFG.CellSpec, tl_model, discard, work_dir: str,
                  name: str = None) -> str:
    """Write an HF checkpoint with `discard` zeroed and return its path.

    circuitkit's save_pruned_checkpoint zeroes each pruned head's q_proj rows
    and o_proj columns (shared K/V stay intact under GQA) and each pruned MLP
    block's gate/up/down weights, starting from the original HF weights. These
    are the checkpoints behind the paper's accRet (Appendix O).
    """
    from circuitkit.evaluation.hf_checkpoint import save_pruned_checkpoint
    path = os.path.join(work_dir, name or (spec.tag() + "_zeroed"))
    save_pruned_checkpoint(tl_model, list(discard), path, overwrite=True,
                           weights="original")
    return path


def eval_checkpoint(spec: CFG.CellSpec, path: str, keep: bool = False,
                    with_ppl: bool = True) -> dict:
    """lm-eval accuracy (and WikiText-2 PPL) of an exported checkpoint, scored
    exactly like the base model. Deletes the checkpoint unless `keep`."""
    import shutil
    model, tok = common.load_hf_model(path, CFG.PRECISION, tokenizer_id=spec.model_id)
    try:
        acc = eval_task_accuracy(model, tok, spec.task)
        ppl = (common.wikitext_perplexity(model, tok, CFG.PPL_DATASET)
               if with_ppl else float("nan"))
    finally:
        del model, tok
        common.free()
        if not keep:
            shutil.rmtree(path, ignore_errors=True)
    return {"acc_pruned": acc, "ppl": ppl}


def node_patchf(spec: CFG.CellSpec, tl_model, scores: dict) -> float:
    """The paper's patchF, from the same node-level scores and model that
    node_scores() returns.

    The circuit is the top PATCHF_CIRCUIT_FRAC of the node-level graph's
    forward slots by |score| (the input node is always kept), evaluated with
    activation patching on 300 held-out pairs drawn with seed + 1000 from the
    task's eval partition:
        patchF = (y_patch - y_corrupt) / (y_clean - y_corrupt),
    with y the task metric. This is the camera-ready audit's port of the
    grid's code, which reproduced the grid's clean and corrupt baselines
    (Appendix O). Returns NaN when the clean-corrupt gap is degenerate.
    """
    import torch
    from circuitkit.backends.eap.graph import AttentionNode, Graph, MLPNode
    from circuitkit.evaluation.evaluate import evaluate_baseline, evaluate_graph
    task = task_spec(spec.task)

    device = next(tl_model.parameters()).device
    dataloader = task.build_dataloader(tl_model, {
        "algorithm": "eap", "task": spec.task, "level": "node",
        "batch_size": CFG.PATCHF_BATCH_SIZE, "model_name": spec.model_id,
        "data_params": {"num_examples": CFG.PATCHF_EXAMPLES,
                        "seed": spec.seed + CFG.PATCHF_EVAL_SEED_OFFSET,
                        "data_partition": "eval"},
    }, device)
    task_metric = task.metric_fn()

    def y(logits, clean_logits, input_lengths, labels):
        return -task_metric(logits, clean_logits, input_lengths, labels)

    graph = Graph.from_model(tl_model, node_scores=True, neuron_level=False,
                             mlp_hook="mlp_out")
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            name = f"A{node.layer}.{node.head}"
        elif isinstance(node, MLPNode):
            name = f"MLP {node.layer}"
        else:
            continue
        graph.nodes_scores[graph.forward_index(node, attn_slice=False)] = \
            abs(scores.get(name, 0.0))
    graph.apply_topn(max(1, int(graph.n_forward * CFG.PATCHF_CIRCUIT_FRAC)),
                     level="node", absolute=True, reset=True)

    def mean(res):
        res = res[0] if isinstance(res, list) else res
        res = res.detach().float().cpu()
        return float(res.item()) if res.ndim == 0 else float(res.mean().item())

    with torch.inference_mode():
        y_clean = mean(evaluate_baseline(tl_model, dataloader, [y],
                                         run_corrupted=False, quiet=True))
        y_corrupt = mean(evaluate_baseline(tl_model, dataloader, [y],
                                           run_corrupted=True, quiet=True))
    y_patch = mean(evaluate_graph(tl_model, graph, dataloader, [y],
                                  intervention="patching", quiet=True))
    gap = y_clean - y_corrupt
    return float("nan") if abs(gap) <= 1e-8 else (y_patch - y_corrupt) / gap


# ---------------------------------------------------------------------------
# 4b. Structural prune + accRet (--pruner llm-pruner): not the paper's protocol
# ---------------------------------------------------------------------------
def prune_and_accret(spec: CFG.CellSpec, model, tokenizer, importance) -> dict:
    """Remove the discard set at `sparsity` with LLM-Pruner, eval accRet + PPL.

    This removes 30% of every layer, in whole KV-head groups and single MLP
    channels. The paper instead zeroes whole heads and MLP blocks ranked on one
    global list (section 4a above, the default).

    accRet = acc_pruned / acc_base; base is measured once per (model, task, seed)
    and cached by the caller. Returns {accRet_raw acc, ppl}. Mirrors the
    internal harness _apply_llm_pruner exactly (GQA-aware, whole-KV-head prune).
    """
    _require_llm_pruner()
    _apply_llm_pruner(model, importance, spec.sparsity, tokenizer)
    acc = eval_task_accuracy(model, tokenizer, spec.task)
    ppl = common.wikitext_perplexity(model, tokenizer, CFG.PPL_DATASET)
    return {"acc_pruned": acc, "ppl": ppl}


def _apply_llm_pruner(model, importance, sparsity, tokenizer, keep_grads=False):
    """One-shot GQA-aware block prune -- verbatim logic from the internal
    q2 harness (whole KV heads via consecutive_groups=head_dim; MLP channels
    on gate_proj). Skips k_proj if n_kv*sparsity < 1 head."""
    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from circuitkit.applications import (
        detect_model_architecture, get_arch_config, get_layers,
        get_attn_proj, get_mlp_proj,
    )
    arch = get_arch_config(detect_model_architecture(model))
    layers = get_layers(model, arch)
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    skip_attn = int(n_kv * sparsity) < 1

    ch_sparsity_dict, consecutive_groups, roots = {}, {}, []
    for layer in layers:
        if not skip_attn:
            k = get_attn_proj(layer, arch, "k_proj")
            ch_sparsity_dict[k] = sparsity
            consecutive_groups[k] = layer.self_attn.head_dim
            roots.append(k)
        g = get_mlp_proj(layer, arch, "gate_proj")
        if g is not None:
            ch_sparsity_dict[g] = sparsity
            roots.append(g)

    rms = next((type(m) for m in model.modules() if "RMSNorm" in type(m).__name__), None)
    custom = {rms: llama_pruner.hf_rmsnorm_pruner} if rms else {}
    prompts = tokenizer(["The capital of France is", "In recent years,"],
                        return_tensors="pt", padding=True)["input_ids"].to(
                            next(model.parameters()).device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.config.use_cache = False
    pruner = tp.pruner.MetaPruner(
        model, prompts, importance=importance, global_pruning=False,
        iterative_steps=1, ch_sparsity=sparsity, ch_sparsity_dict=ch_sparsity_dict,
        ignored_layers=[], consecutive_groups=consecutive_groups,
        customized_pruners=custom, root_module_types=None, root_instances=roots)
    model.config.use_cache = True
    if not keep_grads:
        model.zero_grad()
    pruner.step()
    for layer in layers:
        a = layer.self_attn
        a.num_heads = a.q_proj.weight.shape[0] // a.head_dim
        a.num_key_value_heads = a.k_proj.weight.shape[0] // a.head_dim
    common.free(pruner)


def eval_task_accuracy(model, tokenizer, task: str, limit: Optional[int] = None) -> float:
    """lm-eval-harness accuracy of the in-memory model on the task.

    `confirm_run_unsafe_code` was added to simple_evaluate() in a specific
    lm-eval release and isn't present in every version pip resolves (older
    releases don't have it; some newer ones renamed/dropped it). Passing it
    unconditionally makes this brittle across environments, so only pass it
    when the installed simple_evaluate() actually accepts it.
    """
    import inspect
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    fewshot = {"boolq": CFG.BOOLQ_FEWSHOT, "sst2": CFG.SST2_FEWSHOT,
               "winogrande": CFG.WINOGRANDE_FEWSHOT}.get(task, 0)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=CFG.LM_EVAL_BATCH)
    kwargs = dict(model=lm, tasks=[task], num_fewshot=fewshot, limit=limit)
    params = inspect.signature(simple_evaluate).parameters
    if "confirm_run_unsafe_code" in params:
        kwargs["confirm_run_unsafe_code"] = True
    # The grid scored instruction-tuned models inside their chat template.
    use_chat = CFG.APPLY_CHAT_TEMPLATE
    if use_chat == "auto":
        use_chat = getattr(tokenizer, "chat_template", None) is not None
    if "apply_chat_template" in params:
        kwargs["apply_chat_template"] = bool(use_chat)
    elif use_chat:
        import warnings
        warnings.warn("installed lm-eval cannot apply the chat template; "
                      "accuracies will not match the paper's protocol")
    out = simple_evaluate(**kwargs)
    res = out["results"][task]
    for k in ("acc,none", "acc", "acc_norm,none"):
        if k in res:
            return float(res[k])
    raise KeyError(f"no acc metric in {res}")


def scores_for_selector(spec, model, tokenizer, disco_dir):
    """Unified per-selector resolution. Returns:

        (importance, cfg, artifact, scores_path)

    where `importance` drives the structural prune (accRet, every selector),
    and (`cfg`, `artifact`|`scores_path`) drive circuit_faithfulness (patchF).

    * circuit selectors (eap/eap-ig/eap-gp/ibcircuit/cdt, +obs)
          -> exact accRet AND exact patchF (discovery artifact + sidecar).
    * magnitude / taylor / random
          -> exact accRet; patchF is NaN for now (artifact=None, scores_path=None
             -- see baseline_scores() docstring). Reference patchF ships in
             results/reference/master_seed42.csv.
    * wanda / multi_granular
          -> raises NotImplementedError (HookedTransformer-only selectors not
             yet bridged to the HF-model prune path). run_grid.py's per-cell
             try/except logs the FAIL and continues the grid.
    """
    cfg, artifact = build_config(spec, disco_dir)

    if spec.selector in CFG.CIRCUIT_SELECTORS or spec.selector == "obs":
        _require_llm_pruner()      # CircuitKitImportance imports LLMPruner at
        from circuitkit.applications.selective_finetuning.score_loader import load_scores
        from circuitkit.applications.pruning.score_extractor import (
            aggregate_to_kv_heads, build_importance_dict)
        from circuitkit.applications.pruning.importance import CircuitKitImportance
        sidecar = discover_scores(spec, disco_dir)
        head, mlp, meta = load_scores(scores_path=sidecar, model_name=spec.model_id)
        n_q = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_q)
        kv = aggregate_to_kv_heads(head, n_q, n_kv, reduction=CFG.HEAD_AGGREGATION)
        n_layers = model.config.num_hidden_layers
        sd = build_importance_dict(model, kv, mlp,
                                   attn_layers=list(range(n_layers)),
                                   mlp_layers=list(range(n_layers)))
        return CircuitKitImportance(sd), cfg, artifact, sidecar

    # magnitude/taylor/random (wanda/multi_granular raise inside baseline_scores):
    # `artifact` from build_config() above was never written for these -- no
    # discover_circuit() call happens on this branch -- so it must NOT be
    # forwarded, or evaluate_circuit() 404s on a .pt that doesn't exist.
    importance = baseline_scores(spec, model, tokenizer)
    return importance, cfg, None, None
