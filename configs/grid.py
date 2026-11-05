"""
Central experiment configuration for
"Faithfulness Is Not Actionability: Extrinsic Validation of Circuit Selectors
Under Structured Pruning".

Every knob the paper reports lives here so the drivers stay declarative.
All values trace to the paper's Experimental Setup (Section 3) and appendix
tables. Edit here, never inline in a driver.

Reproduced with the PUBLIC library:  pip install circuitkit
plus LLM-Pruner (horseee/LLM-Pruner) and lm-evaluation-harness (EleutherAI).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Models  (HF ids; both loaded in the precision the paper used)
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-3-4b-it",
}
PRECISION = "bfloat16"

# ---------------------------------------------------------------------------
# Tasks  (built-in CircuitKit task specs; SST-2 uses authored contrastive
# pairs -- pass --custom-data to override with the paper's exact CSV).
# ---------------------------------------------------------------------------
TASKS: tuple[str, ...] = ("boolq", "sst2", "winogrande")

# The five model.task cells the paper measures (Table 1).
CELLS: tuple[tuple[str, str], ...] = (
    ("llama", "boolq"),
    ("llama", "sst2"),
    ("llama", "winogrande"),
    ("gemma", "boolq"),
    ("gemma", "winogrande"),
)
# The two significant cells; multi-seed replication of these is the paper's
# first item of future work (not yet run -- see README "seed question").
HEADLINE_CELLS: tuple[tuple[str, str], ...] = (("llama", "boolq"), ("llama", "sst2"))

# ---------------------------------------------------------------------------
# Selectors  (10 audited + OBS baseline on Llama.BoolQ only)
#   circuit-discovery -> circuitkit.api.discover_circuit -> CircuitKitImportance
#   pruning baselines -> importance computed at prune time
# ---------------------------------------------------------------------------
CIRCUIT_SELECTORS: tuple[str, ...] = ("eap", "eap-ig", "eap-gp", "ibcircuit", "cdt")
BASELINE_SELECTORS: tuple[str, ...] = ("magnitude", "taylor", "wanda",
                                       "multi_granular", "random")
ALL_SELECTORS: tuple[str, ...] = CIRCUIT_SELECTORS + BASELINE_SELECTORS

# selector -> paper's Section-3 method family (drives the LOFO analysis)
SELECTOR_FAMILY: dict[str, str] = {
    "eap": "circuit-discovery", "eap-ig": "circuit-discovery",
    "eap-gp": "circuit-discovery", "ibcircuit": "circuit-discovery",
    "cdt": "circuit-discovery",
    "magnitude": "weight-saliency", "taylor": "weight-saliency",
    "wanda": "weight-saliency",
    "multi_granular": "granularity-aware",
    "random": "control",
    "obs": "obs",
}
# OBS (SparseGPT scoring rule) is an 11th point on Llama.BoolQ only, and is
# excluded from that cell's headline n=10 rho (it appears in |patchF-1|, n=11).
OBS_CELL: tuple[str, str] = ("llama", "boolq")

# ---------------------------------------------------------------------------
# Seeds
#   The paper's headline grid is seed 42 only, with one exception: Gemma.BoolQ
#   is also checked at seed 123 and reproduces the same ordering. Multi-seed
#   replication of the two significant cells (HEADLINE_CELLS above) has not
#   been run yet. {42, 123, 456} is the internal harness's own convention for
#   that replication; keep 42 first, since it is the single draw every other
#   table uses.
# ---------------------------------------------------------------------------
SEEDS: tuple[int, ...] = (42, 123, 456)

# ---------------------------------------------------------------------------
# Pruning budgets
# ---------------------------------------------------------------------------
HEADLINE_SPARSITY = 0.30
SPARSITY_SWEEP: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50)  # robustness (App. B/C)
# patchF scores the top-30% circuit at every pruning budget (paper Section 3
# and Appendix C), not the top-(1 - sparsity) set the pruner keeps.
PATCHF_CIRCUIT_FRAC = 0.30

# How the discard set is removed for accRet.
#   "zero"       (default) the paper's intervention: rank every attention head
#                and MLP block on one global list and zero the bottom
#                round(N * sparsity) in an exported checkpoint (210 of 700
#                components on Llama, 92 of 306 on Gemma at 30%).
#   "llm-pruner" structural removal with LLM-Pruner at 30% of every layer,
#                in whole KV-head groups and single MLP channels. Not the
#                paper's headline protocol; closer to Appendix G's per-layer
#                variant.
PRUNERS: tuple[str, ...] = ("zero", "llm-pruner")
DEFAULT_PRUNER = "zero"

# Node-level selector scoring for the zeroing path, as the paper's grid ran it
# (circuitkit.selection.get_selector on a HookedTransformer).
SELECTOR_CFG = dict(batch_size=2, num_examples=128, ig_steps=5,
                    discovery_data_partition="discovery")
# TransformerLens's from_pretrained defaults, which the grid's models used.
TL_WEIGHT_PROCESSING = dict(fold_ln=True, center_writing_weights=True,
                            center_unembed=True, fold_value_biases=True)

SCOPE = "both"                # attention heads + MLP channels
HEAD_AGGREGATION = "mean"     # Q-heads -> KV-heads for GQA

# ---------------------------------------------------------------------------
# Discovery hyperparameters (Section 3)
# ---------------------------------------------------------------------------
DISCOVERY_EXAMPLES = 128      # disjoint from evaluation
PATCHF_EXAMPLES = 300         # held-out partition for faithfulness
PATCHF_EVAL_SEED_OFFSET = 1000  # eval draw seed = discovery seed + 1000 (grid)
PATCHF_BATCH_SIZE = 2
LEVEL = "neuron"              # whole attention heads + per-neuron MLP channels
MLP_HOOK = "post_act"         # required for d_mlp -> gate_proj column mapping
IG_STEPS = 5
BATCH_SIZE = 4
IB_BATCH_SIZE = 1             # IBCircuit activation cache OOMs above bs=1 on 3-4B

# IBCircuit optimizer (paper defaults)
IB = dict(num_epochs=1000, learning_rate=0.05, alpha=1.0, beta=0.001, alpha_loss="kl")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
LM_EVAL_BATCH = 8
BOOLQ_FEWSHOT = 0             # BoolQ: lm-eval defaults, full validation
SST2_FEWSHOT = 0
WINOGRANDE_FEWSHOT = 0
# The grid wrapped lm-eval prompts in the model's chat template ("auto": iff
# the tokenizer ships one, as for both instruction-tuned models here).
APPLY_CHAT_TEMPLATE = "auto"
PPL_DATASET = "wikitext-2-raw-v1"   # coherence / confound measurement

# ---------------------------------------------------------------------------
# Quantization (Section 4.2) -- mixed precision
# ---------------------------------------------------------------------------
QUANT = dict(weight_bits=3, act_bits=8, float_fallback=0.05)

# ---------------------------------------------------------------------------
# LoRA post-hoc healing (Section 4.2)
# ---------------------------------------------------------------------------
LORA = dict(rank=16, alpha=16, dropout=0.0, ft_examples=1000, epochs=2, lr=3e-4,
            data="alpaca")

# ---------------------------------------------------------------------------
# Cross-swap causal test (Section 4.3) -- three selector pairs on Llama.BoolQ
#   (high-patchF selector, low-patchF selector) on the same cell.
# ---------------------------------------------------------------------------
CROSSSWAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("cdt", "taylor"),        # Pair 1
    ("magnitude", "taylor"),  # Pair 2
    ("cdt", "ibcircuit"),     # Pair 3
)
CROSSSWAP_CELL: tuple[str, str] = ("llama", "boolq")


@dataclass
class CellSpec:
    """A fully-resolved (model, task, selector, seed, sparsity) run."""
    model: str
    task: str
    selector: str
    seed: int
    sparsity: float = HEADLINE_SPARSITY

    @property
    def model_id(self) -> str:
        return MODELS[self.model]

    @property
    def family(self) -> str:
        return SELECTOR_FAMILY[self.selector]

    def tag(self) -> str:
        sp = f"{self.sparsity:.2f}".replace(".", "p")
        return f"{self.model}_{self.task}_{self.selector}_seed{self.seed}_sp{sp}"


def headline_grid(seeds: tuple[int, ...] = (42,)) -> list[CellSpec]:
    """The 5-cell x 10-selector grid at 30% sparsity (+ OBS on Llama.BoolQ)."""
    specs: list[CellSpec] = []
    for model, task in CELLS:
        for sel in ALL_SELECTORS:
            for seed in seeds:
                specs.append(CellSpec(model, task, sel, seed))
    for seed in seeds:                       # OBS extra point
        specs.append(CellSpec(*OBS_CELL, "obs", seed))
    return specs


def multiseed_grid() -> list[CellSpec]:
    """The two significant cells across all seeds (no dedicated paper table;
    see README "seed question" -- this replication has not been run yet)."""
    return [CellSpec(m, t, sel, s)
            for (m, t) in HEADLINE_CELLS
            for sel in ALL_SELECTORS
            for s in SEEDS]
