# Copied verbatim (below this header) from the paper's grid code,
# aryapratinavseth/circuitkit @ 43273cd, examples/experiments/rebuttal/gpu/sst2_task.py.
# lib.task_spec() registers it automatically for the "sst2" cells.

"""SST-2 as a first-class contrastive task (T3.2, replaces RTE) -- the third
binary English task done *properly*.

Why SST-2 and not RTE
---------------------
The third task must let us measure patch-faithfulness, which needs a CLEAN vs
CORRUPT pair whose only difference is the behaviour under study. RTE failed at
this: its meaning-preserving paraphrase corruption left y_clean == y_corrupt, so
patchF was undefined (degenerate denominator) and the EAP-family attribution
collapsed. Sentiment is the canonical setting for clean minimal-pair contrastive
circuits (Tigges et al. 2023, "Linear Representations of Sentiment in LLMs"): a
single sentiment-bearing adjective flips the label, so a one-token swap gives a
decisive, token-aligned, grammatical counter-factual.

Design (satisfies the four circuit-discovery constraints by construction)
-------------------------------------------------------------------------
1. Rigid template, single adjective slot:
       "Review: The <noun> was {adj}.\nSentiment (positive or negative):"
   The format-following circuit is held fixed; only the sentiment adjective and
   the answer token vary.
2. Minimal, label-flipping edit: clean uses a positive adjective (answer
   " positive"); corrupt swaps in a negative adjective (answer " negative"),
   or vice-versa. Edit distance = one word.
3. Strict token alignment: adjectives and answer tokens are filtered to
   SINGLE tokens under *this model's* tokenizer, and every clean/corrupt pair is
   kept only if the two full prompts tokenize to the SAME length. So patching
   along the sequence dimension is position-aligned.
4. Grammatical parity: every swap is adjective-for-adjective in the same
   predicate slot, so the corrupt sentence is as grammatical as the clean one
   (we never probe a syntax-error / perplexity-spike circuit).

The metric is logit_diff = logit(answer) - logit(answer_incorrect) at the final
prompt position, i.e. logit(" positive") - logit(" negative") on a clean
(positive) example. Discovery uses these controlled pairs; benchmark ACCURACY
(accRet) is still measured on the REAL SST-2 validation set via lm-eval
(task_to_lm_eval {sst2: sst2}) -- the standard discover-on-controlled /
evaluate-on-real separation.

Usage
-----
    import sst2_task
    sst2_task.register()          # idempotent; registers the "sst2" task
    # then run_grid with task "sst2" and task_to_lm_eval {sst2: sst2}

Run the standalone sanity check FIRST (verifies alignment + non-degeneracy on
the real tokenizer/model before spending grid GPU-hours):
    python sst2_task.py --model meta-llama/Llama-3.2-3B-Instruct --self-check
"""
from __future__ import annotations

import argparse
import logging
import random
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# --- generation vocabulary --------------------------------------------------- #
# Predicate-adjective carriers: one {adj} slot, grammatical for any adjective.
CARRIER_TEMPLATES: List[str] = [
    "Review: The movie was {adj}.",
    "Review: The film was {adj}.",
    "Review: The acting was {adj}.",
    "Review: The plot was {adj}.",
    "Review: The story was {adj}.",
    "Review: The ending was {adj}.",
    "Review: The dialogue was {adj}.",
    "Review: Overall, the movie was {adj}.",
]
PROMPT_SUFFIX = "\nSentiment (positive or negative):"

# Sentiment adjectives. Curated to be common, unambiguously polar, and usually
# single-token; the runtime tokenizer filter is the real guarantee.
POSITIVE_ADJ: List[str] = [
    "great", "good", "excellent", "wonderful", "fantastic", "brilliant",
    "amazing", "superb", "delightful", "terrific", "charming", "impressive",
    "enjoyable", "gripping", "beautiful", "funny", "clever", "touching",
    "satisfying", "memorable", "compelling", "captivating", "refreshing",
    "moving", "entertaining",
]
NEGATIVE_ADJ: List[str] = [
    "terrible", "awful", "bad", "boring", "dreadful", "horrible",
    "disappointing", "dull", "tedious", "forgettable", "weak", "bland",
    "lifeless", "painful", "annoying", "confusing", "predictable", "mediocre",
    "clumsy", "tiresome", "shallow", "laughable", "unbearable", "disjointed",
    "pointless",
]

# Answer label pairs, most-preferred first. The pair must tokenize to single,
# distinct tokens under the model; the template suffix names the first option.
LABEL_PAIRS: List[Tuple[str, str]] = [
    (" positive", " negative"),
    (" Positive", " Negative"),
    (" good", " bad"),
    (" yes", " no"),
    (" Yes", " No"),
]

TokLen = Callable[[str], int]   # text -> number of tokens (no BOS)


def _pick_label_pair(tok_len: TokLen) -> Tuple[str, str]:
    """First label pair whose BOTH members are single tokens (and distinct)."""
    for pos, neg in LABEL_PAIRS:
        if tok_len(pos) == 1 and tok_len(neg) == 1 and pos.strip() != neg.strip():
            return pos, neg
    raise RuntimeError(
        "No sentiment label pair tokenizes to single tokens under this model; "
        f"tried {LABEL_PAIRS}. Add a model-appropriate pair to LABEL_PAIRS."
    )


def _single_token_adjectives(adjs: List[str], tok_len: TokLen) -> List[str]:
    """Keep adjectives that are a single token in mid-sentence position.

    The adjective follows a space (``was {adj}``), so we test the space-prefixed
    form -- that is how the tokenizer actually sees it.
    """
    keep = [a for a in adjs if tok_len(" " + a) == 1]
    return keep


def build_contrastive_pairs(
    n: int,
    tok_len: TokLen,
    *,
    seed: int = 42,
    max_attempts_factor: int = 40,
) -> List[Dict[str, object]]:
    """Generate ``n`` token-aligned, label-flipping SST-2 contrastive pairs.

    Each returned example dict has the columns the (fixed) GLUE EAP dataloader
    consumes:
        prompt            -- clean prompt (positive OR negative carrier)
        answer            -- clean answer label (e.g. " positive")
        corrupted_prompt  -- minimal-edit counter-factual (opposite adjective)
        corrupted_answer  -- opposite answer label (e.g. " negative")
        label             -- 1 if clean is positive else 0 (SST-2 convention)

    Guarantees (all verified against ``tok_len``):
      * clean and corrupt differ by exactly one word (the adjective),
      * clean and corrupt tokenize to the SAME length (position alignment),
      * answer and corrupted_answer are distinct single tokens.

    Raises RuntimeError if it cannot assemble ``n`` valid pairs (e.g. the
    tokenizer splits every candidate) -- fail loud, never emit degenerate data.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    pos_label, neg_label = _pick_label_pair(tok_len)
    pos_adj = _single_token_adjectives(POSITIVE_ADJ, tok_len)
    neg_adj = _single_token_adjectives(NEGATIVE_ADJ, tok_len)
    if len(pos_adj) < 2 or len(neg_adj) < 2:
        raise RuntimeError(
            "Too few single-token sentiment adjectives under this model "
            f"(pos={len(pos_adj)}, neg={len(neg_adj)}). Extend POSITIVE_ADJ/"
            "NEGATIVE_ADJ with more single-token words for this tokenizer."
        )

    examples: List[Dict[str, object]] = []
    seen: set = set()
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 200)
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        carrier = rng.choice(CARRIER_TEMPLATES)
        p = rng.choice(pos_adj)
        q = rng.choice(neg_adj)
        clean_is_positive = rng.random() < 0.5

        pos_prompt = carrier.format(adj=p) + PROMPT_SUFFIX
        neg_prompt = carrier.format(adj=q) + PROMPT_SUFFIX
        # Position alignment: the two full prompts must be equal length. Only the
        # adjective differs, so this holds whenever the BPE merges do not spill
        # across the slot; we verify rather than assume.
        if tok_len(pos_prompt) != tok_len(neg_prompt):
            continue

        if clean_is_positive:
            prompt, answer = pos_prompt, pos_label
            corrupted_prompt, corrupted_answer = neg_prompt, neg_label
            label = 1
        else:
            prompt, answer = neg_prompt, neg_label
            corrupted_prompt, corrupted_answer = pos_prompt, pos_label
            label = 0

        key = (prompt, corrupted_prompt)
        if key in seen:
            continue
        seen.add(key)
        examples.append({
            "prompt": prompt,
            "answer": answer,
            "corrupted_prompt": corrupted_prompt,
            "corrupted_answer": corrupted_answer,
            "label": label,
        })

    if len(examples) < n:
        raise RuntimeError(
            f"Only assembled {len(examples)}/{n} aligned SST-2 pairs in "
            f"{attempts} attempts. The tokenizer may be splitting the adjectives "
            "unevenly; add more single-token candidates or lower num_examples."
        )
    return examples


# --- task spec --------------------------------------------------------------- #

def _make_sst2_spec():
    """A GLUE spec whose data is our controlled contrastive sentiment set.

    Subclasses the proven GLUETaskSpec (so tokenize / collate / metric_fn / EAP
    path are reused verbatim) and only replaces the data source: instead of
    loading raw GLUE sst2 + a paraphrase corruption, it emits authored
    label-flipping pairs. The (fixed) ``_build_eap_dataloader`` picks up the
    ``corrupted_prompt`` / ``corrupted_answer`` columns automatically.
    """
    from circuitkit.tasks.builtins.glue import GLUETaskSpec

    class SST2ContrastiveTaskSpec(GLUETaskSpec):
        """GLUE-SST2 surfaced as 'sst2' with controlled contrastive pairs."""

        def __init__(self):
            super().__init__("sst2")   # sets task_info/default_subtask to sst2
            self.name = "sst2"          # clean grid keys / result filenames

        def _get_or_load_glue_data(self, discovery_cfg, model):
            # Discovery uses `num_examples` (seed 42). The faithfulness pass runs
            # with an eval-shifted seed and sets `eval_num_examples`, so we honour
            # it here -> patchF is measured on a DIFFERENT, held-out contrastive
            # set than the one the circuit was discovered on (same discovery/eval
            # partition separation the rest of the audit uses).
            n = int(discovery_cfg.get("eval_num_examples")
                    or discovery_cfg.get("num_examples") or 128)
            seed = int(discovery_cfg.get("seed", 42))

            def tok_len(text: str) -> int:
                # tokens excluding BOS -- how EAP sees the string.
                return int(model.to_tokens(text, prepend_bos=False).shape[1])

            pairs = build_contrastive_pairs(n, tok_len, seed=seed)
            logger.info(
                "sst2 contrastive: built %d aligned pairs (single-token labels, "
                "one-adjective edit, equal clean/corrupt token length)", len(pairs)
            )
            return pairs

    return SST2ContrastiveTaskSpec()


def register() -> None:
    """Register the 'sst2' contrastive task (idempotent)."""
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import is_task_registered, register_task

    _bootstrap_builtin_tasks()
    if is_task_registered("sst2"):
        logger.info("task 'sst2' already registered")
        return
    register_task(_make_sst2_spec())
    logger.info("registered task 'sst2' (contrastive sentiment, binary)")


# --- standalone sanity check (run on the GPU box before the grid) ------------ #

def _self_check(model_name: str, n: int = 16, seed: int = 42) -> int:
    """Load the real model, build pairs, assert every constraint, and show that
    y_clean != y_corrupt on a forward pass. Returns a process exit code.
    """
    import circuitkit as ck
    import torch

    print(f"[sst2 self-check] loading {model_name} ...")
    model = ck.load_model(model_name, dtype="bfloat16", device="cuda")
    model.cfg.model_name = model_name

    def tok_len(text: str) -> int:
        return int(model.to_tokens(text, prepend_bos=False).shape[1])

    pairs = build_contrastive_pairs(n, tok_len, seed=seed)
    print(f"[sst2 self-check] built {len(pairs)} pairs; verifying constraints ...")

    for ex in pairs:
        clean, corrupt = ex["prompt"], ex["corrupted_prompt"]
        # exactly one word differs
        cw, qw = clean.split(), corrupt.split()
        assert len(cw) == len(qw), f"word-count mismatch:\n{clean}\n{corrupt}"
        diffs = [i for i, (a, b) in enumerate(zip(cw, qw)) if a != b]
        assert len(diffs) == 1, f"expected 1 differing word, got {len(diffs)}:\n{clean}\n{corrupt}"
        # equal token length (position alignment)
        assert tok_len(clean) == tok_len(corrupt), f"token-length mismatch:\n{clean}\n{corrupt}"
        # distinct single-token answers
        assert tok_len(ex["answer"]) == 1 and tok_len(ex["corrupted_answer"]) == 1
        assert ex["answer"].strip() != ex["corrupted_answer"].strip()
    print("[sst2 self-check] OK: one-word edit, equal length, single-token distinct answers.")

    # Forward pass: logit_diff(answer, corrupted_answer) at the last position.
    # Wrap prompts in the chat template exactly as discovery does, so the gap we
    # report previews the real discovery metric (chat models score raw task text
    # poorly). Falls back to raw prompts if the chat helpers are unavailable.
    try:
        from circuitkit.tasks._chat import resolve_chat_template, to_tokens, wrap_prompt
        apply = resolve_chat_template("auto", model)
    except Exception:  # noqa: BLE001
        resolve_chat_template = None
        apply = False

    def _tokens(prompt: str):
        if apply and resolve_chat_template is not None:
            return to_tokens(model, wrap_prompt(model, prompt, apply=apply), templated=apply)
        return model.to_tokens(prompt)

    def logit_diff(prompt: str, ans: str, ans_neg: str) -> float:
        toks = _tokens(prompt)
        with torch.no_grad():
            logits = model(toks)[0, -1]
        pos_id = model.to_tokens(ans, prepend_bos=False)[0, -1].item()
        neg_id = model.to_tokens(ans_neg, prepend_bos=False)[0, -1].item()
        return float(logits[pos_id] - logits[neg_id])

    y_clean, y_corrupt = [], []
    for ex in pairs:
        y_clean.append(logit_diff(ex["prompt"], ex["answer"], ex["corrupted_answer"]))
        # same metric tokens, but on the corrupt prompt (answer flips)
        y_corrupt.append(logit_diff(ex["corrupted_prompt"], ex["answer"], ex["corrupted_answer"]))
    mc = sum(y_clean) / len(y_clean)
    mq = sum(y_corrupt) / len(y_corrupt)
    print(f"[sst2 self-check] mean logit_diff  clean={mc:+.3f}  corrupt={mq:+.3f}  "
          f"gap={mc - mq:+.3f}")
    if abs(mc - mq) < 0.5:
        print("[sst2 self-check] FAIL: clean/corrupt gap too small -- metric would be "
              "near-degenerate. Inspect the template/labels.")
        return 1
    print("[sst2 self-check] PASS: clean and corrupt are well separated "
          "(patch-faithfulness denominator is non-degenerate).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--self-check", action="store_true",
                    help="load the model and verify alignment + non-degeneracy")
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.self_check:
        raise SystemExit(_self_check(args.model, n=args.num_examples, seed=args.seed))

    register()
    from circuitkit.tasks.registry import is_task_registered
    print("sst2 registered:", is_task_registered("sst2"))
