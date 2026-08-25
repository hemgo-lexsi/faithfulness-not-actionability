#!/usr/bin/env python3
r"""
analyze.py -- compute every statistic reported in
"Faithfulness Is Not Actionability: Extrinsic Validation of Circuit Selectors
Under Structured Pruning".

This is the single source of truth for the paper's numbers. It reads the
per-run master CSV(s) emitted by the drivers (or the shipped seed-42 reference
values) and reproduces, with no GPU required:

  * Table 1  -- per-cell Spearman rho(patchF, accRet) + exact/MC-permutation p
  * Table 4  -- leave-one-family-out (LOFO) on Llama.BoolQ
  * A multi-seed check (no dedicated paper table -- replication of the two
    significant cells has not been run yet; see README "seed question")
  * The coherence-confound analysis (partial Spearman | log PPL; restricted subset)
  * The |patchF - 1| vs accRet reframing (n=11, OBS included)
  * Pooled selector-mean correlation over the balanced 2x2 grid
  * Cross-swap asymmetry (mean Delta-acc + paired t-test)
  * Multiplicity: binomial joint probability + Bonferroni-Holm
  * ablF (mean-ablation faithfulness) sanity correlations

Every number is computed from data, never hard-coded. Run:

    python analyze.py --results-dir ../results/reference     # reproduce paper
    python analyze.py --results-dir ../results               # your own runs
    python analyze.py --self-test                            # assert paper values

Inputs it looks for under --results-dir:
    master_seed{SEED}.csv     columns: model,task,seed,selector,family,
                                       is_control,patchF,accRet,ablF,ppl
    crossswap_seed{SEED}.csv  columns: pair,high_patchf_selector,
                                       low_patchf_selector,low_patchf_harm,
                                       high_patchf_harm,random_harm
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 20260824          # fixes Monte-Carlo permutation + bootstrap draws
MC_RESAMPLES = 500_000       # used when n! is too large to enumerate exactly
BOOT_RESAMPLES = 20_000
EXACT_MAX_N = 9              # n! <= 362_880 -> enumerate exactly


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
def _centered_ranks(v: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v)
    return r - r.mean()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def spearman_perm_p(x: np.ndarray, y: np.ndarray,
                    rng: Optional[np.random.Generator] = None) -> tuple[float, float, str]:
    """Two-sided permutation p-value for Spearman rho.

    Exact enumeration for n <= EXACT_MAX_N, else fixed-seed Monte Carlo.
    Returns (rho, p, mode). Vectorised: permute one centred rank vector and
    correlate against the fixed other (denominator is constant under
    permutation, so only the dot product varies).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xc = _centered_ranks(x)
    yc = _centered_ranks(y)
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    r_obs = float(xc @ yc / denom)

    if n <= EXACT_MAX_N:
        perms = np.array(list(itertools.permutations(yc)))   # (n!, n)
        r_all = (perms @ xc) / denom
        p = float(np.mean(np.abs(r_all) >= abs(r_obs) - 1e-12))
        return r_obs, p, f"exact({len(perms)})"

    rng = rng or np.random.default_rng(RNG_SEED)
    # vectorised MC: argsort of uniforms gives independent permutations
    idx = np.argsort(rng.random((MC_RESAMPLES, n)), axis=1)
    r_all = (yc[idx] @ xc) / denom
    # +1 numerator/denominator = unbiased MC p-value
    p = float((np.sum(np.abs(r_all) >= abs(r_obs) - 1e-12) + 1) / (MC_RESAMPLES + 1))
    return r_obs, p, f"mc({MC_RESAMPLES})"


def spearman_boot_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05,
                     rng: Optional[np.random.Generator] = None) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho (paired resampling)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rng = rng or np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(BOOT_RESAMPLES, n))
    rhos = np.empty(BOOT_RESAMPLES)
    for b in range(BOOT_RESAMPLES):
        xi, yi = x[idx[b]], y[idx[b]]
        if np.all(xi == xi[0]) or np.all(yi == yi[0]):
            rhos[b] = 0.0
        else:
            rhos[b] = spearman(xi, yi)
    lo, hi = np.nanpercentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def partial_spearman(a, b, c) -> tuple[float, float]:
    """Spearman partial correlation of a,b controlling for c: rank-transform
    all three, residualise ranks of a and b on ranks of c, correlate residuals.
    """
    ra, rb, rc = (stats.rankdata(v).astype(float) for v in (a, b, c))

    def resid(y, x):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef

    era, erb = resid(ra, rc), resid(rb, rc)
    r, p = stats.pearsonr(era, erb)
    return float(r), float(p)


def sig_stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_master(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "master_seed*.csv")))
    if not files:
        sys.exit(f"[analyze] no master_seed*.csv under {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ("patchF", "accRet", "ablF", "ppl"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[analyze] loaded {len(df)} rows from {len(files)} file(s): "
          f"{[os.path.basename(f) for f in files]}")
    return df


def cell(df: pd.DataFrame, model: str, task: str, seed: int,
         include_obs: bool = False) -> pd.DataFrame:
    m = (df.model == model) & (df.task == task) & (df.seed == seed)
    sub = df[m].copy()
    if not include_obs:
        sub = sub[sub.family != "obs"]
    return sub.dropna(subset=["patchF", "accRet"])


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def hdr(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def report_master_grid(df, seed=42):
    hdr(f"TABLE 1 (top) -- per model.task cell, Spearman rho(patchF, accRet), seed {seed}")
    print(f"{'model.task':22} {'n':>3} {'rho':>8} {'p':>9} {'95% CI':>20}  verdict")
    cells = [("llama", "boolq"), ("llama", "sst2"), ("llama", "winogrande"),
             ("gemma", "boolq"), ("gemma", "winogrande")]
    n_sig = 0
    pvals = {}
    for model, task in cells:
        c = cell(df, model, task, seed)
        if len(c) < 3:
            continue
        rho, p, mode = spearman_perm_p(c.patchF.values, c.accRet.values)
        lo, hi = spearman_boot_ci(c.patchF.values, c.accRet.values)
        if rho <= -0.40 and p < 0.10:
            verdict = "anti-predicts"
        elif rho >= 0.40 and p < 0.10:
            verdict = "predicts"
        else:
            verdict = "no signal"
        if p < 0.05:
            n_sig += 1
        pvals[f"{model}.{task}"] = p
        print(f"{model+'.'+task:22} {len(c):>3} {rho:+8.3f} {p:9.4f} "
              f"[{lo:+.2f}, {hi:+.2f}]{'':6}  {verdict}{sig_stars(p)}  ({mode})")
    return pvals, n_sig


def report_patchf_from_one(df, seed=42):
    hdr("Reframing: distance-from-perfect |patchF - 1| vs accRet on Llama.BoolQ (OBS incl., n=11)")
    c = cell(df, "llama", "boolq", seed, include_obs=True)
    d = (c.patchF - 1.0).abs().values
    rho, p, mode = spearman_perm_p(d, c.accRet.values)
    print(f"  rho(|patchF-1|, accRet) = {rho:+.3f}, p = {p:.4f}, n = {len(c)}  ({mode})")
    print("  (paper: rho = +0.77, p = 0.007, n = 11 -- the nearest-to-perfect compress worst)")


def report_lofo(df, seed=42):
    hdr("TABLE 4 -- Leave-one-family-out on Llama.BoolQ (30% sparsity)")
    c = cell(df, "llama", "boolq", seed)
    print(f"{'family dropped':26} {'n':>3} {'rho':>8} {'p':>9}")
    for fam in ["weight-saliency", "granularity-aware", "control", "circuit-discovery"]:
        sub = c[c.family != fam]
        rho, p, _ = spearman_perm_p(sub.patchF.values, sub.accRet.values)
        print(f"{'drop '+fam:26} {len(sub):>3} {rho:+8.3f} {p:9.4f}{sig_stars(p)}")
    only = c[c.family == "circuit-discovery"]
    rho, p, _ = spearman_perm_p(only.patchF.values, only.accRet.values)
    print(f"{'circuit-discovery only':26} {len(only):>3} {rho:+8.3f} {p:9.4f}{sig_stars(p)}")


def report_confound(df, seed=42):
    hdr("Coherence confound on Llama.BoolQ -- does the negative rho survive PPL?")
    c = cell(df, "llama", "boolq", seed).dropna(subset=["ppl"])
    if c.empty:
        print("  [skip] no PPL column populated for this cell")
        return
    logppl = np.log10(c.ppl.values)
    print(f"  rho(patchF, log10 PPL) = {spearman(c.patchF.values, logppl):+.3f}"
          "   (entanglement: faithful selectors are the incoherent ones)")
    print(f"  rho(accRet, log10 PPL) = {spearman(c.accRet.values, logppl):+.3f}")
    pr, pp = partial_spearman(c.patchF.values, c.accRet.values, logppl)
    print(f"  partial Spearman(patchF, accRet | log PPL) = {pr:+.3f}, p = {pp:.3f}, n = {len(c)}")
    coh = c[c.ppl < 500]
    rho, p, mode = spearman_perm_p(coh.patchF.values, coh.accRet.values)
    print(f"  restricted to coherent selectors (PPL<500): rho = {rho:+.3f}, "
          f"p = {p:.3f}, n = {len(coh)}  ({mode})")
    print(f"    kept: {sorted(coh.selector.tolist())}")


def report_ablf(df, seed=42):
    hdr("ablF (mean-ablation faithfulness) sanity on Llama.BoolQ")
    c = cell(df, "llama", "boolq", seed).dropna(subset=["ablF"])
    if c.empty:
        print("  [skip] no ablF populated")
        return
    r1 = spearman(c.ablF.values, c.patchF.values)
    r2 = spearman(c.ablF.values, c.accRet.values)
    print(f"  rho(ablF, patchF)  = {r1:+.3f}   rho(ablF, accRet) = {r2:+.3f}")
    print("  (paper: ablF neither tracks patchF nor predicts accRet)")


def report_pooled(df, seed=42):
    hdr("Pooled selector-mean correlation over the balanced 2x2 grid "
        "(Llama/Gemma x BoolQ/WinoGrande)")
    cells = [("llama", "boolq"), ("llama", "winogrande"),
             ("gemma", "boolq"), ("gemma", "winogrande")]
    frames = [cell(df, m, t, seed)[["selector", "patchF", "accRet"]] for m, t in cells]
    if any(len(f) == 0 for f in frames):
        print("  [skip] need all four balanced-grid cells")
        return
    allc = pd.concat(frames)
    g = allc.groupby("selector").mean(numeric_only=True)
    rho, p, mode = spearman_perm_p(g.patchF.values, g.accRet.values)
    print(f"  rho = {rho:+.3f}, p = {p:.3f}, n = {len(g)}  ({mode})   (paper: -0.64, p=0.054)")


def report_crossswap(results_dir, seed=42):
    hdr("Cross-swap causal test -- accuracy-harm asymmetry (paired)")
    path = os.path.join(results_dir, f"crossswap_seed{seed}.csv")
    if not os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} not found")
        return
    cs = pd.read_csv(path)
    low, high, rnd = cs.low_patchf_harm.values, cs.high_patchf_harm.values, cs.random_harm.values
    diff = high - low
    t, p = stats.ttest_rel(high, low)
    print(f"  low-patchF discards : mean harm = {low.mean():+.3f}  (safe)")
    print(f"  high-patchF discards: mean harm = {high.mean():+.3f}")
    print(f"  random baseline     : mean harm = {rnd.mean():+.3f}")
    print(f"  mean asymmetry Delta-acc (high - low) = {diff.mean():+.3f}")
    print(f"  paired t({len(cs)-1}) = {t:.2f}, p = {p:.3f}   (paper: +0.343, t(2)=12.83, p=0.006)")


def report_multiplicity(pvals: dict, n_sig: int):
    hdr("Multiplicity across the five model.task cells")
    m = len(pvals)
    p_ge2 = float(1 - stats.binom.cdf(1, m, 0.05))
    print(f"  {n_sig} of {m} cells significant at alpha=0.05")
    print(f"  P(>= 2 of {m} significant under global null) = {p_ge2:.4f} (binomial)")
    print("  Bonferroni-Holm (step-down):")
    for i, (k, v) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        adj = min((m - i) * v, 1.0)
        print(f"    {k:22} p={v:.3f}  x{m-i} -> {adj:.3f} {sig_stars(adj)}")


def report_multiseed(df, results_dir):
    hdr("Multi-seed replication of the two significant cells "
        "(no dedicated paper table -- not yet run)")
    seeds = sorted(df.seed.unique())
    print(f"  seeds present: {seeds}")
    print(f"{'cell':16} " + " ".join(f"seed{s:>5}" for s in seeds) + f"{'mean':>9}")
    for model, task in [("llama", "boolq"), ("llama", "sst2")]:
        row, vals = [], []
        for s in seeds:
            c = cell(df, model, task, s)
            if len(c) >= 3:
                rho, p, _ = spearman_perm_p(c.patchF.values, c.accRet.values)
                row.append(f"{rho:+.2f}{sig_stars(p)}")
                vals.append(rho)
            else:
                row.append("  --  ")
        mean = f"{np.mean(vals):+.2f}" if vals else "  --  "
        print(f"{model+'.'+task:16} " + " ".join(f"{r:>9}" for r in row) + f"{mean:>9}")
    if seeds == [42]:
        print("  (only seed 42 present -- run drivers with --seeds 42 123 456 to fill)")


# ---------------------------------------------------------------------------
# Self-test: assert we reproduce the published values
# ---------------------------------------------------------------------------
def self_test(df, results_dir):
    hdr("SELF-TEST -- asserting published seed-42 values")
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:+.3f}, want {want:+.3f} (+-{tol})")

    b = cell(df, "llama", "boolq", 42)
    check("BoolQ rho", spearman(b.patchF.values, b.accRet.values), -0.784, 0.02)
    s = cell(df, "llama", "sst2", 42)
    check("SST-2 rho", spearman(s.patchF.values, s.accRet.values), +0.772, 0.02)
    w = cell(df, "llama", "winogrande", 42)
    check("WinoGrande rho", spearman(w.patchF.values, w.accRet.values), +0.333, 0.02)
    gb = cell(df, "gemma", "boolq", 42)
    # Gemma.BoolQ: the paper reports -0.32 from full-precision scores. The
    # reference CSV stores them at 3 dp, which creates accRet ties the real run
    # does not have, so recomputation here lands at -0.33. Tolerance covers it.
    check("Gemma.BoolQ rho", spearman(gb.patchF.values, gb.accRet.values), -0.32, 0.03)

    # Leave-CD-T-out sensitivity: CD-T is the one selector whose implementation
    # differs between the headline BoolQ cell (full beta/gamma) and the rest
    # (gradient-hook approximation), so the cross-task sign comparison is
    # reported without it as well.
    bx = b[b.selector != "cdt"]
    check("BoolQ rho, CD-T dropped",
          spearman(bx.patchF.values, bx.accRet.values), -0.728, 0.02)
    sx = s[s.selector != "cdt"]
    check("SST-2 rho, CD-T dropped",
          spearman(sx.patchF.values, sx.accRet.values), +0.819, 0.02)

    bo = cell(df, "llama", "boolq", 42, include_obs=True)
    d = (bo.patchF - 1.0).abs().values
    check("|patchF-1| rho (n=11)", spearman(d, bo.accRet.values), +0.77, 0.03)

    lp = np.log10(b.ppl.values)
    pr, _ = partial_spearman(b.patchF.values, b.accRet.values, lp)
    check("partial rho | logPPL", pr, -0.66, 0.03)

    cs = pd.read_csv(os.path.join(results_dir, "crossswap_seed42.csv"))
    t, _ = stats.ttest_rel(cs.high_patchf_harm.values, cs.low_patchf_harm.values)
    check("cross-swap t(2)", t, 12.83, 0.5)
    check("cross-swap asymmetry",
          float((cs.high_patchf_harm - cs.low_patchf_harm).mean()), 0.343, 0.01)

    print("\n  " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir",
                    default=os.path.join(here, "..", "results", "reference"))
    ap.add_argument("--seed", type=int, default=42,
                    help="Which seed's per-cell tables to print (default 42).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    df = load_master(args.results_dir)

    pvals, n_sig = report_master_grid(df, args.seed)
    report_patchf_from_one(df, args.seed)
    report_lofo(df, args.seed)
    report_confound(df, args.seed)
    report_ablf(df, args.seed)
    report_pooled(df, args.seed)
    report_crossswap(args.results_dir, args.seed)
    report_multiplicity(pvals, n_sig)
    report_multiseed(df, args.results_dir)

    if args.self_test:
        if not self_test(df, args.results_dir):
            sys.exit(1)


if __name__ == "__main__":
    main()
