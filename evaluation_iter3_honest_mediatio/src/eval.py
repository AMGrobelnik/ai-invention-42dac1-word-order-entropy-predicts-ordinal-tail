#!/usr/bin/env python3
"""Honest Mediation Reanalysis, Alternative Causal Models, and Unexplained Variance Decomposition.

Evaluates exp_id1 (GEV tail-constraint analysis) and exp_id4 (data quality validation)
for mediation honesty, alternative causal models, confound robustness, and unexplained
variance decomposition.
"""

import gc
import json
import math
import os
import resource
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────

def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.7, 20) * 1e9)  # 70% of container, max 20 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET / 1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
NAN_SENTINEL = -999.0
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 42
FEAT_COMPLETENESS_THRESHOLD = 0.5

# ── Paths ────────────────────────────────────────────────────────────────────
_DEP_BASE = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bto/3_invention_loop/iter_2/gen_art")
EXP1_PATH = _DEP_BASE / "exp_id1_it2__opus" / "full_method_out.json"
EXP4_PATH = _DEP_BASE / "exp_id4_it2__opus" / "full_method_out.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    """Convert value to a safe JSON-compatible float (no NaN / Inf)."""
    if val is None:
        return NAN_SENTINEL
    try:
        f = float(val)
    except (TypeError, ValueError):
        return NAN_SENTINEL
    if math.isnan(f) or math.isinf(f):
        return NAN_SENTINEL
    return f


# ═════════════════════════════════════════════════════════════════════════════
# STEP 0  Data Loading & Merging
# ═════════════════════════════════════════════════════════════════════════════

def load_exp_data():
    """Load exp_id1 and exp_id4 data, build merged DataFrame."""
    logger.info("STEP 0: Loading and merging data")

    # ── exp_id1 ──
    logger.info(f"  Loading exp_id1 from {EXP1_PATH}")
    exp1 = json.loads(EXP1_PATH.read_text())
    meta1 = exp1["metadata"]
    examples1 = exp1["datasets"][0]["examples"]
    logger.info(f"  exp_id1: {len(examples1)} treebanks")

    rows1 = []
    for ex in examples1:
        out = json.loads(ex["output"])
        rows1.append({
            "treebank_id": ex["input"],
            "xi_raw": out["xi_raw"],
            "xi_raw_se": out["xi_raw_se"],
            "xi_norm": out["xi_norm"],
            "n_qualifying_bins": out["n_qualifying_bins"],
            "family": out.get("family") or ex.get("metadata_family"),
            "morph_richness": ex["metadata_morph_richness"],
            "head_direction_ratio": ex["metadata_head_direction_ratio"],
            "word_order_entropy": ex["metadata_word_order_entropy"],
            "mean_dd_all": ex["metadata_mean_dd_all"],
            "modality": ex["metadata_modality"],
            "genre": ex["metadata_genre"],
            "language": ex["metadata_language"],
            "iso_code": ex["metadata_iso_code"],
            "n_bins": ex["metadata_n_bins"],
        })
    df1 = pd.DataFrame(rows1)

    # ── exp_id4 ──
    logger.info(f"  Loading exp_id4 from {EXP4_PATH}")
    exp4 = json.loads(EXP4_PATH.read_text())
    examples4 = exp4["datasets"][0]["examples"]
    logger.info(f"  exp_id4: {len(examples4)} treebanks")

    rows4 = []
    for ex in examples4:
        out = json.loads(ex["output"])
        val = json.loads(ex["predict_comprehensive_validation"])
        rows4.append({
            "treebank_id": ex["input"],
            "feat_completeness": out["feat_completeness"],
            "n_sentences_total": out["n_sentences_total"],
            "n_binned": out["n_binned"],
            "overall_quality": val.get("overall_quality", "UNKNOWN"),
        })
    df4 = pd.DataFrame(rows4)

    # ── Left join ──
    df = df1.merge(
        df4[["treebank_id", "feat_completeness", "n_sentences_total", "n_binned", "overall_quality"]],
        on="treebank_id",
        how="left",
    )
    logger.info(f"  Merged DataFrame: {len(df)} treebanks, {df['feat_completeness'].notna().sum()} with quality data")

    # ── Regression subset (match exp_id1: n_qualifying_bins >= 2, non-null family/predictors) ──
    df_reg = df[df.n_qualifying_bins >= 2].dropna(
        subset=["family", "morph_richness", "head_direction_ratio", "word_order_entropy"]
    ).copy()
    logger.info(f"  Regression subset: {len(df_reg)} treebanks (expect ~172)")

    # ── z-score predictors based on regression subset statistics ──
    z_stats = {}
    for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
        mu = df_reg[col].mean()
        sd = df_reg[col].std()
        z_stats[col] = (mu, sd)
        df[f"{col}_z"] = (df[col] - mu) / sd
        logger.debug(f"  {col}: mean={mu:.4f}, std={sd:.4f}")

    # Recompute df_reg with z columns
    df_reg = df[df.n_qualifying_bins >= 2].dropna(
        subset=["family", "morph_richness", "head_direction_ratio", "word_order_entropy"]
    ).copy()

    del exp1, exp4, examples1, examples4, rows1, rows4, df1, df4
    gc.collect()
    return df, df_reg, meta1


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1  Honest Mediation Reporting
# ═════════════════════════════════════════════════════════════════════════════

def step1_honest_mediation(meta1: dict) -> dict:
    """Extract mediation results and flag the suppression pattern."""
    logger.info("STEP 1: Honest Mediation Reporting")
    med = meta1["mediation"]

    indirect = med["indirect_effect_mean"]
    direct   = med["direct_effect_mean"]
    total    = med["total_effect_mean"]
    prop_med = med["proportion_mediated"]

    opposing_paths       = (indirect > 0) and (direct < 0)
    total_near_zero      = abs(total) < 0.01
    prop_med_nonsensical = abs(prop_med) > 2.0

    explanation = (
        f"Proportion mediated = indirect/total = {indirect:.3f}/{total:.3f} = {prop_med:.2f}. "
        f"This is nonsensical because the total effect is near zero due to opposing indirect "
        f"(+{indirect:.3f}) and direct ({direct:.3f}) paths (a classical suppression pattern). "
        f"The mediation label '{med['interpretation']}' is technically correct (indirect significant, "
        f"direct not) but misleading: it describes opposing statistical paths rather than a clean "
        f"causal mechanism."
    )

    indistinguishable_models = [
        "morph_richness -> word_order_entropy -> xi (forward mediation)",
        "word_order_entropy -> morph_richness -> xi (reverse mediation)",
        "unobserved_confound -> {morph_richness, word_order_entropy} -> xi (common cause)",
    ]

    logger.info(f"  Opposing paths: {opposing_paths}")
    logger.info(f"  Total near zero: {total_near_zero}  (|total|={abs(total):.4f})")
    logger.info(f"  Proportion mediated nonsensical: {prop_med_nonsensical}  ({prop_med:.2f})")

    return {
        "total": total,
        "indirect": indirect,
        "indirect_ci": med["indirect_effect_ci"],
        "direct": direct,
        "direct_ci": med["direct_effect_ci"],
        "proportion_mediated_raw": prop_med,
        "opposing_paths": opposing_paths,
        "total_near_zero": total_near_zero,
        "prop_med_nonsensical": prop_med_nonsensical,
        "explanation": explanation,
        "indistinguishable_models": indistinguishable_models,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2  Alternative Causal Models
# ═════════════════════════════════════════════════════════════════════════════

def _preacher_hayes_bootstrap(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Preacher-Hayes mediation bootstrap.  X -> M -> Y."""
    rng = np.random.RandomState(seed)
    n = len(X)

    # ── observed coefficients via OLS ──
    ones = np.ones(n)
    XC = np.column_stack([ones, X])
    beta_a = np.linalg.lstsq(XC, M, rcond=None)[0]
    a_obs = beta_a[1]

    XMC = np.column_stack([ones, X, M])
    beta_bc = np.linalg.lstsq(XMC, Y, rcond=None)[0]
    c_prime_obs = beta_bc[1]   # direct
    b_obs       = beta_bc[2]   # M -> Y | X

    indirect_obs = a_obs * b_obs
    total_obs    = c_prime_obs + indirect_obs

    # ── bootstrap ──
    boot_a = np.empty(n_bootstrap)
    boot_b = np.empty(n_bootstrap)
    boot_indirect = np.empty(n_bootstrap)
    boot_direct   = np.empty(n_bootstrap)
    boot_total    = np.empty(n_bootstrap)
    n_valid = 0

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        ones_b = np.ones(len(idx))
        try:
            ba = np.linalg.lstsq(np.column_stack([ones_b, Xb]), Mb, rcond=None)[0]
            bbc = np.linalg.lstsq(np.column_stack([ones_b, Xb, Mb]), Yb, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        boot_a[n_valid] = ba[1]
        boot_b[n_valid] = bbc[2]
        boot_indirect[n_valid] = ba[1] * bbc[2]
        boot_direct[n_valid]   = bbc[1]
        boot_total[n_valid]    = bbc[1] + ba[1] * bbc[2]
        n_valid += 1

    # trim
    boot_a        = boot_a[:n_valid]
    boot_b        = boot_b[:n_valid]
    boot_indirect = boot_indirect[:n_valid]
    boot_direct   = boot_direct[:n_valid]
    boot_total    = boot_total[:n_valid]

    def _ci(arr):
        return np.percentile(arr, [2.5, 97.5]).tolist()

    def _sig(ci):
        return bool(ci[0] > 0 or ci[1] < 0)

    ind_ci  = _ci(boot_indirect)
    dir_ci  = _ci(boot_direct)
    a_ci    = _ci(boot_a)
    b_ci    = _ci(boot_b)
    tot_ci  = _ci(boot_total)

    return {
        "a": float(a_obs), "a_ci": a_ci, "a_significant": _sig(a_ci),
        "b": float(b_obs), "b_ci": b_ci, "b_significant": _sig(b_ci),
        "indirect": float(indirect_obs), "indirect_mean": float(np.mean(boot_indirect)),
        "indirect_ci": ind_ci, "indirect_significant": _sig(ind_ci),
        "direct": float(c_prime_obs), "direct_mean": float(np.mean(boot_direct)),
        "direct_ci": dir_ci, "direct_significant": _sig(dir_ci),
        "total": float(total_obs), "total_mean": float(np.mean(boot_total)),
        "total_ci": tot_ci,
        "n_valid_bootstraps": n_valid,
    }


def step2_alternative_models(df_reg: pd.DataFrame) -> dict:
    """Run reverse mediation, entropy-only OLS, entropy-only mixed-effects."""
    logger.info("STEP 2: Alternative Causal Models")

    X_morph   = df_reg["morph_richness_z"].values.astype(np.float64)
    X_entropy = df_reg["word_order_entropy_z"].values.astype(np.float64)
    X_head    = df_reg["head_direction_ratio_z"].values.astype(np.float64)
    Y         = df_reg["xi_raw"].values.astype(np.float64)

    # ── 2a: Forward mediation (morph -> entropy -> xi) ──
    logger.info("  2a: Forward Mediation (morph -> entropy -> xi)")
    forward_med = _preacher_hayes_bootstrap(X_morph, X_entropy, Y)
    logger.info(f"      indirect={forward_med['indirect']:.4f}  CI={forward_med['indirect_ci']}  sig={forward_med['indirect_significant']}")
    logger.info(f"      direct={forward_med['direct']:.4f}  CI={forward_med['direct_ci']}  sig={forward_med['direct_significant']}")
    logger.info(f"      a (morph->entropy)={forward_med['a']:.4f}  CI={forward_med['a_ci']}")

    # ── 2a: Reverse mediation (entropy -> morph -> xi) ──
    logger.info("  2a: Reverse Mediation (entropy -> morph -> xi)")
    reverse_med = _preacher_hayes_bootstrap(X_entropy, X_morph, Y)
    logger.info(f"      indirect={reverse_med['indirect']:.4f}  CI={reverse_med['indirect_ci']}  sig={reverse_med['indirect_significant']}")
    logger.info(f"      direct={reverse_med['direct']:.4f}  CI={reverse_med['direct_ci']}  sig={reverse_med['direct_significant']}")

    # If BOTH forward and reverse indirect are significant, causal ordering is ambiguous
    reverse_vs_forward = not (reverse_med["indirect_significant"] and forward_med["indirect_significant"])
    logger.info(f"  Distinguishable: {reverse_vs_forward}")

    # ── 2b: Entropy-only OLS ──
    logger.info("  2b: Entropy-Only OLS")
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    X_ent_c = sm.add_constant(X_entropy)
    ols_ent = OLS(Y, X_ent_c).fit()
    entropy_only_beta = float(ols_ent.params[1])
    entropy_only_r2   = float(ols_ent.rsquared)
    entropy_only_p    = float(ols_ent.pvalues[1])
    logger.info(f"      beta={entropy_only_beta:.4f}  R2={entropy_only_r2:.4f}  p={entropy_only_p:.6f}")

    # Full 3-predictor OLS for fair comparison
    X_full_c = sm.add_constant(np.column_stack([X_morph, X_head, X_entropy]))
    ols_full = OLS(Y, X_full_c).fit()
    full_ols_r2 = float(ols_full.rsquared)
    r2_increment = full_ols_r2 - entropy_only_r2
    logger.info(f"      Full OLS R2={full_ols_r2:.4f}  increment={r2_increment:.4f}")

    # ── 2c: Entropy-only mixed-effects ──
    logger.info("  2c: Entropy-Only Mixed-Effects")
    entropy_mixed = _fit_entropy_mixed(df_reg, Y)

    # Entropy sufficiency verdict
    if r2_increment < 0.01:
        entropy_verdict = "entropy_alone_sufficient"
    elif r2_increment < 0.05:
        entropy_verdict = "entropy_primary_marginal_gain_from_others"
    else:
        entropy_verdict = "other_predictors_add_meaningful_variance"
    logger.info(f"  Entropy verdict: {entropy_verdict}")

    return {
        "forward_med": forward_med,
        "reverse_med": reverse_med,
        "reverse_vs_forward_distinguishable": reverse_vs_forward,
        "entropy_only_beta": entropy_only_beta,
        "entropy_only_r2": entropy_only_r2,
        "entropy_only_p": entropy_only_p,
        "full_ols_r2": full_ols_r2,
        "r2_increment_full_vs_entropy_only": r2_increment,
        "entropy_mixed": entropy_mixed,
        "entropy_verdict": entropy_verdict,
    }


def _fit_entropy_mixed(df_reg: pd.DataFrame, Y: np.ndarray) -> dict:
    """Fit entropy-only mixed-effects model; fall back to OLS+dummies on failure."""
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
        import statsmodels.api as sm

        tmp = df_reg[["xi_raw", "word_order_entropy_z", "family"]].copy()
        mdl = MixedLM.from_formula("xi_raw ~ word_order_entropy_z", groups="family", data=tmp)
        res = mdl.fit(reml=True)
        beta  = float(res.fe_params["word_order_entropy_z"])
        pval  = float(res.pvalues["word_order_entropy_z"])
        resid = res.resid
        pr2   = float(1.0 - np.var(resid) / np.var(Y))
        logger.info(f"      MixedLM: beta={beta:.4f}  pseudo-R2={pr2:.4f}  p={pval:.6f}")
        return {"beta": beta, "p": pval, "pseudo_r2": pr2, "converged": True, "method": "MixedLM"}

    except Exception as e:
        logger.warning(f"      MixedLM failed ({e}); falling back to OLS + family dummies")
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS

        tmp = df_reg[["xi_raw", "word_order_entropy_z", "family"]].copy()
        dummies = pd.get_dummies(tmp["family"], prefix="fam", drop_first=True)
        X_fb = sm.add_constant(pd.concat([tmp[["word_order_entropy_z"]], dummies], axis=1))
        fit = OLS(tmp["xi_raw"].values, X_fb.values.astype(float)).fit()
        beta = float(fit.params[1])
        pval = float(fit.pvalues[1])
        pr2  = float(fit.rsquared)
        logger.info(f"      Fallback OLS: beta={beta:.4f}  R2={pr2:.4f}  p={pval:.6f}")
        return {"beta": beta, "p": pval, "pseudo_r2": pr2, "converged": False, "method": "OLS_family_dummies"}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3  Confound-Restricted Mediation
# ═════════════════════════════════════════════════════════════════════════════

def step3_confound_restricted(df: pd.DataFrame, df_reg: pd.DataFrame, full_indirect: float) -> dict:
    """Mediation on the feat_completeness > threshold subset."""
    logger.info("STEP 3: Confound-Restricted Mediation")

    threshold = FEAT_COMPLETENESS_THRESHOLD
    df_hi = df[(df.feat_completeness > threshold)].copy()
    df_restricted = df_hi[df_hi.n_qualifying_bins >= 2].dropna(
        subset=["family", "morph_richness", "word_order_entropy"]
    ).copy()
    n_r = len(df_restricted)
    n_fam = df_restricted["family"].nunique() if n_r > 0 else 0
    logger.info(f"  Restricted subset: {n_r} treebanks, {n_fam} families (threshold={threshold})")

    result: dict = {"n_treebanks": n_r, "n_families": n_fam, "threshold": threshold}

    # Lower threshold if too few
    if n_r < 30:
        threshold = 0.3
        logger.warning(f"  Too few ({n_r}). Retrying with threshold={threshold}")
        df_hi = df[(df.feat_completeness > threshold)].copy()
        df_restricted = df_hi[df_hi.n_qualifying_bins >= 2].dropna(
            subset=["family", "morph_richness", "word_order_entropy"]
        ).copy()
        n_r = len(df_restricted)
        n_fam = df_restricted["family"].nunique() if n_r > 0 else 0
        result.update({"n_treebanks": n_r, "n_families": n_fam, "threshold": threshold})
        logger.info(f"  Adjusted: {n_r} treebanks, {n_fam} families")

    if n_r < 30:
        logger.warning("  Still < 30. Confound analysis infeasible.")
        result.update({
            "infeasible": True, "indirect": NAN_SENTINEL,
            "indirect_ci": [NAN_SENTINEL, NAN_SENTINEL], "indirect_significant": False,
            "direct": NAN_SENTINEL, "direct_ci": [NAN_SENTINEL, NAN_SENTINEL],
            "direct_significant": False, "total": NAN_SENTINEL,
            "opposing_persists": False, "verdict": "infeasible_insufficient_data",
            "restricted_vs_full_indirect_diff": NAN_SENTINEL,
        })
        return result

    # Re-standardize within restricted subset
    for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
        mu = df_restricted[col].mean()
        sd = df_restricted[col].std()
        df_restricted[f"{col}_zr"] = (df_restricted[col] - mu) / sd if sd > 0 else 0.0

    X = df_restricted["morph_richness_zr"].values.astype(np.float64)
    M = df_restricted["word_order_entropy_zr"].values.astype(np.float64)
    Y = df_restricted["xi_raw"].values.astype(np.float64)
    med = _preacher_hayes_bootstrap(X, M, Y)

    opposing = (med["indirect"] > 0) and (med["direct"] < 0)
    diff = med["indirect"] - full_indirect

    # Verdict
    if opposing and med["indirect_significant"]:
        verdict = "confound_does_not_substantially_bias_mediation"
    elif not med["indirect_significant"]:
        verdict = "confound_partially_drives_mediation_pattern"
    else:
        verdict = "confound_materially_alters_mediation_interpretation"

    result.update({
        "infeasible": False,
        "indirect": med["indirect"], "indirect_ci": med["indirect_ci"],
        "indirect_significant": med["indirect_significant"],
        "direct": med["direct"], "direct_ci": med["direct_ci"],
        "direct_significant": med["direct_significant"],
        "total": med["total"], "opposing_persists": opposing,
        "n_valid_bootstraps": med["n_valid_bootstraps"],
        "verdict": verdict, "restricted_vs_full_indirect_diff": diff,
    })
    logger.info(f"  indirect={med['indirect']:.4f}  CI={med['indirect_ci']}  sig={med['indirect_significant']}")
    logger.info(f"  Opposing persists: {opposing}")
    logger.info(f"  Verdict: {verdict}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4  Unexplained Variance Decomposition
# ═════════════════════════════════════════════════════════════════════════════

def step4_unexplained_variance(df: pd.DataFrame, df_reg: pd.DataFrame, meta1: dict) -> dict:
    """Compute residuals and correlate with corpus size, annotation quality, etc."""
    logger.info("STEP 4: Unexplained Variance Decomposition")
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    coefs = meta1["regression"]["coefficients"]
    beta_morph   = coefs["morph_richness_z"]["beta"]
    beta_head    = coefs["head_direction_ratio_z"]["beta"]
    beta_entropy = coefs["word_order_entropy_z"]["beta"]
    pseudo_r2    = meta1["regression"]["pseudo_r2"]

    # Intercept ~ mean(xi) for z-scored predictors
    intercept = df_reg["xi_raw"].mean()
    logger.info(f"  Intercept (approx): {intercept:.4f}")

    # ── Compute residuals for ALL treebanks ──
    df = df.copy()
    df["xi_predicted"] = (
        intercept
        + beta_morph   * df["morph_richness_z"]
        + beta_head    * df["head_direction_ratio_z"]
        + beta_entropy * df["word_order_entropy_z"]
    )
    df["xi_residual"]     = df["xi_raw"] - df["xi_predicted"]
    df["xi_residual_abs"] = df["xi_residual"].abs()

    # Also for regression subset
    df_reg = df_reg.copy()
    df_reg["xi_predicted"] = (
        intercept
        + beta_morph   * df_reg["morph_richness_z"]
        + beta_head    * df_reg["head_direction_ratio_z"]
        + beta_entropy * df_reg["word_order_entropy_z"]
    )
    df_reg["xi_residual"]     = df_reg["xi_raw"] - df_reg["xi_predicted"]
    df_reg["xi_residual_abs"] = df_reg["xi_residual"].abs()

    # ── 4a: Genre variance ──
    genre_ctrl = meta1.get("genre_control", {})
    within_ranges = {lang: info["within_lang_xi_range"] for lang, info in genre_ctrl.items()}
    mean_range = float(np.mean(list(within_ranges.values()))) if within_ranges else 0.0
    max_range  = float(max(within_ranges.values())) if within_ranges else 0.0
    cross_std  = float(df["xi_raw"].std())
    genre_ratio = mean_range / cross_std if cross_std > 0 else 0.0
    logger.info(f"  Genre: mean_range={mean_range:.4f}  max={max_range:.4f}  cross_std={cross_std:.4f}  ratio={genre_ratio:.4f}")

    # ── 4b-d: Spearman correlations on regression-subset residuals ──
    def _spearman_safe(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 10:
            return NAN_SENTINEL, NAN_SENTINEL
        r, p = stats.spearmanr(a[mask], b[mask])
        return float(r), float(p)

    residuals_reg = df_reg["xi_residual"].values

    # Corpus size
    log_n = np.log10(df_reg["n_sentences_total"].fillna(1).clip(lower=1).values)
    corpus_r, corpus_p = _spearman_safe(log_n, residuals_reg)
    logger.info(f"  Corpus size: r={corpus_r:.4f}  p={corpus_p:.4f}")

    # Annotation completeness
    fc_vals = df_reg["feat_completeness"].values
    fc_r, fc_p = _spearman_safe(fc_vals, residuals_reg)
    logger.info(f"  Feat completeness: r={fc_r:.4f}  p={fc_p:.4f}")

    # N binned
    nb_vals = df_reg["n_binned"].fillna(0).values.astype(float)
    nb_r, nb_p = _spearman_safe(nb_vals, residuals_reg)
    logger.info(f"  N binned: r={nb_r:.4f}  p={nb_p:.4f}")

    # ── Partial R^2 increments ──
    Y_reg = df_reg["xi_raw"].values.astype(np.float64)
    X_base = df_reg[["morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z"]].values.astype(np.float64)

    additional_r2: dict[str, float] = {}

    # Corpus size increment
    df_cs = df_reg.dropna(subset=["n_sentences_total"]).copy()
    if len(df_cs) > 10:
        df_cs["log_n_sentences"] = np.log10(df_cs["n_sentences_total"].clip(lower=1))
        X_b = sm.add_constant(df_cs[["morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z"]].values)
        X_a = sm.add_constant(df_cs[["morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z", "log_n_sentences"]].values)
        r2_base_sub = OLS(df_cs["xi_raw"].values, X_b).fit().rsquared
        r2_aug = OLS(df_cs["xi_raw"].values, X_a).fit().rsquared
        additional_r2["log_n_sentences"] = float(r2_aug - r2_base_sub)

    # Feat completeness increment
    df_fc = df_reg.dropna(subset=["feat_completeness"]).copy()
    if len(df_fc) > 10:
        X_b2 = sm.add_constant(df_fc[["morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z"]].values)
        X_a2 = sm.add_constant(
            np.column_stack([
                df_fc[["morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z"]].values,
                df_fc["feat_completeness"].values,
            ])
        )
        r2_base2 = OLS(df_fc["xi_raw"].values, X_b2).fit().rsquared
        r2_aug2  = OLS(df_fc["xi_raw"].values, X_a2).fit().rsquared
        additional_r2["feat_completeness"] = float(r2_aug2 - r2_base2)

    cumulative = sum(additional_r2.values())
    still_unexpl = 100.0 * (1.0 - pseudo_r2 - cumulative)
    logger.info(f"  Additional R2: {additional_r2}")
    logger.info(f"  Still unexplained: {still_unexpl:.1f}%")

    unmeasured = [
        "areal_effects", "language_contact", "genealogical_inertia",
        "discourse_genre_composition", "annotator_conventions",
        "treebank_sampling_bias", "non_projectivity_rate",
    ]

    return {
        "model_pseudo_r2": pseudo_r2,
        "unexplained_variance_pct": 100.0 * (1.0 - pseudo_r2),
        "genre_within_lang_mean": mean_range,
        "genre_within_lang_max": max_range,
        "genre_ratio": genre_ratio,
        "corpus_size_spearman_r": corpus_r,
        "corpus_size_p": corpus_p,
        "annotation_spearman_r": fc_r,
        "annotation_p": fc_p,
        "n_binned_spearman_r": nb_r,
        "n_binned_p": nb_p,
        "additional_r2": additional_r2,
        "cumulative_additional_r2": cumulative,
        "still_unexplained_pct": still_unexpl,
        "unmeasured_factors": unmeasured,
        "df_with_residuals": df,
        "df_reg_with_residuals": df_reg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5  Mediation Path Diagram
# ═════════════════════════════════════════════════════════════════════════════

def step5_mediation_diagram(step1_res: dict, forward_med: dict) -> str | None:
    """Create a publication-quality mediation path diagram."""
    logger.info("STEP 5: Creating mediation path diagram")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis("off")

        # Boxes
        bp_blue   = dict(boxstyle="round,pad=0.4", facecolor="#D0E8FF", edgecolor="black", linewidth=2)
        bp_yellow = dict(boxstyle="round,pad=0.4", facecolor="#FFF8D0", edgecolor="black", linewidth=2)

        ax.text(2, 5.5, "Morphological\nRichness", ha="center", va="center",
                fontsize=13, fontweight="bold", bbox=bp_blue)
        ax.text(6, 7, "Word-Order\nEntropy", ha="center", va="center",
                fontsize=13, fontweight="bold", bbox=bp_blue)
        ax.text(10, 5.5, r"GEV $\xi$" + "\n(tail constraint)", ha="center", va="center",
                fontsize=13, fontweight="bold", bbox=bp_yellow)

        # Arrow morph -> entropy (path a)
        a_val = forward_med["a"]
        ax.annotate("", xy=(4.8, 6.8), xytext=(3.2, 6.0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="blue"))
        a_stars = "***" if forward_med["a_significant"] else " ns"
        ax.text(3.6, 6.75, f"a = {a_val:.2f}{a_stars}", fontsize=11, color="blue", fontweight="bold")

        # Arrow entropy -> xi (path b)
        b_val = forward_med["b"]
        ax.annotate("", xy=(8.8, 6.0), xytext=(7.2, 6.8),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="blue"))
        b_stars = "***" if forward_med["b_significant"] else " ns"
        ax.text(8.0, 6.75, f"b = {b_val:.3f}{b_stars}", fontsize=11, color="blue", fontweight="bold")

        # Dashed arrow morph -> xi (path c')
        direct    = step1_res["direct"]
        direct_ci = step1_res["direct_ci"]
        ax.annotate("", xy=(8.8, 5.5), xytext=(3.3, 5.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="gray", linestyle="dashed"))
        ax.text(5.5, 5.15, f"c' = {direct:.3f} ns", fontsize=11, color="gray",
                ha="center", fontweight="bold")
        ax.text(5.5, 4.75, f"[{direct_ci[0]:.3f}, {direct_ci[1]:.3f}]",
                fontsize=9, color="gray", ha="center")

        # Summary table
        indirect = step1_res["indirect"]
        ind_ci   = step1_res["indirect_ci"]
        total    = step1_res["total"]
        prop_med = step1_res["proportion_mediated_raw"]

        y = 3.7
        ax.text(6, y, f"Indirect (a*b) = +{indirect:.3f}*  [{ind_ci[0]:.3f}, {ind_ci[1]:.3f}]",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(6, y - 0.45, f"Direct  (c')  = {direct:.3f} ns  [{direct_ci[0]:.3f}, {direct_ci[1]:.3f}]",
                ha="center", fontsize=11)
        ax.text(6, y - 0.9, f"Total   (c)   = {total:.3f} ns",
                ha="center", fontsize=11)

        # Warning box
        ax.text(6, y - 1.55,
                "OPPOSING PATHS: indirect (+) vs direct (-) -> near-zero total",
                ha="center", fontsize=12, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor="red", linewidth=1.5))
        ax.text(6, y - 2.15,
                f"Proportion mediated = {prop_med:.2f}  [UNDEFINED due to near-zero total]",
                ha="center", fontsize=10, color="darkred", fontstyle="italic")

        plt.tight_layout()
        fig_path = WORKSPACE / "mediation_path_diagram.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved diagram: {fig_path}")
        return str(fig_path)

    except Exception as e:
        logger.error(f"  Figure generation failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6  Output Assembly
# ═════════════════════════════════════════════════════════════════════════════

def step6_output_assembly(
    df: pd.DataFrame,
    df_reg: pd.DataFrame,
    meta1: dict,
    step1_res: dict,
    step2_res: dict,
    step3_res: dict,
    step4_res: dict,
    fig_path: str | None,
) -> Path:
    """Assemble eval_out.json in the exp_eval_sol_out schema."""
    logger.info("STEP 6: Output Assembly")

    df_out     = step4_res["df_with_residuals"].copy()
    df_reg_out = step4_res["df_reg_with_residuals"]

    # Mark restricted-subset membership
    thr = step3_res.get("threshold", FEAT_COMPLETENESS_THRESHOLD)
    df_out["restricted_included"] = (
        (df_out["feat_completeness"] > thr)
        & (df_out["n_qualifying_bins"] >= 2)
        & df_out["family"].notna()
        & df_out["morph_richness"].notna()
    ).astype(float)

    # Corpus-size log
    df_out["corpus_size_log"] = df_out["n_sentences_total"].apply(
        lambda x: float(np.log10(max(x, 1))) if pd.notna(x) else NAN_SENTINEL
    )

    # ── Per-treebank examples ──
    examples = []
    for _, row in df_out.iterrows():
        output_dict = {
            "xi_raw": _safe_float(row.get("xi_raw")),
            "morph_richness": _safe_float(row.get("morph_richness")),
            "word_order_entropy": _safe_float(row.get("word_order_entropy")),
            "feat_completeness": _safe_float(row.get("feat_completeness")),
        }
        ex = {
            "input": str(row["treebank_id"]),
            "output": json.dumps(output_dict),
            # -- metadata --
            "metadata_language": str(row.get("language", "")),
            "metadata_iso_code": str(row.get("iso_code", "")),
            "metadata_family": str(row.get("family", "")),
            "metadata_modality": str(row.get("modality", "")),
            "metadata_genre": str(row.get("genre", "")),
            "metadata_feat_completeness": _safe_float(row.get("feat_completeness")),
            # -- global eval: Step 1 --
            "eval_original_total_effect": _safe_float(step1_res["total"]),
            "eval_original_indirect_effect": _safe_float(step1_res["indirect"]),
            "eval_original_direct_effect": _safe_float(step1_res["direct"]),
            "eval_original_proportion_mediated": _safe_float(step1_res["proportion_mediated_raw"]),
            "eval_opposing_paths_flag": 1.0 if step1_res["opposing_paths"] else 0.0,
            "eval_total_effect_near_zero_flag": 1.0 if step1_res["total_near_zero"] else 0.0,
            # -- global eval: Step 2a reverse --
            "eval_reverse_indirect_effect": _safe_float(step2_res["reverse_med"]["indirect"]),
            "eval_reverse_indirect_ci_lower": _safe_float(step2_res["reverse_med"]["indirect_ci"][0]),
            "eval_reverse_indirect_ci_upper": _safe_float(step2_res["reverse_med"]["indirect_ci"][1]),
            "eval_reverse_indirect_significant": 1.0 if step2_res["reverse_med"]["indirect_significant"] else 0.0,
            "eval_reverse_direct_effect": _safe_float(step2_res["reverse_med"]["direct"]),
            "eval_reverse_direct_ci_lower": _safe_float(step2_res["reverse_med"]["direct_ci"][0]),
            "eval_reverse_direct_ci_upper": _safe_float(step2_res["reverse_med"]["direct_ci"][1]),
            "eval_reverse_direct_significant": 1.0 if step2_res["reverse_med"]["direct_significant"] else 0.0,
            "eval_reverse_total_effect": _safe_float(step2_res["reverse_med"]["total"]),
            # -- global eval: Step 2b entropy-only --
            "eval_entropy_only_beta": _safe_float(step2_res["entropy_only_beta"]),
            "eval_entropy_only_r2": _safe_float(step2_res["entropy_only_r2"]),
            "eval_entropy_only_p": _safe_float(step2_res["entropy_only_p"]),
            "eval_entropy_only_r2_vs_full": _safe_float(step2_res["r2_increment_full_vs_entropy_only"]),
            # -- global eval: Step 2c entropy mixed --
            "eval_entropy_mixed_beta": _safe_float(step2_res["entropy_mixed"].get("beta", NAN_SENTINEL)),
            "eval_entropy_mixed_pseudo_r2": _safe_float(step2_res["entropy_mixed"].get("pseudo_r2", NAN_SENTINEL)),
            "eval_entropy_mixed_p": _safe_float(step2_res["entropy_mixed"].get("p", NAN_SENTINEL)),
            # -- global eval: Step 3 restricted --
            "eval_restricted_included": _safe_float(row.get("restricted_included", 0.0)),
            "eval_restricted_indirect_effect": _safe_float(step3_res.get("indirect", NAN_SENTINEL)),
            "eval_restricted_direct_effect": _safe_float(step3_res.get("direct", NAN_SENTINEL)),
            "eval_restricted_total_effect": _safe_float(step3_res.get("total", NAN_SENTINEL)),
            "eval_restricted_opposing_persists": 1.0 if step3_res.get("opposing_persists", False) else 0.0,
            # -- per-treebank eval: Step 4 --
            "eval_xi_residual": _safe_float(row.get("xi_residual", NAN_SENTINEL)),
            "eval_xi_residual_abs": _safe_float(row.get("xi_residual_abs", NAN_SENTINEL)),
            "eval_corpus_size_log": _safe_float(row.get("corpus_size_log", NAN_SENTINEL)),
            "eval_feat_completeness": _safe_float(row.get("feat_completeness", NAN_SENTINEL)),
        }
        examples.append(ex)

    # ── metrics_agg (all values must be JSON numbers) ──
    s1 = step1_res
    s2 = step2_res
    s3 = step3_res
    s4 = step4_res
    rm = s2["reverse_med"]
    fm = s2["forward_med"]

    def _r2d(v):
        """Restricted-safe float."""
        return _safe_float(v) if v != NAN_SENTINEL else NAN_SENTINEL

    metrics_agg = {
        # Step 1
        "mediation_total_effect":          _safe_float(s1["total"]),
        "mediation_indirect_effect":       _safe_float(s1["indirect"]),
        "mediation_direct_effect":         _safe_float(s1["direct"]),
        "mediation_proportion_mediated_raw": _safe_float(s1["proportion_mediated_raw"]),
        "mediation_opposing_paths_detected": 1.0 if s1["opposing_paths"] else 0.0,
        # Step 2a forward
        "forward_mediation_a_path":               _safe_float(fm["a"]),
        "forward_mediation_indirect_recomputed":   _safe_float(fm["indirect"]),
        "forward_mediation_n_valid_bootstraps":    float(fm["n_valid_bootstraps"]),
        # Step 2a reverse
        "reverse_mediation_indirect_mean":        _safe_float(rm["indirect"]),
        "reverse_mediation_indirect_ci_lower":    _safe_float(rm["indirect_ci"][0]),
        "reverse_mediation_indirect_ci_upper":    _safe_float(rm["indirect_ci"][1]),
        "reverse_mediation_indirect_significant": 1.0 if rm["indirect_significant"] else 0.0,
        "reverse_mediation_direct_mean":          _safe_float(rm["direct"]),
        "reverse_mediation_direct_significant":   1.0 if rm["direct_significant"] else 0.0,
        "reverse_vs_forward_distinguishable":     1.0 if s2["reverse_vs_forward_distinguishable"] else 0.0,
        # Step 2b entropy-only
        "entropy_only_ols_beta":          _safe_float(s2["entropy_only_beta"]),
        "entropy_only_ols_r2":            _safe_float(s2["entropy_only_r2"]),
        "entropy_only_ols_p":             _safe_float(s2["entropy_only_p"]),
        "entropy_only_mixed_beta":        _safe_float(s2["entropy_mixed"].get("beta", NAN_SENTINEL)),
        "entropy_only_mixed_pseudo_r2":   _safe_float(s2["entropy_mixed"].get("pseudo_r2", NAN_SENTINEL)),
        "r2_increment_full_vs_entropy_only": _safe_float(s2["r2_increment_full_vs_entropy_only"]),
        # Step 3 restricted
        "restricted_n_treebanks":             float(s3["n_treebanks"]),
        "restricted_n_families":              float(s3["n_families"]),
        "restricted_indirect_effect":         _safe_float(s3.get("indirect", NAN_SENTINEL)),
        "restricted_indirect_ci_lower":       _safe_float(s3.get("indirect_ci", [NAN_SENTINEL, NAN_SENTINEL])[0]),
        "restricted_indirect_ci_upper":       _safe_float(s3.get("indirect_ci", [NAN_SENTINEL, NAN_SENTINEL])[1]),
        "restricted_indirect_significant":    1.0 if s3.get("indirect_significant", False) else 0.0,
        "restricted_direct_effect":           _safe_float(s3.get("direct", NAN_SENTINEL)),
        "restricted_direct_ci_lower":         _safe_float(s3.get("direct_ci", [NAN_SENTINEL, NAN_SENTINEL])[0]),
        "restricted_direct_ci_upper":         _safe_float(s3.get("direct_ci", [NAN_SENTINEL, NAN_SENTINEL])[1]),
        "restricted_direct_significant":      1.0 if s3.get("direct_significant", False) else 0.0,
        "restricted_total_effect":            _safe_float(s3.get("total", NAN_SENTINEL)),
        "restricted_opposing_pattern_persists": 1.0 if s3.get("opposing_persists", False) else 0.0,
        "restricted_vs_full_indirect_diff":   _safe_float(s3.get("restricted_vs_full_indirect_diff", NAN_SENTINEL)),
        # Step 4 variance decomposition
        "model_pseudo_r2":                    _safe_float(s4["model_pseudo_r2"]),
        "unexplained_variance_pct":           _safe_float(s4["unexplained_variance_pct"]),
        "genre_within_lang_variance_mean":    _safe_float(s4["genre_within_lang_mean"]),
        "genre_within_lang_variance_max":     _safe_float(s4["genre_within_lang_max"]),
        "corpus_size_residual_spearman_r":    _safe_float(s4["corpus_size_spearman_r"]),
        "corpus_size_residual_p":             _safe_float(s4["corpus_size_p"]),
        "annotation_completeness_residual_spearman_r": _safe_float(s4["annotation_spearman_r"]),
        "annotation_completeness_residual_p": _safe_float(s4["annotation_p"]),
        "n_sentences_binned_residual_spearman_r": _safe_float(s4["n_binned_spearman_r"]),
        "n_sentences_binned_residual_p":      _safe_float(s4["n_binned_p"]),
        "cumulative_additional_r2":           _safe_float(s4["cumulative_additional_r2"]),
        "still_unexplained_pct":              _safe_float(s4["still_unexplained_pct"]),
    }

    # ── top-level metadata (strings/lists allowed) ──
    top_meta = {
        "evaluation_name": "honest_mediation_reanalysis",
        "description": (
            "Honest mediation reporting, alternative causal models, "
            "confound-restricted mediation, and unexplained variance decomposition"
        ),
        "mediation_interpretation_revised": "statistically_consistent_not_definitive",
        "mediation_explanation": s1["explanation"],
        "indistinguishable_causal_models": s1["indistinguishable_models"],
        "entropy_sufficiency_verdict": s2["entropy_verdict"],
        "confound_impact_verdict": s3.get("verdict", "unknown"),
        "unmeasured_factors_enumerated": s4.get("unmeasured_factors", []),
        "mediation_diagram_path": fig_path,
        "n_treebanks_total": len(df_out),
        "n_treebanks_regression": len(df_reg_out),
    }

    output = {
        "metadata": top_meta,
        "metrics_agg": metrics_agg,
        "datasets": [{"dataset": "mediation_reanalysis", "examples": examples}],
    }

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"  Saved eval_out.json  ({len(examples)} examples, {len(metrics_agg)} agg metrics)")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Honest Mediation Reanalysis Evaluation")
    logger.info("=" * 60)

    # STEP 0
    df, df_reg, meta1 = load_exp_data()

    # STEP 1
    step1_res = step1_honest_mediation(meta1)

    # STEP 2
    step2_res = step2_alternative_models(df_reg)

    # STEP 3
    step3_res = step3_confound_restricted(df, df_reg, full_indirect=step1_res["indirect"])

    # STEP 4
    step4_res = step4_unexplained_variance(df, df_reg, meta1)

    # STEP 5
    fig_path = step5_mediation_diagram(step1_res, step2_res["forward_med"])

    # STEP 6
    out_path = step6_output_assembly(
        df, df_reg, meta1, step1_res, step2_res, step3_res, step4_res, fig_path,
    )

    logger.info("=" * 60)
    logger.info(f"Evaluation complete! -> {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
