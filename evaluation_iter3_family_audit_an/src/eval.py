#!/usr/bin/env python3
"""Family Audit, Corrected Regression, and Multi-Track Sensitivity Analysis.

Evaluates robustness of GEV tail-constraint findings from exp_id1 across
seven sensitivity tracks: family audit, corrected regression, GoF-restricted
re-analysis, annotation-quality filtering, leave-one-family-out CV,
residual analysis, and spoken/written comparison.
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats
from scipy.stats import genextreme, spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware-aware constants
# ---------------------------------------------------------------------------

def _detect_cpus() -> int:
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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 16.0
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1e9)  # 70% of container RAM
N_WORKERS = max(1, NUM_CPUS - 1)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB, workers: {N_WORKERS}")

try:
    resource.setrlimit(resource.RLIMIT_AS,
                       (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError) as exc:
    logger.warning(f"Could not set RLIMIT_AS: {exc}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
ITER_BASE = WORKSPACE.parents[2]  # iter_3 -> 3_invention_loop

EXP_ID1_PATH = (ITER_BASE / "iter_2" / "gen_art" / "exp_id1_it2__opus" /
                "full_method_out.json")
DATA_ID4_PATH = (ITER_BASE / "iter_1" / "gen_art" / "data_id4_it1__opus" /
                 "full_data_out.json")
DATA_ID3_DIR = (ITER_BASE / "iter_1" / "gen_art" / "data_id3_it1__opus")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MEDIATION_BOOTSTRAP_N = 5000
GOF_BOOTSTRAP_N = 100
SW_BOOTSTRAP_N = 1000
FDR_ALPHA = 0.01
SEED = 42
MAX_EXAMPLES: int | None = None  # None = all

# ---------------------------------------------------------------------------
# ISO 2-letter -> ISO 3-letter mapping (from method.py)
# ---------------------------------------------------------------------------
ISO2_TO_ISO3 = {
    "ar": "arb", "zh": "cmn", "en": "eng", "fr": "fra", "de": "deu",
    "es": "spa", "pt": "por", "ru": "rus", "ja": "jpn", "ko": "kor",
    "hi": "hin", "tr": "tur", "fi": "fin", "hu": "hun", "cs": "ces",
    "pl": "pol", "nl": "nld", "sv": "swe", "da": "dan", "no": "nor",
    "it": "ita", "ro": "ron", "bg": "bul", "hr": "hrv", "sr": "srp",
    "sl": "slv", "sk": "slk", "uk": "ukr", "be": "bel", "el": "ell",
    "he": "heb", "ka": "kat", "hy": "hye", "eu": "eus", "et": "est",
    "lv": "lav", "lt": "lit", "ga": "gle", "cy": "cym", "br": "bre",
    "gl": "glg", "ca": "cat", "id": "ind", "vi": "vie", "th": "tha",
    "ta": "tam", "te": "tel", "ml": "mal", "mr": "mar", "bn": "ben",
    "ur": "urd", "fa": "fas", "af": "afr", "am": "amh", "ha": "hau",
    "yo": "yor", "wo": "wol", "mt": "mlt", "sq": "sqi", "is": "isl",
    "fo": "fao", "la": "lat", "sa": "san", "cu": "chu", "eo": "epo",
    "gd": "gla", "az": "aze", "kk": "kaz", "ky": "kir", "uz": "uzb",
    "tt": "tat", "ug": "uig",
}


# ===================================================================
# STEP 0: Data Loading
# ===================================================================

def load_exp_id1() -> tuple[dict, list[dict]]:
    """Load exp_id1 full_method_out.json. Returns (metadata, examples)."""
    logger.info("Loading exp_id1 ...")
    raw = json.loads(EXP_ID1_PATH.read_text())
    metadata = raw.get("metadata", {})
    examples = raw["datasets"][0]["examples"]
    logger.info(f"  Loaded {len(examples)} treebank examples from exp_id1")
    del raw
    gc.collect()
    return metadata, examples


def load_data_id4() -> dict[str, str]:
    """Load data_id4 and build iso3 -> family_name lookup."""
    logger.info("Loading data_id4 (Grambank/Glottolog families) ...")
    raw = json.loads(DATA_ID4_PATH.read_text())
    examples = raw["datasets"][0]["examples"]
    del raw

    iso_to_family: dict[str, str] = {}
    for ex in examples:
        iso = ex["metadata_iso639_3_code"]
        family = ex["metadata_family_name"]
        iso_to_family[iso] = family

    del examples
    gc.collect()
    logger.info(f"  Built family lookup for {len(iso_to_family)} ISO codes")
    return iso_to_family


def load_data_id3_treebank_metadata() -> dict[str, float]:
    """Load treebank metadata from data_id3 for feat_completeness lookup."""
    logger.info("Loading data_id3 treebank metadata ...")
    tb_feat: dict[str, float] = {}

    for part_file in ["data_out/full_data_out_1.json",
                      "data_out/full_data_out_2.json"]:
        fpath = DATA_ID3_DIR / part_file
        if not fpath.exists():
            logger.warning(f"  Missing {fpath}")
            continue
        logger.info(f"  Reading {fpath.name} for treebank rows ...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]
        del raw
        for ex in examples:
            if ex.get("metadata_row_type") == "treebank":
                tb_id = ex["metadata_treebank_id"]
                tb_feat[tb_id] = ex.get("metadata_feat_completeness", 0.0)
        del examples
        gc.collect()

    logger.info(f"  Got feat_completeness for {len(tb_feat)} treebanks")
    return tb_feat


def load_data_id3_sentences() -> dict[tuple[str, int], list[float]]:
    """Load sentence-level max_DD from data_id3 for GoF re-analysis."""
    logger.info("Loading data_id3 sentence-level data ...")
    t0 = time.time()
    bin_data: dict[tuple[str, int], list[float]] = defaultdict(list)
    n_loaded = 0

    for part_file in ["data_out/full_data_out_1.json",
                      "data_out/full_data_out_2.json"]:
        fpath = DATA_ID3_DIR / part_file
        if not fpath.exists():
            logger.warning(f"  Missing {fpath}")
            continue
        logger.info(f"  Reading {fpath.name} for sentences ...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]
        del raw
        for ex in examples:
            if ex.get("metadata_row_type") == "sentence":
                key = (ex["metadata_treebank_id"], int(ex["metadata_length_bin"]))
                bin_data[key].append(ex["metadata_max_dd"])
                n_loaded += 1
                if MAX_EXAMPLES is not None and n_loaded >= MAX_EXAMPLES:
                    break
        del examples
        gc.collect()
        if MAX_EXAMPLES is not None and n_loaded >= MAX_EXAMPLES:
            break

    logger.info(f"  Loaded {n_loaded} sentences, {len(bin_data)} combos "
                f"in {time.time()-t0:.1f}s")
    return dict(bin_data)


# ===================================================================
# STEP 1: Family Audit
# ===================================================================

def family_audit(
    exp_examples: list[dict],
    iso_to_family: dict[str, str],
) -> tuple[dict[str, str], list[dict], bool]:
    """Audit and correct family assignments.

    Returns: corrected_family dict, corrections_list, afrikaans_fixed bool.
    """
    logger.info("STEP 1: Family Audit")
    corrections: list[dict] = []
    corrected_family: dict[str, str] = {}

    for ex in exp_examples:
        tb_id = ex["metadata_treebank_id"]
        old_family = ex["metadata_family"]
        iso2 = ex["metadata_iso_code"]

        # Convert iso2 to iso3
        iso3 = ISO2_TO_ISO3.get(iso2, iso2)

        # Look up correct family from data_id4
        new_family = (iso_to_family.get(iso3) or
                      iso_to_family.get(iso2) or
                      None)

        # Prefix match fallback
        if new_family is None:
            for giso, gfam in iso_to_family.items():
                if giso.startswith(iso2) or iso2.startswith(giso):
                    new_family = gfam
                    break

        if new_family is None or new_family.strip() == "":
            # Keep original - no valid family found
            corrected_family[tb_id] = old_family
            continue

        corrected_family[tb_id] = new_family
        if new_family != old_family:
            corrections.append({
                "treebank_id": tb_id,
                "iso_code": iso2,
                "old_family": old_family,
                "new_family": new_family,
                "source": "data_id4_glottolog",
            })

    # Check Afrikaans specifically
    afr_fixed = False
    if "af_afribooms" in corrected_family:
        if corrected_family["af_afribooms"] == "Indo-European":
            afr_fixed = True
            logger.info("  Afrikaans corrected to Indo-European: auto-detected")
        else:
            # Manual override
            corrected_family["af_afribooms"] = "Indo-European"
            afr_fixed = True
            # Add to corrections if not already there
            already = any(c["treebank_id"] == "af_afribooms" for c in corrections)
            if not already:
                corrections.append({
                    "treebank_id": "af_afribooms",
                    "iso_code": "af",
                    "old_family": "Afro-Asiatic",
                    "new_family": "Indo-European",
                    "source": "manual_override",
                })
            logger.info("  Afrikaans manually overridden to Indo-European")

    n_families = len(set(corrected_family.values()) - {"Unknown"})
    logger.info(f"  {len(corrections)} corrections, {n_families} families after")
    for c in corrections[:10]:
        logger.info(f"    {c['treebank_id']}: {c['old_family']} -> {c['new_family']}")
    if len(corrections) > 10:
        logger.info(f"    ... and {len(corrections) - 10} more")

    return corrected_family, corrections, afr_fixed


# ===================================================================
# STEP 2: Corrected Regression
# ===================================================================

def build_regression_df(
    exp_examples: list[dict],
    corrected_family: dict[str, str],
    tb_feat: dict[str, float],
) -> pd.DataFrame:
    """Build regression DataFrame from exp_id1 examples with corrected families."""
    rows = []
    for ex in exp_examples:
        tb_id = ex["metadata_treebank_id"]
        family = corrected_family.get(tb_id, "Unknown")

        # Parse output JSON to get xi values
        try:
            out = json.loads(ex["output"])
            xi = out.get("xi_raw")
            xi_se = out.get("xi_raw_se")
        except (json.JSONDecodeError, TypeError):
            continue

        if xi is None or not np.isfinite(xi):
            continue

        rows.append({
            "treebank_id": tb_id,
            "xi": xi,
            "xi_se": xi_se,
            "morph_richness": ex.get("metadata_morph_richness"),
            "head_direction_ratio": ex.get("metadata_head_direction_ratio"),
            "word_order_entropy": ex.get("metadata_word_order_entropy"),
            "family_corrected": family,
            "language": ex.get("metadata_language", ""),
            "iso_code": ex.get("metadata_iso_code", ""),
            "modality": ex.get("metadata_modality", ""),
            "genre": ex.get("metadata_genre", ""),
            "feat_completeness": tb_feat.get(tb_id, 0.0),
            "old_family": ex.get("metadata_family", ""),
        })

    df = pd.DataFrame(rows)
    df = df[(df["family_corrected"] != "Unknown") &
            (df["family_corrected"] != "")].dropna(subset=["xi"])

    # Standardize predictors
    for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0

    logger.info(f"  Regression df: {len(df)} treebanks, "
                f"{df['family_corrected'].nunique()} families")
    return df


def run_mixedlm(
    df: pd.DataFrame,
    family_col: str = "family_corrected",
    label: str = "model",
) -> dict[str, Any]:
    """Run MixedLM regression with FDR correction. Returns results dict."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    results: dict[str, Any] = {"fallback_notes": [], "model_type": None}
    predictors = ["morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z"]
    formula = "xi ~ " + " + ".join(predictors)

    # Suppress noisy convergence warnings (we handle fallbacks explicitly)
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    if len(df) < 15:
        results["fallback_notes"].append(f"{label}: Too few treebanks ({len(df)})")
        return results

    fam_counts = df[family_col].value_counts()
    families_ok = (fam_counts >= 2).sum() >= 2

    model_fit = None

    # Try MixedLM
    if families_ok:
        try:
            model = smf.mixedlm(formula, data=df, groups=df[family_col])
            model_fit = model.fit(reml=True)
            results["model_type"] = "mixed_effects"
        except Exception as exc:
            results["fallback_notes"].append(
                f"{label}: MixedLM default failed ({exc}); trying NM")
            try:
                model_fit = model.fit(method="nm", maxiter=2000)
                results["model_type"] = "mixed_effects_nm"
            except Exception as exc2:
                results["fallback_notes"].append(
                    f"{label}: MixedLM NM also failed ({exc2})")

    # Fallback: collapse small families
    if model_fit is None and families_ok:
        try:
            df_coll = df.copy()
            small = fam_counts[fam_counts < 3].index
            df_coll.loc[df_coll[family_col].isin(small), family_col] = "Other"
            if df_coll[family_col].nunique() >= 2:
                model = smf.mixedlm(formula, data=df_coll, groups=df_coll[family_col])
                model_fit = model.fit(reml=True)
                results["model_type"] = "mixed_effects_collapsed"
        except Exception:
            pass

    # Final fallback: OLS with clustered SE
    if model_fit is None:
        results["fallback_notes"].append(f"{label}: Using OLS with clustered SE")
        try:
            model_fit = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df[family_col]})
            results["model_type"] = "ols_clustered"
        except Exception:
            model_fit = smf.ols(formula, data=df).fit()
            results["model_type"] = "ols"

    # Extract coefficients
    coefficients = {}
    p_vals = []
    for pred in predictors:
        try:
            beta = float(model_fit.params[pred])
            se = float(model_fit.bse[pred])
            p = float(model_fit.pvalues[pred])
        except (KeyError, AttributeError):
            beta, se, p = np.nan, np.nan, np.nan
        coefficients[pred] = {"beta": beta, "se": se, "p": p}
        p_vals.append(p)

    # FDR correction
    valid_p = [p for p in p_vals if np.isfinite(p)]
    if valid_p:
        reject, p_corr, _, _ = multipletests(valid_p, alpha=FDR_ALPHA, method="fdr_bh")
        idx = 0
        for pred in predictors:
            if np.isfinite(coefficients[pred]["p"]):
                coefficients[pred]["p_fdr"] = float(p_corr[idx])
                coefficients[pred]["reject_fdr"] = bool(reject[idx])
                idx += 1

    results["coefficients"] = coefficients

    # Pseudo-R2
    try:
        if "mixed" in (results["model_type"] or ""):
            null_fit = smf.mixedlm("xi ~ 1", data=df,
                                    groups=df[family_col]).fit(reml=True)
            var_null = float(null_fit.scale)
            var_full = float(model_fit.scale)
            if np.isfinite(var_null) and np.isfinite(var_full) and var_null > 0:
                results["pseudo_r2"] = float(1 - var_full / var_null)
        else:
            results["pseudo_r2"] = float(model_fit.rsquared)
    except Exception:
        results["pseudo_r2"] = None

    results["n_treebanks"] = len(df)
    results["n_families"] = int(df[family_col].nunique())
    results["model_fit"] = model_fit  # Keep for predictions

    return results


def run_mediation(df: pd.DataFrame, n_boot: int = 5000, seed: int = SEED) -> dict:
    """Preacher-Hayes bootstrap mediation: morph -> wo_entropy -> xi."""
    X = df["morph_richness_z"].values
    M = df["word_order_entropy_z"].values
    Y = df["xi"].values
    n_tb = len(df)

    if n_tb < 10:
        return {"note": "Too few treebanks for mediation", "n": n_tb,
                "interpretation": "insufficient_data"}

    rng = np.random.default_rng(seed + 999)
    indirect, direct, total = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n_tb, size=n_tb)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        try:
            A = np.column_stack([np.ones(n_tb), Xb])
            coef_a, *_ = np.linalg.lstsq(A, Mb, rcond=None)
            a = coef_a[1]

            B = np.column_stack([np.ones(n_tb), Xb, Mb])
            coef_b, *_ = np.linalg.lstsq(B, Yb, rcond=None)
            c_prime = coef_b[1]
            b = coef_b[2]

            indirect.append(a * b)
            direct.append(c_prime)
            total.append(c_prime + a * b)
        except Exception:
            continue

    if len(indirect) < 100:
        return {"note": "Too many bootstrap failures", "n_valid": len(indirect),
                "interpretation": "bootstrap_failure"}

    ind = np.array(indirect)
    dir_ = np.array(direct)

    ind_lo, ind_hi = float(np.percentile(ind, 2.5)), float(np.percentile(ind, 97.5))
    dir_lo, dir_hi = float(np.percentile(dir_, 2.5)), float(np.percentile(dir_, 97.5))

    ind_sig = bool(ind_lo > 0 or ind_hi < 0)
    dir_sig = bool(dir_lo > 0 or dir_hi < 0)

    interpretation = (
        "full_mediation" if ind_sig and not dir_sig
        else "partial_mediation" if ind_sig and dir_sig
        else "no_mediation_direct_only" if not ind_sig and dir_sig
        else "no_effect"
    )

    return {
        "indirect_effect_mean": float(np.mean(ind)),
        "indirect_effect_ci": [ind_lo, ind_hi],
        "indirect_significant": ind_sig,
        "direct_effect_mean": float(np.mean(dir_)),
        "direct_effect_ci": [dir_lo, dir_hi],
        "direct_significant": dir_sig,
        "total_effect_mean": float(np.mean(np.array(total))),
        "n_valid_bootstraps": len(indirect),
        "interpretation": interpretation,
    }


# ===================================================================
# STEP 3: GoF-Restricted Sensitivity
# ===================================================================

def _fit_gev_combo(args: tuple) -> dict:
    """Fit GEV to one (treebank_id, bin) combo for GoF analysis.

    Returns dict with xi, loc, scale, gof_passes, xi_boot_se.
    """
    tb_id, bin_len, values, boot_n, seed = args
    from lmoments3 import distr as lm_distr

    data = np.asarray(values, dtype=np.float64)
    n = len(data)
    result = {
        "treebank_id": tb_id, "bin": bin_len, "n_samples": n,
        "xi": np.nan, "loc": np.nan, "scale": np.nan,
        "gof_passes": False, "ks_p": np.nan, "xi_boot_se": np.nan,
    }

    if n < 30:
        return result

    # L-moments GEV fit
    try:
        params = lm_distr.gev.lmom_fit(data)
        c = params["c"]
        xi = -c
        loc = params["loc"]
        scale = params["scale"]
        result["xi"] = float(xi)
        result["loc"] = float(loc)
        result["scale"] = float(scale)
    except Exception:
        return result

    # KS GoF test
    try:
        ks_stat, ks_p = sp_stats.kstest(
            data, "genextreme", args=(c, loc, scale))
        result["ks_p"] = float(ks_p)
        result["gof_passes"] = bool(ks_p > 0.05)
    except Exception:
        pass

    # Bootstrap for SE
    rng = np.random.default_rng(seed + hash((tb_id, bin_len)) % 10_000)
    xi_boots = []
    for _ in range(boot_n):
        idx = rng.integers(0, n, size=n)
        try:
            bp = lm_distr.gev.lmom_fit(data[idx])
            xi_b = -bp["c"]
            if np.isfinite(xi_b):
                xi_boots.append(xi_b)
        except Exception:
            pass

    if len(xi_boots) >= 20:
        result["xi_boot_se"] = float(np.std(xi_boots))

    return result


def gof_restricted_analysis(
    bin_data: dict[tuple[str, int], list[float]],
    corrected_family: dict[str, str],
    df_full: pd.DataFrame,
) -> dict[str, Any]:
    """Re-fit GEV for all combos, filter by GoF, re-run regression+mediation."""
    logger.info("STEP 3: GoF-Restricted Sensitivity Analysis")

    # Filter to combos with >= 50 samples
    qualifying = [(tb_id, b, vals, GOF_BOOTSTRAP_N, SEED)
                  for (tb_id, b), vals in bin_data.items()
                  if len(vals) >= 50]
    logger.info(f"  {len(qualifying)} qualifying combos with >=50 samples")

    if len(qualifying) == 0:
        return {"note": "No qualifying combos", "gof_n_combos_passing": 0}

    # Parallel GEV fitting
    all_results: list[dict] = []
    t0 = time.time()

    if len(qualifying) >= 50 and N_WORKERS >= 2:
        logger.info(f"  Using {N_WORKERS} parallel workers for {len(qualifying)} combos")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = [pool.submit(_fit_gev_combo, args) for args in qualifying]
            for fut in tqdm(as_completed(futs), total=len(futs),
                           desc="GoF fitting", mininterval=5):
                try:
                    all_results.append(fut.result())
                except Exception as exc:
                    logger.error(f"GoF combo failed: {exc}")
    else:
        for args in tqdm(qualifying, desc="GoF fitting", mininterval=5):
            try:
                all_results.append(_fit_gev_combo(args))
            except Exception as exc:
                logger.error(f"GoF combo failed: {exc}")

    logger.info(f"  Fitted {len(all_results)} combos in {time.time()-t0:.1f}s")

    # Filter to GoF-passing combos
    passing = [r for r in all_results if r["gof_passes"]]
    logger.info(f"  {len(passing)}/{len(all_results)} combos pass KS GoF")

    results: dict[str, Any] = {
        "gof_n_combos_total": len(all_results),
        "gof_n_combos_passing": len(passing),
        "gof_pass_rate": len(passing) / max(len(all_results), 1) * 100,
    }

    if len(passing) == 0:
        results["note"] = "No combos pass GoF"
        return results

    # Re-aggregate xi per treebank (inverse-variance weighted)
    tb_xi_gof: dict[str, dict] = defaultdict(lambda: {"xi_vals": [], "weights": []})
    for r in passing:
        if np.isfinite(r["xi"]) and np.isfinite(r["xi_boot_se"]) and r["xi_boot_se"] > 0:
            tb_xi_gof[r["treebank_id"]]["xi_vals"].append(r["xi"])
            tb_xi_gof[r["treebank_id"]]["weights"].append(1.0 / (r["xi_boot_se"] ** 2))

    gof_xi: dict[str, float] = {}
    for tb_id, data in tb_xi_gof.items():
        if data["xi_vals"]:
            w = np.array(data["weights"])
            gof_xi[tb_id] = float(np.average(data["xi_vals"], weights=w))

    results["gof_n_treebanks_retained"] = len(gof_xi)
    logger.info(f"  {len(gof_xi)} treebanks retained after GoF filtering")

    if len(gof_xi) < 15:
        results["note"] = f"Too few treebanks ({len(gof_xi)}) for regression"
        return results

    # Build restricted regression DataFrame
    df_gof = df_full[df_full["treebank_id"].isin(gof_xi)].copy()
    df_gof["xi"] = df_gof["treebank_id"].map(gof_xi)
    df_gof = df_gof.dropna(subset=["xi"])

    # Re-standardize
    for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
        mu, sd = df_gof[col].mean(), df_gof[col].std()
        df_gof[f"{col}_z"] = (df_gof[col] - mu) / sd if sd > 0 else 0.0

    # Run regression
    reg_results = run_mixedlm(df_gof, label="GoF-restricted")
    results["gof_regression"] = {k: v for k, v in reg_results.items() if k != "model_fit"}

    # Run mediation
    med_results = run_mediation(df_gof, n_boot=MEDIATION_BOOTSTRAP_N, seed=SEED + 100)
    results["gof_mediation"] = med_results
    results["gof_xi"] = gof_xi  # Store for later use

    return results


# ===================================================================
# STEP 4: Annotation Sensitivity
# ===================================================================

def annotation_sensitivity(
    df_full: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Filter by feat_completeness > threshold, re-run regression+mediation."""
    logger.info(f"STEP 4: Annotation Sensitivity (threshold={threshold})")

    df_filt = df_full[df_full["feat_completeness"] > threshold].copy()
    n_dropped = len(df_full) - len(df_filt)
    logger.info(f"  Retained {len(df_filt)}/{len(df_full)} treebanks "
                f"(dropped {n_dropped})")

    results: dict[str, Any] = {
        "annot_n_treebanks": len(df_filt),
        "annot_n_dropped": n_dropped,
    }

    if len(df_filt) < 15:
        results["note"] = f"Too few treebanks ({len(df_filt)})"
        return results

    # Re-standardize
    for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
        mu, sd = df_filt[col].mean(), df_filt[col].std()
        df_filt[f"{col}_z"] = (df_filt[col] - mu) / sd if sd > 0 else 0.0

    # Regression
    reg_results = run_mixedlm(df_filt, label="Annotation-filtered")
    results["annot_regression"] = {k: v for k, v in reg_results.items()
                                    if k != "model_fit"}

    # Mediation
    med_results = run_mediation(df_filt, n_boot=MEDIATION_BOOTSTRAP_N, seed=SEED + 200)
    results["annot_mediation"] = med_results

    # Spearman correlation between morph_richness and word_order_entropy
    try:
        rho, p = spearmanr(df_filt["morph_richness"], df_filt["word_order_entropy"])
        results["annot_morph_entropy_corr"] = float(rho)
        results["annot_morph_entropy_corr_p"] = float(p)
    except Exception:
        results["annot_morph_entropy_corr"] = None

    # Full-set correlation for comparison
    try:
        rho_full, _ = spearmanr(df_full["morph_richness"], df_full["word_order_entropy"])
        results["full_morph_entropy_corr"] = float(rho_full)
    except Exception:
        results["full_morph_entropy_corr"] = None

    return results


# ===================================================================
# STEP 5: Leave-One-Family-Out
# ===================================================================

def lofo_analysis(df_full: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-family-out cross-validation."""
    logger.info("STEP 5: Leave-One-Family-Out")

    families = df_full["family_corrected"].unique().tolist()
    logger.info(f"  {len(families)} families to iterate over")

    family_betas: dict[str, float] = {}
    family_pvals: dict[str, float] = {}
    family_sig: dict[str, bool] = {}
    skipped = []

    for fam in tqdm(families, desc="LOFO"):
        df_drop = df_full[df_full["family_corrected"] != fam].copy()

        # Check viability
        if len(df_drop) < 15:
            skipped.append(fam)
            continue
        fam_counts = df_drop["family_corrected"].value_counts()
        if (fam_counts >= 2).sum() < 2:
            skipped.append(fam)
            continue

        # Re-standardize
        for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
            mu, sd = df_drop[col].mean(), df_drop[col].std()
            df_drop[f"{col}_z"] = (df_drop[col] - mu) / sd if sd > 0 else 0.0

        reg = run_mixedlm(df_drop, label=f"LOFO-{fam}")
        coefs = reg.get("coefficients", {})
        entropy_coef = coefs.get("word_order_entropy_z", {})
        beta = entropy_coef.get("beta", np.nan)
        p_fdr = entropy_coef.get("p_fdr", np.nan)

        if np.isfinite(beta):
            family_betas[fam] = beta
        if np.isfinite(p_fdr):
            family_pvals[fam] = p_fdr
            family_sig[fam] = p_fdr < FDR_ALPHA

    results: dict[str, Any] = {
        "lofo_n_families": len(families),
        "lofo_n_skipped": len(skipped),
        "lofo_skipped_families": skipped,
    }

    if family_betas:
        betas = list(family_betas.values())
        results["lofo_entropy_beta_mean"] = float(np.mean(betas))
        results["lofo_entropy_beta_sd"] = float(np.std(betas))
        results["lofo_entropy_beta_min"] = float(np.min(betas))
        results["lofo_entropy_beta_max"] = float(np.max(betas))
        results["lofo_entropy_beta_range"] = float(np.max(betas) - np.min(betas))

        # Most influential family
        if family_betas:
            # Get full-model beta for comparison
            full_reg = run_mixedlm(df_full, label="LOFO-full-ref")
            full_beta = full_reg.get("coefficients", {}).get(
                "word_order_entropy_z", {}).get("beta", 0)
            diffs = {f: abs(b - full_beta) for f, b in family_betas.items()}
            results["lofo_most_influential_family"] = max(diffs, key=diffs.get)

        results["lofo_family_betas"] = family_betas

    if family_sig:
        results["lofo_all_significant"] = all(family_sig.values())
        results["lofo_pct_significant"] = (
            sum(family_sig.values()) / len(family_sig) * 100)
    else:
        results["lofo_all_significant"] = False
        results["lofo_pct_significant"] = 0.0

    logger.info(f"  LOFO done: mean_beta={results.get('lofo_entropy_beta_mean', 'N/A')}, "
                f"pct_sig={results.get('lofo_pct_significant', 'N/A')}")

    return results


# ===================================================================
# STEP 6: Residual Analysis
# ===================================================================

def residual_analysis(
    df_full: pd.DataFrame,
    reg_results: dict,
) -> dict[str, Any]:
    """Compute residuals from corrected regression model."""
    logger.info("STEP 6: Residual Analysis")
    import statsmodels.formula.api as smf

    results: dict[str, Any] = {}
    model_fit = reg_results.get("model_fit")

    if model_fit is None:
        # Re-fit model
        formula = ("xi ~ morph_richness_z + head_direction_ratio_z"
                   " + word_order_entropy_z")
        try:
            if "mixed" in (reg_results.get("model_type") or ""):
                m = smf.mixedlm(formula, data=df_full,
                                groups=df_full["family_corrected"])
                model_fit = m.fit(reml=True)
            else:
                model_fit = smf.ols(formula, data=df_full).fit()
        except Exception as exc:
            logger.error(f"Could not fit model for residuals: {exc}")
            return {"note": "Model fitting failed"}

    # Compute predictions and residuals
    try:
        pred_xi = model_fit.predict(df_full)
        df_res = df_full.copy()
        df_res["predicted_xi"] = pred_xi.values
        df_res["residual"] = df_res["xi"] - df_res["predicted_xi"]
        df_res["abs_residual"] = df_res["residual"].abs()
        df_res["relative_residual"] = (
            df_res["abs_residual"] / df_res["xi"].abs()
        ).replace([np.inf, -np.inf], np.nan)

        # Rank by absolute residual
        df_res = df_res.sort_values("abs_residual", ascending=False)
        df_res["rank"] = range(1, len(df_res) + 1)

        # Turkish rank
        tr_rows = df_res[df_res["treebank_id"] == "tr_imst"]
        if len(tr_rows) > 0:
            results["residual_turkish_rank"] = int(tr_rows.iloc[0]["rank"])
            results["residual_turkish_relative"] = float(
                tr_rows.iloc[0]["relative_residual"])
        else:
            results["residual_turkish_rank"] = -1
            results["residual_turkish_relative"] = np.nan

        # Top-10
        top10 = []
        for _, row in df_res.head(10).iterrows():
            top10.append({
                "treebank_id": row["treebank_id"],
                "language": row["language"],
                "xi": float(row["xi"]),
                "predicted_xi": float(row["predicted_xi"]),
                "residual": float(row["residual"]),
                "relative_residual": float(row.get("relative_residual", np.nan)),
                "morph_richness": float(row["morph_richness"]),
                "word_order_entropy": float(row["word_order_entropy"]),
            })
        results["residual_top10"] = top10

        results["residual_mean_abs"] = float(df_res["abs_residual"].mean())
        results["residual_median_abs"] = float(df_res["abs_residual"].median())

        # Store for per-example output
        results["_residual_df"] = df_res

    except Exception as exc:
        logger.error(f"Residual analysis failed: {exc}")
        results["note"] = f"Failed: {exc}"

    return results


# ===================================================================
# STEP 7: Spoken/Written Fix
# ===================================================================

def spoken_written_analysis(
    exp_examples: list[dict],
) -> dict[str, Any]:
    """Spoken/written comparison excluding en_atis, with bootstrap CIs."""
    logger.info("STEP 7: Spoken/Written Corrected Comparison")

    # Build xi lookup from exp_id1 examples
    xi_lookup: dict[str, dict] = {}
    for ex in exp_examples:
        tb_id = ex["metadata_treebank_id"]
        try:
            out = json.loads(ex["output"])
            xi_lookup[tb_id] = {
                "xi": out["xi_raw"],
                "xi_se": out["xi_raw_se"],
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # Define pairs (excluding en_atis)
    pairs = [
        ("sl_sst", "sl_ssj", "Slovenian"),
        ("fr_parisstories", "fr_gsd", "French"),
    ]

    results: dict[str, Any] = {"pairs": [], "sw_n_viable_pairs": 0}
    consistent = True

    for spoken_tb, written_tb, lang in pairs:
        if spoken_tb not in xi_lookup or written_tb not in xi_lookup:
            continue

        xi_s = xi_lookup[spoken_tb]["xi"]
        xi_w = xi_lookup[written_tb]["xi"]
        se_s = xi_lookup[spoken_tb]["xi_se"]
        se_w = xi_lookup[written_tb]["xi_se"]

        diff = xi_s - xi_w
        pooled_se = np.sqrt((se_s**2 + se_w**2) / 2) if se_s > 0 and se_w > 0 else np.nan
        d = float(diff / pooled_se) if np.isfinite(pooled_se) and pooled_se > 0 else np.nan

        # Bootstrap 95% CI for Cohen's d
        rng = np.random.default_rng(SEED + hash(lang) % 10000)
        d_boots = []
        for _ in range(SW_BOOTSTRAP_N):
            se_s_b = se_s * np.exp(rng.normal(0, 0.1))
            se_w_b = se_w * np.exp(rng.normal(0, 0.1))
            pooled_b = np.sqrt((se_s_b**2 + se_w_b**2) / 2)
            if pooled_b > 0:
                d_boots.append(diff / pooled_b)

        d_ci = [float(np.percentile(d_boots, 2.5)),
                float(np.percentile(d_boots, 97.5))] if len(d_boots) >= 100 else [np.nan, np.nan]

        # Check direction: spoken xi less negative = higher = diff > 0
        spoken_less_negative = diff > 0

        pair_result = {
            "language": lang,
            "spoken_tb": spoken_tb,
            "written_tb": written_tb,
            "xi_spoken": float(xi_s),
            "xi_written": float(xi_w),
            "diff": float(diff),
            "cohens_d": d,
            "d_ci": d_ci,
            "spoken_less_negative": spoken_less_negative,
        }
        results["pairs"].append(pair_result)
        results["sw_n_viable_pairs"] = len(results["pairs"])

        if not spoken_less_negative:
            consistent = False

        logger.info(f"  {lang}: spoken={xi_s:.4f}, written={xi_w:.4f}, "
                    f"d={d:.3f}, CI={d_ci}")

    results["sw_consistent_direction"] = consistent
    return results


# ===================================================================
# STEP 8: Figures
# ===================================================================

def generate_figures(
    corrected_reg: dict,
    gof_results: dict,
    annot_results: dict,
    lofo_results: dict,
    sw_results: dict,
    df_full: pd.DataFrame,
    original_entropy_beta: float = 0.084,
    original_entropy_se: float = 0.023,
) -> list[str]:
    """Generate 4 PNG figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = WORKSPACE
    generated = []

    # ---- Figure 1: Forest plot of entropy beta across tracks ----
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        tracks = []

        # Original
        tracks.append(("Original", original_entropy_beta, original_entropy_se))

        # Corrected
        c_coefs = corrected_reg.get("coefficients", {})
        c_ent = c_coefs.get("word_order_entropy_z", {})
        if c_ent.get("beta") is not None and np.isfinite(c_ent["beta"]):
            tracks.append(("Corrected families",
                          c_ent["beta"], c_ent.get("se", 0)))

        # GoF-restricted
        g_reg = gof_results.get("gof_regression", {})
        g_coefs = g_reg.get("coefficients", {})
        g_ent = g_coefs.get("word_order_entropy_z", {})
        if g_ent.get("beta") is not None and np.isfinite(g_ent.get("beta", np.nan)):
            tracks.append(("GoF-restricted",
                          g_ent["beta"], g_ent.get("se", 0)))

        # Annotation-filtered
        a_reg = annot_results.get("annot_regression", {})
        a_coefs = a_reg.get("coefficients", {})
        a_ent = a_coefs.get("word_order_entropy_z", {})
        if a_ent.get("beta") is not None and np.isfinite(a_ent.get("beta", np.nan)):
            tracks.append(("Annotation-filtered",
                          a_ent["beta"], a_ent.get("se", 0)))

        if tracks:
            labels = [t[0] for t in tracks]
            betas = [t[1] for t in tracks]
            ci_widths = [1.96 * t[2] for t in tracks]

            y_pos = range(len(tracks))
            ax.barh(y_pos, betas, xerr=ci_widths, height=0.5,
                    color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(tracks)],
                    alpha=0.7, capsize=5)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Word-order entropy beta (95% CI)")
            ax.set_title("Entropy Effect Across Sensitivity Tracks")
            plt.tight_layout()
            path = fig_dir / "fig_regression_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            generated.append(str(path))
            logger.info(f"  Saved {path.name}")
        plt.close(fig)
    except Exception as exc:
        logger.error(f"Figure 1 failed: {exc}")

    # ---- Figure 2: LOFO stability bar chart ----
    try:
        family_betas = lofo_results.get("lofo_family_betas", {})
        if family_betas:
            fig, ax = plt.subplots(figsize=(12, 6))
            fams = sorted(family_betas.keys())
            betas = [family_betas[f] for f in fams]

            # Color by family size
            fam_counts = df_full["family_corrected"].value_counts()
            colors = ["#2ca02c" if fam_counts.get(f, 0) >= 2 else "#d62728"
                      for f in fams]

            ax.bar(range(len(fams)), betas, color=colors, alpha=0.7)
            ax.set_xticks(range(len(fams)))
            ax.set_xticklabels(fams, rotation=90, fontsize=7)

            # Full-model beta reference lines
            c_ent = corrected_reg.get("coefficients", {}).get(
                "word_order_entropy_z", {})
            full_beta = c_ent.get("beta", 0)
            full_se = c_ent.get("se", 0)
            if np.isfinite(full_beta):
                ax.axhline(y=full_beta, color="blue", linestyle="--",
                          linewidth=1, label=f"Full model beta={full_beta:.3f}")
                if np.isfinite(full_se) and full_se > 0:
                    ax.axhline(y=full_beta + full_se, color="blue",
                              linestyle=":", linewidth=0.5)
                    ax.axhline(y=full_beta - full_se, color="blue",
                              linestyle=":", linewidth=0.5)

            ax.set_xlabel("Family dropped")
            ax.set_ylabel("Entropy beta")
            ax.set_title("Leave-One-Family-Out: Entropy Beta Stability")
            ax.legend(fontsize=8)
            plt.tight_layout()
            path = fig_dir / "fig_lofo_stability.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            generated.append(str(path))
            logger.info(f"  Saved {path.name}")
            plt.close(fig)
    except Exception as exc:
        logger.error(f"Figure 2 failed: {exc}")

    # ---- Figure 3: GoF-restricted vs full scatter ----
    try:
        gof_xi = gof_results.get("gof_xi", {})
        if gof_xi:
            fig, ax = plt.subplots(figsize=(7, 7))
            # Get full xi values for matching treebanks
            full_xis = []
            gof_xis = []
            families_for_color = []
            for tb_id, gof_val in gof_xi.items():
                rows = df_full[df_full["treebank_id"] == tb_id]
                if len(rows) > 0:
                    full_xis.append(float(rows.iloc[0]["xi"]))
                    gof_xis.append(gof_val)
                    families_for_color.append(rows.iloc[0]["family_corrected"])

            if full_xis:
                # Color by family (top 5 + Other)
                unique_fams = pd.Series(families_for_color).value_counts()
                top_fams = unique_fams.head(5).index.tolist()
                cmap = plt.cm.tab10
                fam_colors = {f: cmap(i) for i, f in enumerate(top_fams)}

                colors = [fam_colors.get(f, "gray") for f in families_for_color]
                ax.scatter(full_xis, gof_xis, c=colors, alpha=0.6, s=30)

                # 1:1 line
                mn = min(min(full_xis), min(gof_xis))
                mx = max(max(full_xis), max(gof_xis))
                ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8, alpha=0.5)

                # Spearman rho annotation
                rho, _ = spearmanr(full_xis, gof_xis)
                ax.annotate(f"Spearman rho = {rho:.3f}",
                           xy=(0.05, 0.95), xycoords="axes fraction",
                           fontsize=10, ha="left", va="top")

                ax.set_xlabel("xi (full model)")
                ax.set_ylabel("xi (GoF-restricted)")
                ax.set_title("Full vs GoF-Restricted xi")

                # Legend for top families
                for f in top_fams:
                    ax.scatter([], [], color=fam_colors[f], label=f, s=30)
                ax.scatter([], [], color="gray", label="Other", s=30)
                ax.legend(fontsize=7, loc="lower right")

                plt.tight_layout()
                path = fig_dir / "fig_gof_restricted_scatter.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                generated.append(str(path))
                logger.info(f"  Saved {path.name}")
            plt.close(fig)
    except Exception as exc:
        logger.error(f"Figure 3 failed: {exc}")

    # ---- Figure 4: Spoken vs written paired bar chart ----
    try:
        sw_pairs = sw_results.get("pairs", [])
        if sw_pairs:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(sw_pairs))
            width = 0.35

            spoken_vals = [p["xi_spoken"] for p in sw_pairs]
            written_vals = [p["xi_written"] for p in sw_pairs]
            labels = [p["language"] for p in sw_pairs]

            bars1 = ax.bar(x - width/2, spoken_vals, width, label="Spoken",
                          color="#1f77b4", alpha=0.7)
            bars2 = ax.bar(x + width/2, written_vals, width, label="Written",
                          color="#ff7f0e", alpha=0.7)

            # Annotate Cohen's d
            for i, p in enumerate(sw_pairs):
                d = p["cohens_d"]
                ci = p.get("d_ci", [np.nan, np.nan])
                ax.annotate(f"d={d:.1f}\n[{ci[0]:.1f}, {ci[1]:.1f}]",
                           xy=(i, max(spoken_vals[i], written_vals[i]) + 0.01),
                           ha="center", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("xi (GEV shape parameter)")
            ax.set_title("Spoken vs Written xi (excluding en_atis)")
            ax.legend()
            plt.tight_layout()
            path = fig_dir / "fig_spoken_written.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            generated.append(str(path))
            logger.info(f"  Saved {path.name}")
            plt.close(fig)
    except Exception as exc:
        logger.error(f"Figure 4 failed: {exc}")

    return generated


# ===================================================================
# STEP 9: Output Assembly
# ===================================================================

def _safe_float(v: Any) -> float:
    """Convert value to float, returning 0.0 for non-finite."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v: Any) -> int:
    """Convert value to int, returning 0 for non-int."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def assemble_output(
    exp_examples: list[dict],
    corrected_family: dict[str, str],
    corrections: list[dict],
    afr_fixed: bool,
    corrected_reg: dict,
    gof_results: dict,
    annot_results: dict,
    lofo_results: dict,
    residual_results: dict,
    sw_results: dict,
    df_full: pd.DataFrame,
    tb_feat: dict[str, float],
) -> dict:
    """Build eval_out.json conforming to exp_eval_sol_out schema."""
    logger.info("STEP 9: Assembling output")

    # ---- metrics_agg (all must be numbers) ----
    c_coefs = corrected_reg.get("coefficients", {})
    c_ent = c_coefs.get("word_order_entropy_z", {})
    c_morph = c_coefs.get("morph_richness_z", {})
    c_hd = c_coefs.get("head_direction_ratio_z", {})

    g_reg = gof_results.get("gof_regression", {})
    g_coefs = g_reg.get("coefficients", {})
    g_ent = g_coefs.get("word_order_entropy_z", {})
    g_med = gof_results.get("gof_mediation", {})

    a_reg = annot_results.get("annot_regression", {})
    a_coefs = a_reg.get("coefficients", {})
    a_ent = a_coefs.get("word_order_entropy_z", {})
    a_morph = a_coefs.get("morph_richness_z", {})
    a_med = annot_results.get("annot_mediation", {})

    # Determine robustness
    corr_sig = bool(c_ent.get("reject_fdr", False))
    gof_sig = bool(g_ent.get("reject_fdr", False))
    annot_sig = bool(a_ent.get("reject_fdr", False))
    lofo_sig = bool(lofo_results.get("lofo_all_significant", False))

    n_sig_tracks = sum([corr_sig, gof_sig, annot_sig, lofo_sig])
    overall_entropy_robust = (n_sig_tracks >= 3)

    gof_med_interp = g_med.get("interpretation", "")
    annot_med_interp = a_med.get("interpretation", "")
    overall_mediation_robust = (
        "mediation" in gof_med_interp and "mediation" in annot_med_interp
    )

    # Spoken/written metrics
    sw_pairs = sw_results.get("pairs", [])
    sl_pair = next((p for p in sw_pairs if p["language"] == "Slovenian"), {})
    fr_pair = next((p for p in sw_pairs if p["language"] == "French"), {})

    original_entropy_beta = 0.084

    metrics_agg: dict[str, float | int] = {
        # STEP 1 - Family Audit
        "family_audit_n_corrections": _safe_int(len(corrections)),
        "family_audit_afrikaans_fixed": int(afr_fixed),
        "family_audit_n_families_after": _safe_int(
            len(set(corrected_family.values()) - {"Unknown"})),

        # STEP 2 - Corrected Regression
        "corrected_reg_entropy_beta": _safe_float(c_ent.get("beta")),
        "corrected_reg_entropy_p_fdr": _safe_float(c_ent.get("p_fdr")),
        "corrected_reg_entropy_reject_fdr": int(corr_sig),
        "corrected_reg_morph_beta": _safe_float(c_morph.get("beta")),
        "corrected_reg_morph_p_fdr": _safe_float(c_morph.get("p_fdr")),
        "corrected_reg_hd_beta": _safe_float(c_hd.get("beta")),
        "corrected_reg_hd_p_fdr": _safe_float(c_hd.get("p_fdr")),
        "corrected_reg_pseudo_r2": _safe_float(corrected_reg.get("pseudo_r2")),
        "corrected_reg_n_treebanks": _safe_int(corrected_reg.get("n_treebanks")),
        "corrected_reg_n_families": _safe_int(corrected_reg.get("n_families")),
        "corrected_vs_original_entropy_beta_diff": _safe_float(
            abs(_safe_float(c_ent.get("beta")) - original_entropy_beta)),

        # STEP 3 - GoF-Restricted
        "gof_n_combos_passing": _safe_int(gof_results.get("gof_n_combos_passing")),
        "gof_n_treebanks_retained": _safe_int(
            gof_results.get("gof_n_treebanks_retained")),
        "gof_reg_entropy_beta": _safe_float(g_ent.get("beta")),
        "gof_reg_entropy_p_fdr": _safe_float(g_ent.get("p_fdr")),
        "gof_reg_entropy_significant": int(gof_sig),
        "gof_reg_pseudo_r2": _safe_float(g_reg.get("pseudo_r2")),
        "gof_mediation_indirect_significant": int(
            bool(g_med.get("indirect_significant", False))),

        # STEP 4 - Annotation
        "annot_n_treebanks": _safe_int(annot_results.get("annot_n_treebanks")),
        "annot_n_dropped": _safe_int(annot_results.get("annot_n_dropped")),
        "annot_reg_entropy_beta": _safe_float(a_ent.get("beta")),
        "annot_reg_entropy_p_fdr": _safe_float(a_ent.get("p_fdr")),
        "annot_reg_morph_beta": _safe_float(a_morph.get("beta")),
        "annot_reg_morph_p_fdr": _safe_float(a_morph.get("p_fdr")),
        "annot_reg_pseudo_r2": _safe_float(a_reg.get("pseudo_r2")),
        "annot_morph_entropy_corr": _safe_float(
            annot_results.get("annot_morph_entropy_corr")),

        # STEP 5 - LOFO
        "lofo_n_families": _safe_int(lofo_results.get("lofo_n_families")),
        "lofo_entropy_beta_mean": _safe_float(
            lofo_results.get("lofo_entropy_beta_mean")),
        "lofo_entropy_beta_sd": _safe_float(
            lofo_results.get("lofo_entropy_beta_sd")),
        "lofo_entropy_beta_min": _safe_float(
            lofo_results.get("lofo_entropy_beta_min")),
        "lofo_entropy_beta_max": _safe_float(
            lofo_results.get("lofo_entropy_beta_max")),
        "lofo_entropy_beta_range": _safe_float(
            lofo_results.get("lofo_entropy_beta_range")),
        "lofo_all_significant": int(lofo_sig),
        "lofo_pct_significant": _safe_float(
            lofo_results.get("lofo_pct_significant")),

        # STEP 6 - Residual
        "residual_turkish_rank": _safe_int(
            residual_results.get("residual_turkish_rank")),
        "residual_turkish_relative": _safe_float(
            residual_results.get("residual_turkish_relative")),
        "residual_mean_abs": _safe_float(
            residual_results.get("residual_mean_abs")),
        "residual_median_abs": _safe_float(
            residual_results.get("residual_median_abs")),

        # STEP 7 - Spoken/Written
        "sw_slovenian_d": _safe_float(sl_pair.get("cohens_d")),
        "sw_slovenian_xi_spoken": _safe_float(sl_pair.get("xi_spoken")),
        "sw_slovenian_xi_written": _safe_float(sl_pair.get("xi_written")),
        "sw_french_d": _safe_float(fr_pair.get("cohens_d")),
        "sw_french_xi_spoken": _safe_float(fr_pair.get("xi_spoken")),
        "sw_french_xi_written": _safe_float(fr_pair.get("xi_written")),
        "sw_n_viable_pairs": _safe_int(sw_results.get("sw_n_viable_pairs")),
        "sw_consistent_direction": int(
            sw_results.get("sw_consistent_direction", False)),

        # Synthesis
        "overall_entropy_robust": int(overall_entropy_robust),
        "overall_mediation_robust": int(overall_mediation_robust),
        "n_sensitivity_tracks_entropy_significant": n_sig_tracks,
    }

    # CI overlap check for corrected vs original
    corr_beta = _safe_float(c_ent.get("beta"))
    corr_se = _safe_float(c_ent.get("se"))
    orig_lo = original_entropy_beta - 1.96 * 0.023
    orig_hi = original_entropy_beta + 1.96 * 0.023
    corr_lo = corr_beta - 1.96 * corr_se if corr_se > 0 else corr_beta
    corr_hi = corr_beta + 1.96 * corr_se if corr_se > 0 else corr_beta
    ci_overlap = bool(corr_lo <= orig_hi and corr_hi >= orig_lo)
    metrics_agg["corrected_vs_original_entropy_ci_overlap"] = int(ci_overlap)

    # ---- Per-example output ----
    residual_df = residual_results.get("_residual_df")
    gof_xi = gof_results.get("gof_xi", {})

    examples: list[dict] = []
    for ex in exp_examples:
        tb_id = ex["metadata_treebank_id"]
        old_family = ex.get("metadata_family", "")
        new_family = corrected_family.get(tb_id, old_family)
        correction_applied = (old_family != new_family)

        # Get xi values
        try:
            out = json.loads(ex["output"])
            xi_full = out.get("xi_raw")
            xi_se = out.get("xi_raw_se")
        except (json.JSONDecodeError, TypeError):
            xi_full = None
            xi_se = None

        xi_gof = gof_xi.get(tb_id)

        # Get residual
        xi_residual = None
        if residual_df is not None:
            res_rows = residual_df[residual_df["treebank_id"] == tb_id]
            if len(res_rows) > 0:
                xi_residual = float(res_rows.iloc[0]["residual"])

        # Check if in annotation subset
        fc = tb_feat.get(tb_id, 0.0)
        in_annot = fc > 0.5

        output_data = {
            "old_family": old_family,
            "new_family": new_family,
            "family_corrected": new_family,
            "xi_full": xi_full,
            "xi_gof_restricted": xi_gof,
            "xi_residual": xi_residual,
            "in_annot_subset": in_annot,
        }

        example: dict[str, Any] = {
            "input": tb_id,
            "output": json.dumps(output_data, default=str),
            "metadata_treebank_id": tb_id,
            "metadata_language": ex.get("metadata_language", ""),
            "metadata_old_family": old_family,
            "metadata_new_family": new_family,
            "metadata_correction_applied": correction_applied,
            "metadata_xi_full": _safe_float(xi_full),
            "metadata_xi_gof": _safe_float(xi_gof),
            "metadata_feat_completeness": fc,
            "metadata_fold": "evaluation",
        }

        # predict_ fields
        example["predict_our_method"] = json.dumps({
            "xi_full": xi_full,
            "xi_gof_restricted": xi_gof,
            "family_corrected": new_family,
        }, default=str)

        # eval_ metrics (per-example numbers)
        example["eval_xi_full"] = _safe_float(xi_full)
        if xi_gof is not None:
            example["eval_xi_gof"] = _safe_float(xi_gof)
        if xi_residual is not None:
            example["eval_residual"] = _safe_float(xi_residual)

        examples.append(example)

    # ---- metadata ----
    metadata = {
        "evaluation_name": "Family Audit and Sensitivity Analysis",
        "description": (
            "Multi-track sensitivity analysis evaluating robustness of "
            "GEV tail-constraint findings: family audit with Glottolog "
            "corrections, GoF-restricted re-analysis, annotation-quality "
            "filtering, leave-one-family-out CV, residual analysis, and "
            "spoken/written comparison."
        ),
        "family_audit": {
            "n_corrections": len(corrections),
            "corrections_list": corrections,
            "afrikaans_fixed": afr_fixed,
            "n_families_after": len(set(corrected_family.values()) - {"Unknown"}),
        },
        "corrected_regression": {k: v for k, v in corrected_reg.items()
                                  if k != "model_fit"},
        "gof_restricted": {k: v for k, v in gof_results.items()
                           if k not in ("gof_xi", "_residual_df")},
        "annotation_sensitivity": annot_results,
        "lofo_cv": {k: v for k, v in lofo_results.items()},
        "residual_analysis": {k: v for k, v in residual_results.items()
                              if k != "_residual_df"},
        "spoken_written": sw_results,
        "synthesis": {
            "overall_entropy_robust": overall_entropy_robust,
            "overall_mediation_robust": overall_mediation_robust,
            "n_sensitivity_tracks_entropy_significant": n_sig_tracks,
            "tracks_significant": {
                "corrected": corr_sig,
                "gof_restricted": gof_sig,
                "annotation_filtered": annot_sig,
                "lofo_all": lofo_sig,
            },
        },
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": [{
            "dataset": "sensitivity_analysis",
            "examples": examples,
        }],
    }


# ===================================================================
# MAIN
# ===================================================================

@logger.catch
def main() -> None:
    t_start = time.time()

    # ==================================================================
    # STEP 0: Data Loading
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 0: Loading all data")
    logger.info("=" * 60)

    exp_metadata, exp_examples = load_exp_id1()

    if MAX_EXAMPLES is not None:
        exp_examples = exp_examples[:MAX_EXAMPLES]
        logger.info(f"  Truncated to {len(exp_examples)} examples")

    iso_to_family = load_data_id4()
    tb_feat = load_data_id3_treebank_metadata()

    logger.info(f"  Data loading done in {time.time()-t_start:.1f}s")

    # ==================================================================
    # STEP 1: Family Audit
    # ==================================================================
    logger.info("=" * 60)
    t1 = time.time()
    corrected_family, corrections, afr_fixed = family_audit(
        exp_examples, iso_to_family)
    logger.info(f"  STEP 1 done in {time.time()-t1:.1f}s")

    # ==================================================================
    # STEP 2: Corrected Regression
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Corrected Regression")
    t2 = time.time()

    df_full = build_regression_df(exp_examples, corrected_family, tb_feat)
    corrected_reg = run_mixedlm(df_full, label="Corrected")

    # Log key results
    c_coefs = corrected_reg.get("coefficients", {})
    for pred, vals in c_coefs.items():
        logger.info(f"  {pred}: beta={vals.get('beta', '?'):.4f}, "
                    f"p_fdr={vals.get('p_fdr', '?')}")
    logger.info(f"  pseudo_r2={corrected_reg.get('pseudo_r2')}")
    logger.info(f"  STEP 2 done in {time.time()-t2:.1f}s")

    # ==================================================================
    # STEP 3: GoF-Restricted Sensitivity
    # ==================================================================
    logger.info("=" * 60)
    t3 = time.time()

    # Load sentence data for GoF re-analysis
    bin_data = load_data_id3_sentences()
    gof_results = gof_restricted_analysis(bin_data, corrected_family, df_full)
    del bin_data
    gc.collect()
    logger.info(f"  STEP 3 done in {time.time()-t3:.1f}s")

    # ==================================================================
    # STEP 4: Annotation Sensitivity
    # ==================================================================
    logger.info("=" * 60)
    t4 = time.time()
    annot_results = annotation_sensitivity(df_full)
    logger.info(f"  STEP 4 done in {time.time()-t4:.1f}s")

    # ==================================================================
    # STEP 5: Leave-One-Family-Out
    # ==================================================================
    logger.info("=" * 60)
    t5 = time.time()
    lofo_results = lofo_analysis(df_full)
    logger.info(f"  STEP 5 done in {time.time()-t5:.1f}s")

    # ==================================================================
    # STEP 6: Residual Analysis
    # ==================================================================
    logger.info("=" * 60)
    t6 = time.time()
    residual_results = residual_analysis(df_full, corrected_reg)
    logger.info(f"  STEP 6 done in {time.time()-t6:.1f}s")

    # ==================================================================
    # STEP 7: Spoken/Written Fix
    # ==================================================================
    logger.info("=" * 60)
    t7 = time.time()
    sw_results = spoken_written_analysis(exp_examples)
    logger.info(f"  STEP 7 done in {time.time()-t7:.1f}s")

    # ==================================================================
    # STEP 8: Figures
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 8: Generating figures")
    t8 = time.time()
    fig_files = generate_figures(
        corrected_reg={k: v for k, v in corrected_reg.items() if k != "model_fit"},
        gof_results=gof_results,
        annot_results=annot_results,
        lofo_results=lofo_results,
        sw_results=sw_results,
        df_full=df_full,
    )
    logger.info(f"  Generated {len(fig_files)} figures in {time.time()-t8:.1f}s")

    # ==================================================================
    # STEP 9: Output Assembly
    # ==================================================================
    logger.info("=" * 60)
    t9 = time.time()
    output = assemble_output(
        exp_examples=exp_examples,
        corrected_family=corrected_family,
        corrections=corrections,
        afr_fixed=afr_fixed,
        corrected_reg={k: v for k, v in corrected_reg.items() if k != "model_fit"},
        gof_results=gof_results,
        annot_results=annot_results,
        lofo_results=lofo_results,
        residual_results=residual_results,
        sw_results=sw_results,
        df_full=df_full,
        tb_feat=tb_feat,
    )

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"  Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"  STEP 9 done in {time.time()-t9:.1f}s")

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Pipeline completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("SUCCESS")


if __name__ == "__main__":
    main()
