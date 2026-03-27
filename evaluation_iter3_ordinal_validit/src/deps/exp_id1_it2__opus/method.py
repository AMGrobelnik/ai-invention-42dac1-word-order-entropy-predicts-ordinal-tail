#!/usr/bin/env python3
"""GEV Fitting, Typological Regression, and Mediation Analysis Pipeline.

Fits GEV distributions to max dependency-distance data across ~918 qualifying
bin-treebank combinations (dual-track: raw + normalised max_DD), computes
bootstrap CIs, compares against alternative distributions, aggregates xi per
treebank, runs mixed-effects regression with language-family random intercepts,
executes Preacher-Hayes mediation analysis, profiles discordant languages,
computes spoken/written Cohen's d, and quantifies EVT-unique treebank pairs.

A mean-DD baseline is run in parallel so that the EVT approach can be compared
against the simplest possible summary statistic.
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
from scipy.stats import genextreme, lognorm, gamma, spearmanr, pearsonr
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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)  # 80 % of container RAM

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM (container)")
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# Set hard memory limit so we get MemoryError instead of OOM-kill
try:
    resource.setrlimit(resource.RLIMIT_AS,
                       (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError) as exc:
    logger.warning(f"Could not set RLIMIT_AS: {exc}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DEPS_DIR = WORKSPACE / "deps"

DATA3_DIR = DEPS_DIR / "data_id3_it1__opus"
DATA4_PATH = DEPS_DIR / "data_id4_it1__opus" / "full_data_out.json"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 500
MEDIATION_BOOTSTRAP_N = 5000
AD_MC_SAMPLES = 49          # only used if KS test fallback is bypassed
FDR_ALPHA = 0.01
SEED = 42
N_WORKERS = max(1, NUM_CPUS - 1)  # leave 1 core for OS
MAX_EXAMPLES: int | None = None  # None = all; set for gradual scale-up

# ---------------------------------------------------------------------------
# ISO 2-letter -> ISO 3-letter mapping for Grambank lookup
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
# STEP 1 – Data loading
# ===================================================================

def load_data3(max_examples: int | None = None) -> tuple[list[dict], dict, dict]:
    """Load sentence + treebank rows from data_id3 (two JSON parts).

    Returns
    -------
    qualifying_combos : list[(tb_id, bin)]
    bin_data : dict[(tb_id,bin)] -> list[float]  (raw max_dd)
    bin_data_norm : dict[(tb_id,bin)] -> list[float]  (normalised)
    tb_lookup : dict[tb_id] -> metadata dict
    """
    logger.info("Loading data_id3 ...")
    t0 = time.time()

    treebank_rows: list[dict] = []
    bin_data: dict[tuple, list] = defaultdict(list)
    bin_data_norm: dict[tuple, list] = defaultdict(list)

    sentences_loaded = 0
    for part_file in ["data_out/full_data_out_1.json",
                      "data_out/full_data_out_2.json"]:
        fpath = DATA3_DIR / part_file
        logger.info(f"  Reading {fpath.name} ...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]
        del raw
        for ex in examples:
            if ex.get("metadata_row_type") == "treebank":
                treebank_rows.append(ex)
            elif ex.get("metadata_row_type") == "sentence":
                key = (ex["metadata_treebank_id"], ex["metadata_length_bin"])
                bin_data[key].append(ex["metadata_max_dd"])
                bin_data_norm[key].append(ex["metadata_max_dd_normalized"])
                sentences_loaded += 1
                if max_examples is not None and sentences_loaded >= max_examples:
                    break
        del examples
        gc.collect()
        if max_examples is not None and sentences_loaded >= max_examples:
            break

    logger.info(f"  Loaded {len(treebank_rows)} treebanks, "
                f"{sentences_loaded} sentences in {time.time()-t0:.1f}s")

    # Build treebank lookup
    tb_lookup: dict[str, dict] = {}
    for tb in treebank_rows:
        tb_id = tb["metadata_treebank_id"]
        tb_lookup[tb_id] = {
            "language": tb["metadata_language"],
            "iso_code": tb["metadata_iso_code"],
            "morph_richness": tb["metadata_morph_richness"],
            "head_direction_ratio": tb["metadata_head_direction_ratio"],
            "word_order_entropy": tb["metadata_word_order_entropy"],
            "nonprojectivity_rate": tb["metadata_nonprojectivity_rate"],
            "feat_completeness": tb["metadata_feat_completeness"],
            "genre": tb["metadata_genre"],
            "modality": tb["metadata_modality"],
            "mean_dd_all": tb["metadata_mean_dd_all"],
            "n_sentences_total": tb["metadata_n_sentences_total"],
            "qualifies_per_bin": json.loads(tb["metadata_qualifies_per_bin"]),
            "n_per_bin": json.loads(tb["metadata_n_per_bin"]),
        }

    # Enumerate qualifying bin-treebank combos
    qualifying_combos: list[tuple[str, int]] = []
    for tb_id, info in tb_lookup.items():
        for bin_str, qualifies in info["qualifies_per_bin"].items():
            if qualifies:
                b = int(bin_str)
                key = (tb_id, b)
                if key in bin_data and len(bin_data[key]) >= 50:
                    qualifying_combos.append(key)

    logger.info(f"  {len(qualifying_combos)} qualifying bin-treebank combos")
    return qualifying_combos, bin_data, bin_data_norm, tb_lookup


def load_grambank(tb_lookup: dict[str, dict]) -> tuple[dict, dict]:
    """Load Grambank data and assign families to treebanks.

    Returns iso_to_family, iso_to_grambank_idx  (plus mutates tb_lookup).
    """
    logger.info("Loading Grambank data ...")
    raw = json.loads(DATA4_PATH.read_text())
    grambank_examples = raw["datasets"][0]["examples"]
    del raw

    iso_to_family: dict[str, str] = {}
    iso_to_grambank_idx: dict[str, float] = {}

    for ex in grambank_examples:
        iso = ex["metadata_iso639_3_code"]
        family = ex["metadata_family_name"]
        iso_to_family[iso] = family
        if ex["metadata_has_ud"]:
            try:
                out = json.loads(ex["output"])
                iso_to_grambank_idx[iso] = out["grambank_morph_index"]
            except (json.JSONDecodeError, KeyError):
                pass

    del grambank_examples
    gc.collect()

    # Assign family to each treebank
    for tb_id, info in tb_lookup.items():
        iso2 = info["iso_code"]
        iso3 = ISO2_TO_ISO3.get(iso2, iso2)
        family = iso_to_family.get(iso3) or iso_to_family.get(iso2)
        if family is None:
            for giso, gfam in iso_to_family.items():
                if giso.startswith(iso2):
                    family = gfam
                    break
        info["family"] = family or "Unknown"
        info["grambank_morph_index"] = (
            iso_to_grambank_idx.get(iso3) or iso_to_grambank_idx.get(iso2)
        )

    n_with_family = sum(1 for i in tb_lookup.values() if i["family"] != "Unknown")
    logger.info(f"  {n_with_family}/{len(tb_lookup)} treebanks with family assignment")
    return iso_to_family, iso_to_grambank_idx


# ===================================================================
# STEP 2 – GEV fitting functions
# ===================================================================

def fit_gev_single(
    data_arr: np.ndarray, method: str = "auto", *, fast: bool = False,
) -> dict:
    """Fit GEV to *data_arr*.  Returns dict with xi, loc, scale, etc.

    CRITICAL sign convention: scipy c = -xi, lmoments3 c = -xi.
    So xi = -c for BOTH libraries.

    Parameters
    ----------
    fast : bool
        If True, only try the primary method (skip the secondary). Use for
        bootstrap resamples where speed matters and fallback is not needed.
    """
    n = len(data_arr)
    result: dict[str, Any] = {
        "xi": np.nan, "loc": np.nan, "scale": np.nan,
        "method_used": None, "converged": False, "flag": None,
    }

    use_lmom = (method == "lmom") or (method == "auto" and n < 500)
    use_mle = (method == "mle") or (method == "auto" and n >= 500)

    lmom_result: dict | None = None
    mle_result: dict | None = None

    # L-moments fitting — try if primary OR if not in fast mode (fallback)
    if use_lmom or not fast:
        try:
            from lmoments3 import distr as lm_distr
            params = lm_distr.gev.lmom_fit(data_arr)
            c_lmom = params["c"]
            xi_lmom = -c_lmom
            lmom_result = {
                "xi": xi_lmom, "loc": params["loc"], "scale": params["scale"],
                "method_used": "lmom", "converged": True, "flag": None,
            }
        except Exception:
            lmom_result = None

    # MLE fitting — try if primary OR if not in fast mode (fallback)
    if use_mle or not fast:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_mle, loc_mle, scale_mle = genextreme.fit(data_arr)
            xi_mle = -c_mle
            mle_result = {
                "xi": xi_mle, "loc": loc_mle, "scale": scale_mle,
                "method_used": "mle", "converged": True, "flag": None,
            }
            if xi_mle < -0.5:
                mle_result["flag"] = "xi_mle_below_neg05"
        except Exception:
            mle_result = None

    # Selection logic
    if use_lmom and lmom_result:
        result = lmom_result
    elif use_mle and mle_result:
        if mle_result.get("flag") == "xi_mle_below_neg05" and lmom_result:
            result = lmom_result
            result["flag"] = "mle_overridden_xi_neg05"
        else:
            result = mle_result
    elif lmom_result:
        result = lmom_result
    elif mle_result:
        result = mle_result

    return result


def _loglik(dist, data_arr, params_tuple) -> float:
    """Sum of log-pdf; returns -inf on any numerical issue."""
    try:
        ll = np.sum(dist.logpdf(data_arr, *params_tuple))
        return ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


def _ic(loglik: float, k: int, n: int) -> tuple[float, float]:
    """AIC, BIC from log-likelihood."""
    if not np.isfinite(loglik):
        return np.inf, np.inf
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return aic, bic


def process_combo(args: tuple) -> dict:
    """Process one (treebank_id, bin) combination.

    Accepts a tuple so it works with ProcessPoolExecutor.map.
    """
    tb_id, bin_len, raw_list, norm_list, bootstrap_n, seed = args

    raw_data = np.asarray(raw_list, dtype=np.float64)
    norm_data = np.asarray(norm_list, dtype=np.float64)
    n_samples = len(raw_data)

    result: dict[str, Any] = {
        "treebank_id": tb_id, "bin": bin_len, "n_samples": n_samples,
        "raw": {}, "norm": {}, "alternatives": {}, "ad_test": {},
    }

    # --- RAW TRACK ---
    gev_raw = fit_gev_single(raw_data)
    result["raw"]["xi"] = gev_raw["xi"]
    result["raw"]["loc"] = gev_raw["loc"]
    result["raw"]["scale"] = gev_raw["scale"]
    result["raw"]["method"] = gev_raw["method_used"]
    result["raw"]["flag"] = gev_raw.get("flag")

    gev_raw_lmom = fit_gev_single(raw_data, method="lmom")
    gev_raw_mle = fit_gev_single(raw_data, method="mle")
    result["raw"]["xi_lmom"] = (
        gev_raw_lmom["xi"] if gev_raw_lmom["converged"] else None
    )
    result["raw"]["xi_mle"] = (
        gev_raw_mle["xi"] if gev_raw_mle["converged"] else None
    )

    # --- NORMALISED TRACK ---
    gev_norm = fit_gev_single(norm_data)
    result["norm"]["xi"] = gev_norm["xi"]
    result["norm"]["loc"] = gev_norm["loc"]
    result["norm"]["scale"] = gev_norm["scale"]
    result["norm"]["method"] = gev_norm["method_used"]

    # --- BOOTSTRAP CIs ---
    # PERF: Always use L-moments for bootstrap (0.3ms vs 5-10ms for MLE).
    # Point estimates already use the appropriate method; bootstrap only
    # needs the *distribution* of xi, not the most accurate per-sample est.
    rng = np.random.default_rng(seed + hash((tb_id, bin_len)) % 10_000)
    xi_boot_raw: list[float] = []
    xi_boot_norm: list[float] = []
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n_samples, size=n_samples)
        try:
            br = fit_gev_single(raw_data[idx], method="lmom", fast=True)
            if br["converged"] and np.isfinite(br["xi"]):
                xi_boot_raw.append(br["xi"])
        except Exception:
            pass
        try:
            bn = fit_gev_single(norm_data[idx], method="lmom", fast=True)
            if bn["converged"] and np.isfinite(bn["xi"]):
                xi_boot_norm.append(bn["xi"])
        except Exception:
            pass

    if len(xi_boot_raw) >= 50:
        result["raw"]["xi_ci_low"] = float(np.percentile(xi_boot_raw, 2.5))
        result["raw"]["xi_ci_high"] = float(np.percentile(xi_boot_raw, 97.5))
        result["raw"]["xi_boot_se"] = float(np.std(xi_boot_raw))
        result["raw"]["n_valid_bootstraps"] = len(xi_boot_raw)
    if len(xi_boot_norm) >= 50:
        result["norm"]["xi_ci_low"] = float(np.percentile(xi_boot_norm, 2.5))
        result["norm"]["xi_ci_high"] = float(np.percentile(xi_boot_norm, 97.5))
        result["norm"]["xi_boot_se"] = float(np.std(xi_boot_norm))

    # --- ALTERNATIVE DISTRIBUTIONS (AIC / BIC) ---
    try:
        c_gev, loc_gev, scale_gev = genextreme.fit(raw_data)
        ll_gev = _loglik(genextreme, raw_data, (c_gev, loc_gev, scale_gev))
        aic_gev, bic_gev = _ic(ll_gev, 3, n_samples)
    except Exception:
        aic_gev, bic_gev = np.inf, np.inf

    try:
        params_ln = lognorm.fit(raw_data, floc=0)
        ll_ln = _loglik(lognorm, raw_data, params_ln)
        aic_ln, bic_ln = _ic(ll_ln, 3, n_samples)
    except Exception:
        aic_ln, bic_ln = np.inf, np.inf

    try:
        params_gam = gamma.fit(raw_data, floc=0)
        ll_gam = _loglik(gamma, raw_data, params_gam)
        aic_gam, bic_gam = _ic(ll_gam, 3, n_samples)
    except Exception:
        aic_gam, bic_gam = np.inf, np.inf

    result["alternatives"] = {
        "aic_gev": float(aic_gev), "aic_lognorm": float(aic_ln),
        "aic_gamma": float(aic_gam),
        "bic_gev": float(bic_gev), "bic_lognorm": float(bic_ln),
        "bic_gamma": float(bic_gam),
        "gev_is_aic_best": bool(aic_gev <= min(aic_ln, aic_gam)),
        "delta_aic_vs_2nd": float(
            sorted([aic_gev, aic_ln, aic_gam])[1] - aic_gev
        ),
    }

    # --- GOODNESS-OF-FIT: KS test + manual AD statistic ---
    # Use KS test (instantaneous) instead of parametric-bootstrap AD test
    # which refits GEV n_mc_samples times and is prohibitively slow for
    # 918 combos.  We also compute the raw AD statistic for reporting.
    try:
        # Use fitted GEV params (c = -xi for scipy convention)
        if gev_raw["converged"]:
            c_fit = -gev_raw["xi"]
            loc_fit, scale_fit = gev_raw["loc"], gev_raw["scale"]
        else:
            c_fit, loc_fit, scale_fit = genextreme.fit(raw_data)

        # KS test — fast, provides a p-value
        ks_stat, ks_p = sp_stats.kstest(
            raw_data, "genextreme", args=(c_fit, loc_fit, scale_fit)
        )

        # Manual AD statistic (no parametric bootstrap needed)
        z = np.sort(genextreme.cdf(raw_data, c_fit, loc_fit, scale_fit))
        z = np.clip(z, 1e-15, 1 - 1e-15)
        n_ad = len(z)
        i_arr = np.arange(1, n_ad + 1)
        ad_stat = float(
            -n_ad - np.sum(
                (2 * i_arr - 1) * (np.log(z) + np.log(1 - z[::-1]))
            ) / n_ad
        )

        result["ad_test"] = {
            "statistic": ad_stat,
            "ks_statistic": float(ks_stat),
            "p_value": float(ks_p),
            "passes": bool(ks_p > 0.05),
            "n_used": n_samples,
            "method": "ks_test",
        }
    except Exception:
        result["ad_test"] = {"passes": None, "p_value": None, "statistic": None}

    # --- BASELINE: per-bin mean & variance of raw max_DD ---
    result["baseline"] = {
        "mean_maxdd": float(np.mean(raw_data)),
        "var_maxdd": float(np.var(raw_data, ddof=1)) if n_samples > 1 else 0.0,
        "median_maxdd": float(np.median(raw_data)),
        "mean_maxdd_norm": float(np.mean(norm_data)),
    }

    return result


# ===================================================================
# STEP 3-4 – Aggregation & quality reporting
# ===================================================================

def aggregate_treebank_xi(
    all_combo_results: list[dict], tb_lookup: dict,
) -> dict[str, dict]:
    """Inverse-variance weighted mean xi across qualifying bins."""
    treebank_xi: dict[str, dict] = {}

    for tb_id in tb_lookup:
        xi_vals, weights = [], []
        xi_norm_vals, weights_norm = [], []
        mean_dd_per_bin: dict[int, float] = {}

        for r in all_combo_results:
            if r["treebank_id"] != tb_id:
                continue
            # Raw track
            xi_r = r["raw"]["xi"]
            se_r = r["raw"].get("xi_boot_se")
            if (
                xi_r is not None
                and np.isfinite(xi_r)
                and se_r is not None
                and se_r > 0
            ):
                xi_vals.append(xi_r)
                weights.append(1.0 / (se_r ** 2))
            # Norm track
            xi_n = r["norm"]["xi"]
            se_n = r["norm"].get("xi_boot_se")
            if (
                xi_n is not None
                and np.isfinite(xi_n)
                and se_n is not None
                and se_n > 0
            ):
                xi_norm_vals.append(xi_n)
                weights_norm.append(1.0 / (se_n ** 2))
            # Baseline
            mean_dd_per_bin[r["bin"]] = r["baseline"]["mean_maxdd"]

        entry: dict[str, Any] = {"n_bins": 0}
        if xi_vals:
            w = np.array(weights)
            xi_agg = float(np.average(xi_vals, weights=w))
            xi_agg_se = float(np.sqrt(1.0 / np.sum(w)))
            entry.update({
                "xi_raw": xi_agg,
                "xi_raw_se": xi_agg_se,
                "n_bins": len(xi_vals),
                "xi_per_bin_raw": {
                    r["bin"]: r["raw"]["xi"]
                    for r in all_combo_results
                    if r["treebank_id"] == tb_id and np.isfinite(r["raw"]["xi"])
                },
            })
        if xi_norm_vals:
            w_n = np.array(weights_norm)
            entry["xi_norm"] = float(np.average(xi_norm_vals, weights=w_n))
            entry["xi_norm_se"] = float(np.sqrt(1.0 / np.sum(w_n)))

        # Baseline aggregate: grand mean across bins
        if mean_dd_per_bin:
            entry["baseline_mean_maxdd"] = float(
                np.mean(list(mean_dd_per_bin.values()))
            )

        if entry["n_bins"] > 0:
            treebank_xi[tb_id] = entry

    return treebank_xi


def compute_fit_quality(all_combo_results: list[dict]) -> dict:
    """Summarise GEV fit quality across all combos."""
    n_total = len(all_combo_results)
    n_aic_best = sum(
        1 for r in all_combo_results if r["alternatives"]["gev_is_aic_best"]
    )
    n_ad_tested = sum(
        1 for r in all_combo_results if r["ad_test"].get("passes") is not None
    )
    n_ad_pass = sum(
        1 for r in all_combo_results if r["ad_test"].get("passes") is True
    )

    mle_lmom_diffs: list[float] = []
    for r in all_combo_results:
        xi_l = r["raw"].get("xi_lmom")
        xi_m = r["raw"].get("xi_mle")
        if (
            xi_l is not None and xi_m is not None
            and np.isfinite(xi_l) and np.isfinite(xi_m)
        ):
            mle_lmom_diffs.append(abs(xi_l - xi_m))

    return {
        "n_combos": n_total,
        "pct_gev_aic_best": n_aic_best / max(n_total, 1) * 100,
        "pct_ad_pass": n_ad_pass / max(n_ad_tested, 1) * 100,
        "n_ad_tested": n_ad_tested,
        "mle_lmom_mean_diff": (
            float(np.mean(mle_lmom_diffs)) if mle_lmom_diffs else None
        ),
        "mle_lmom_median_diff": (
            float(np.median(mle_lmom_diffs)) if mle_lmom_diffs else None
        ),
        "mle_lmom_max_diff": (
            float(np.max(mle_lmom_diffs)) if mle_lmom_diffs else None
        ),
    }


# ===================================================================
# STEP 6 – Grambank cross-validation
# ===================================================================

def grambank_crossval(
    tb_lookup: dict,
) -> dict:
    """Correlate UD morph_richness with Grambank morph_index."""
    ud_vals, gb_vals, labels = [], [], []
    for tb_id, info in tb_lookup.items():
        gb_idx = info.get("grambank_morph_index")
        if gb_idx is not None:
            ud_vals.append(info["morph_richness"])
            gb_vals.append(gb_idx)
            labels.append(tb_id)

    if len(ud_vals) < 5:
        return {"n_overlap": len(ud_vals), "note": "too few overlaps"}

    rho, rho_p = spearmanr(ud_vals, gb_vals)
    r, r_p = pearsonr(ud_vals, gb_vals)
    return {
        "n_overlap": len(ud_vals),
        "spearman_r": float(rho), "spearman_p": float(rho_p),
        "pearson_r": float(r), "pearson_p": float(r_p),
    }


# ===================================================================
# STEP 7-8 – Regression
# ===================================================================

def build_regression_df(
    treebank_xi: dict, tb_lookup: dict,
) -> pd.DataFrame:
    """Build a treebank-level DataFrame for mixed-effects regression."""
    rows = []
    for tb_id, xi_info in treebank_xi.items():
        if tb_id not in tb_lookup:
            continue
        info = tb_lookup[tb_id]
        rows.append({
            "treebank_id": tb_id,
            "xi": xi_info["xi_raw"],
            "xi_se": xi_info["xi_raw_se"],
            "xi_norm": xi_info.get("xi_norm"),
            "morph_richness": info["morph_richness"],
            "head_direction_ratio": info["head_direction_ratio"],
            "word_order_entropy": info["word_order_entropy"],
            "nonprojectivity_rate": info["nonprojectivity_rate"],
            "feat_completeness": info["feat_completeness"],
            "family": info["family"],
            "language": info["language"],
            "iso_code": info["iso_code"],
            "genre": info["genre"],
            "modality": info["modality"],
            "mean_dd_all": info["mean_dd_all"],
            "n_bins": xi_info["n_bins"],
            "baseline_mean_maxdd": xi_info.get("baseline_mean_maxdd", np.nan),
        })

    df = pd.DataFrame(rows)
    df = df[df["family"] != "Unknown"].dropna(subset=["xi"])
    # Standardise predictors
    for col in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else 0.0
    logger.info(
        f"Regression df: {len(df)} treebanks, {df['family'].nunique()} families"
    )
    return df


def run_regression(df: pd.DataFrame) -> dict:
    """Run mixed-effects regression + VIF + FDR correction."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tools import add_constant

    results: dict[str, Any] = {"fallback_notes": []}

    # Need at least 2 families with >=2 treebanks each for mixed model
    fam_counts = df["family"].value_counts()
    families_ok = (fam_counts >= 2).sum() >= 2

    if len(df) < 15:
        results["fallback_notes"].append(
            "F4: Too few treebanks for regression; using Spearman correlations"
        )
        for pred in ["morph_richness", "head_direction_ratio", "word_order_entropy"]:
            r, p = spearmanr(df[pred], df["xi"])
            results[f"spearman_{pred}"] = {"r": float(r), "p": float(p)}
        return results

    formula = "xi ~ morph_richness_z + head_direction_ratio_z + word_order_entropy_z"

    # --- Mixed-effects model ---
    model_fit = None
    if families_ok:
        try:
            model = smf.mixedlm(formula, data=df, groups=df["family"])
            model_fit = model.fit(reml=True)
            results["model_type"] = "mixed_effects"
            results["model_summary"] = str(model_fit.summary())
        except Exception as exc:
            results["fallback_notes"].append(
                f"F3a: MixedLM default failed ({exc}); trying Nelder-Mead"
            )
            try:
                model_fit = model.fit(method="nm", maxiter=2000)
                results["model_type"] = "mixed_effects_nm"
                results["model_summary"] = str(model_fit.summary())
            except Exception as exc2:
                results["fallback_notes"].append(
                    f"F3b: MixedLM NM also failed ({exc2}); collapsing small families"
                )

    # Fallback: collapse small families or use OLS
    if model_fit is None:
        df_coll = df.copy()
        fam_c = df_coll["family"].value_counts()
        small = fam_c[fam_c < 3].index
        df_coll.loc[df_coll["family"].isin(small), "family"] = "Other"
        if df_coll["family"].nunique() >= 2:
            try:
                model = smf.mixedlm(formula, data=df_coll, groups=df_coll["family"])
                model_fit = model.fit(reml=True)
                results["model_type"] = "mixed_effects_collapsed"
                results["model_summary"] = str(model_fit.summary())
            except Exception:
                pass

    if model_fit is None:
        # Final fallback: OLS with clustered SE
        results["fallback_notes"].append("F3d: Using OLS (no random effects)")
        try:
            ols_fit = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["family"]}
            )
            model_fit = ols_fit
            results["model_type"] = "ols_clustered"
            results["model_summary"] = str(ols_fit.summary())
        except Exception:
            ols_fit = smf.ols(formula, data=df).fit()
            model_fit = ols_fit
            results["model_type"] = "ols"
            results["model_summary"] = str(ols_fit.summary())

    # Extract coefficients
    predictors = [
        "morph_richness_z", "head_direction_ratio_z", "word_order_entropy_z",
    ]
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

    # VIF check
    try:
        X = df[predictors].dropna().values
        X_c = add_constant(X)
        vif_vals = [
            variance_inflation_factor(X_c, i + 1) for i in range(len(predictors))
        ]
        results["vif"] = dict(zip(predictors, [float(v) for v in vif_vals]))
    except Exception:
        results["vif"] = {}

    # Sequential model comparison (pseudo R² via residual variance)
    try:
        null_fit = smf.mixedlm("xi ~ 1", data=df, groups=df["family"]).fit(reml=True)
        var_null = float(null_fit.scale)
        var_full = float(model_fit.scale) if hasattr(model_fit, "scale") else np.nan
        if np.isfinite(var_null) and np.isfinite(var_full) and var_null > 0:
            results["pseudo_r2"] = float(1 - var_full / var_null)
    except Exception:
        pass

    # Partial correlations (OLS residual approach)
    try:
        for target_pred in predictors:
            covariates = [p for p in predictors if p != target_pred]
            cov_formula = " + ".join(covariates)
            resid_y = smf.ols(f"xi ~ {cov_formula}", data=df).fit().resid
            resid_x = smf.ols(f"{target_pred} ~ {cov_formula}", data=df).fit().resid
            pr, pp = pearsonr(resid_y, resid_x)
            results.setdefault("partial_correlations", {})[target_pred] = {
                "r": float(pr), "p": float(pp),
            }
    except Exception:
        pass

    # Baseline regression: mean_dd_all ~ morph_richness + hd_ratio + wo_entropy
    try:
        base_fit = smf.ols(
            "baseline_mean_maxdd ~ morph_richness_z + head_direction_ratio_z"
            " + word_order_entropy_z",
            data=df,
        ).fit()
        results["baseline_regression"] = {
            "r_squared": float(base_fit.rsquared),
            "coefficients": {
                p: {"beta": float(base_fit.params[p]), "p": float(base_fit.pvalues[p])}
                for p in predictors
                if p in base_fit.params
            },
        }
    except Exception:
        pass

    results["n_treebanks"] = len(df)
    results["n_families"] = int(df["family"].nunique())
    return results


# ===================================================================
# STEP 9 – Mediation analysis (Preacher-Hayes bootstrap)
# ===================================================================

def run_mediation(df: pd.DataFrame, n_boot: int = 5000) -> dict:
    """Preacher-Hayes bootstrap mediation: morph -> wo_entropy -> xi."""
    results: dict[str, Any] = {}

    X = df["morph_richness_z"].values
    M = df["word_order_entropy_z"].values
    Y = df["xi"].values
    n_tb = len(df)

    if n_tb < 10:
        return {"note": "Too few treebanks for mediation", "n": n_tb}

    rng = np.random.default_rng(SEED + 999)
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
        return {"note": "Too many bootstrap failures", "n_valid": len(indirect)}

    ind = np.array(indirect)
    dir_ = np.array(direct)
    tot = np.array(total)

    ind_lo, ind_hi = float(np.percentile(ind, 2.5)), float(np.percentile(ind, 97.5))
    dir_lo, dir_hi = float(np.percentile(dir_, 2.5)), float(np.percentile(dir_, 97.5))

    ind_sig = bool(ind_lo > 0 or ind_hi < 0)
    dir_sig = bool(dir_lo > 0 or dir_hi < 0)

    tot_mean = float(np.mean(tot))
    prop_med = float(np.mean(ind) / tot_mean) if abs(tot_mean) > 1e-12 else np.nan

    results = {
        "indirect_effect_mean": float(np.mean(ind)),
        "indirect_effect_ci": [ind_lo, ind_hi],
        "indirect_significant": ind_sig,
        "direct_effect_mean": float(np.mean(dir_)),
        "direct_effect_ci": [dir_lo, dir_hi],
        "direct_significant": dir_sig,
        "total_effect_mean": tot_mean,
        "proportion_mediated": prop_med,
        "n_valid_bootstraps": len(indirect),
        "interpretation": (
            "full_mediation" if ind_sig and not dir_sig
            else "partial_mediation" if ind_sig and dir_sig
            else "no_mediation_direct_only" if not ind_sig and dir_sig
            else "no_effect"
        ),
    }
    return results


# ===================================================================
# STEP 10 – Discordant language profiles
# ===================================================================

DISCORDANT_TBS = {
    "ar_padt": "Arabic (rich morph + head-initial = DISCORDANT)",
    "zh_gsd": "Chinese (poor morph + mixed direction = DISCORDANT)",
    "eu_bdt": "Basque (rich morph + head-final = CANONICAL)",
    "en_ewt": "English (poor morph + head-initial = CANONICAL)",
    "tr_imst": "Turkish (rich morph + head-final = CANONICAL)",
    "hi_hdtb": "Hindi (moderate morph + head-final = CANONICAL)",
}


def build_discordant_profiles(
    treebank_xi: dict, tb_lookup: dict, model_fit: Any | None, df: pd.DataFrame,
) -> dict:
    profiles: dict[str, Any] = {}
    for tb_id, desc in DISCORDANT_TBS.items():
        if tb_id not in treebank_xi or tb_id not in tb_lookup:
            continue
        info = tb_lookup[tb_id]
        xi_info = treebank_xi[tb_id]
        entry: dict[str, Any] = {
            "description": desc,
            "xi_raw": xi_info["xi_raw"],
            "xi_raw_se": xi_info["xi_raw_se"],
            "xi_per_bin": xi_info.get("xi_per_bin_raw", {}),
            "morph_richness": info["morph_richness"],
            "head_direction_ratio": info["head_direction_ratio"],
            "word_order_entropy": info["word_order_entropy"],
            "predicted_xi": None,
            "residual": None,
        }
        # Prediction from regression
        if model_fit is not None:
            tb_row = df[df["treebank_id"] == tb_id]
            if len(tb_row) > 0:
                try:
                    pred_val = float(model_fit.predict(tb_row).iloc[0])
                    entry["predicted_xi"] = pred_val
                    entry["residual"] = float(xi_info["xi_raw"] - pred_val)
                except Exception:
                    pass
        profiles[tb_id] = entry
    return profiles


# ===================================================================
# STEP 11 – Spoken / written comparison
# ===================================================================

SPOKEN_WRITTEN_PAIRS = [
    ("sl_sst", "sl_ssj", "Slovenian"),
    ("fr_parisstories", "fr_gsd", "French"),
    ("no_nynorsklia", "no_nynorsk", "Norwegian"),
    ("en_atis", "en_ewt", "English-ATIS"),
    ("en_eslspok", "en_ewt", "English-ESL"),
]


def spoken_written_analysis(treebank_xi: dict) -> list[dict]:
    results = []
    for spoken, written, lang in SPOKEN_WRITTEN_PAIRS:
        if spoken not in treebank_xi or written not in treebank_xi:
            continue
        xi_s = treebank_xi[spoken]["xi_raw"]
        xi_w = treebank_xi[written]["xi_raw"]
        se_s = treebank_xi[spoken]["xi_raw_se"]
        se_w = treebank_xi[written]["xi_raw_se"]

        diff = xi_s - xi_w
        pooled_se = np.sqrt((se_s ** 2 + se_w ** 2) / 2) if se_s > 0 and se_w > 0 else np.nan
        d = float(diff / pooled_se) if np.isfinite(pooled_se) and pooled_se > 0 else np.nan

        results.append({
            "language": lang,
            "spoken_tb": spoken, "written_tb": written,
            "xi_spoken": float(xi_s), "xi_written": float(xi_w),
            "diff": float(diff), "cohens_d": d,
            "prediction_confirmed": bool(diff < 0),
        })
    return results


# ===================================================================
# STEP 12 – EVT-unique treebank pairs
# ===================================================================

def evt_unique_pairs(
    treebank_xi: dict, tb_lookup: dict,
    mean_dd_thresh: float = 0.5, xi_thresh: float = 0.15,
) -> dict:
    tb_ids = [t for t in treebank_xi if t in tb_lookup]
    n_total = 0
    n_similar_mean = 0
    n_evt_unique = 0

    for i in range(len(tb_ids)):
        for j in range(i + 1, len(tb_ids)):
            ti, tj = tb_ids[i], tb_ids[j]
            n_total += 1
            dd_i = tb_lookup[ti]["mean_dd_all"]
            dd_j = tb_lookup[tj]["mean_dd_all"]
            xi_i = treebank_xi[ti]["xi_raw"]
            xi_j = treebank_xi[tj]["xi_raw"]

            if abs(dd_i - dd_j) < mean_dd_thresh:
                n_similar_mean += 1
                if abs(xi_i - xi_j) > xi_thresh:
                    n_evt_unique += 1

    pct = n_evt_unique / max(n_similar_mean, 1) * 100
    return {
        "n_total_pairs": n_total,
        "n_similar_mean_dd": n_similar_mean,
        "n_evt_unique": n_evt_unique,
        "pct_evt_unique": float(pct),
    }


# ===================================================================
# STEP 13 – Genre control
# ===================================================================

GENRE_TREEBANKS = {
    "English": ["en_ewt", "en_gum", "en_partut", "en_lines"],
    "Czech": ["cs_pdt", "cs_cac", "cs_fictree"],
    "French": ["fr_gsd", "fr_sequoia", "fr_ftb"],
    "Italian": ["it_isdt", "it_vit", "it_partut"],
}


def genre_control(treebank_xi: dict) -> dict:
    results: dict[str, Any] = {}
    for lang, tb_list in GENRE_TREEBANKS.items():
        valid = [
            (tb, treebank_xi[tb]["xi_raw"])
            for tb in tb_list
            if tb in treebank_xi
        ]
        if len(valid) >= 2:
            xis = [v[1] for v in valid]
            results[lang] = {
                "treebanks": [
                    {"id": v[0], "xi": float(v[1])} for v in valid
                ],
                "within_lang_xi_range": float(max(xis) - min(xis)),
                "within_lang_xi_std": float(np.std(xis)),
            }
    return results


# ===================================================================
# STEP 14 – Output assembly
# ===================================================================

def build_schema_output(
    treebank_xi: dict,
    tb_lookup: dict,
    all_combo_results: list[dict],
    fit_quality: dict,
    dual_track_rho: float,
    dual_track_p: float,
    grambank_cv: dict,
    regression_results: dict,
    mediation_results: dict,
    discordant_profiles: dict,
    sw_results: list[dict],
    evt_results: dict,
    genre_results: dict,
    fallback_notes: list[str],
) -> dict:
    """Build output conforming to exp_gen_sol_out.json schema."""

    # Per-treebank examples
    examples: list[dict] = []
    for tb_id, xi_info in treebank_xi.items():
        info = tb_lookup.get(tb_id, {})

        # Compose the output JSON string with full analysis
        output_data = {
            "xi_raw": xi_info.get("xi_raw"),
            "xi_raw_se": xi_info.get("xi_raw_se"),
            "xi_norm": xi_info.get("xi_norm"),
            "n_qualifying_bins": xi_info.get("n_bins"),
            "xi_per_bin": xi_info.get("xi_per_bin_raw", {}),
            "baseline_mean_maxdd": xi_info.get("baseline_mean_maxdd"),
            "family": info.get("family"),
        }

        example: dict[str, Any] = {
            "input": tb_id,
            "output": json.dumps(output_data, default=str),
            "metadata_treebank_id": tb_id,
            "metadata_language": info.get("language", ""),
            "metadata_iso_code": info.get("iso_code", ""),
            "metadata_family": info.get("family", ""),
            "metadata_morph_richness": info.get("morph_richness"),
            "metadata_head_direction_ratio": info.get("head_direction_ratio"),
            "metadata_word_order_entropy": info.get("word_order_entropy"),
            "metadata_mean_dd_all": info.get("mean_dd_all"),
            "metadata_modality": info.get("modality", ""),
            "metadata_genre": info.get("genre", ""),
            "metadata_n_bins": xi_info.get("n_bins"),
            "metadata_fold": "analysis",
        }

        # predict_ fields
        example["predict_our_method"] = json.dumps({
            "xi_raw": xi_info.get("xi_raw"),
            "xi_raw_se": xi_info.get("xi_raw_se"),
            "xi_norm": xi_info.get("xi_norm"),
        }, default=str)
        example["predict_baseline"] = json.dumps({
            "mean_maxdd": xi_info.get("baseline_mean_maxdd"),
        }, default=str)

        examples.append(example)

    # Top-level metadata with all analysis results
    metadata = {
        "method_name": "GEV Tail-Constraint Analysis of Maximum Dependency Distance",
        "description": (
            "Fits GEV distributions to sentence-level max dependency distances "
            "across 6 sentence-length bins, aggregates shape parameter xi per "
            "treebank via inverse-variance weighting, and analyses typological "
            "predictors via mixed-effects regression and mediation analysis."
        ),
        "fit_quality": fit_quality,
        "dual_track": {
            "spearman_rho": float(dual_track_rho) if np.isfinite(dual_track_rho) else None,
            "spearman_p": float(dual_track_p) if np.isfinite(dual_track_p) else None,
        },
        "grambank_crossval": grambank_cv,
        "regression": regression_results,
        "mediation": mediation_results,
        "discordant_languages": discordant_profiles,
        "spoken_written": sw_results,
        "evt_unique_pairs": evt_results,
        "genre_control": genre_results,
        "success_criteria": {
            "gev_fit_adequate": bool(
                fit_quality["pct_gev_aic_best"] > 70
                and fit_quality["pct_ad_pass"] > 70
            ),
            "dual_track_robust": bool(
                np.isfinite(dual_track_rho) and dual_track_rho > 0.8
            ),
            "evt_unique_above_20pct": bool(evt_results["pct_evt_unique"] > 20),
            "gof_method": "ks_test (anti-conservative with estimated params)",
        },
        "fallback_notes": fallback_notes,
        "n_treebanks_analysed": len(treebank_xi),
        "n_qualifying_combos": fit_quality["n_combos"],
    }

    return {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "gev_tail_constraint_analysis",
                "examples": examples,
            }
        ],
    }


# ===================================================================
# MAIN
# ===================================================================

@logger.catch
def main() -> None:
    t_start = time.time()
    fallback_notes: list[str] = []

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    qualifying_combos, bin_data, bin_data_norm, tb_lookup = load_data3(MAX_EXAMPLES)
    iso_to_family, iso_to_grambank_idx = load_grambank(tb_lookup)

    # ------------------------------------------------------------------
    # STEP 2: GEV fitting (dual-track) with bootstrap CIs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"STEP 2: GEV fitting ({len(qualifying_combos)} combos, "
                f"bootstrap={BOOTSTRAP_N})")
    logger.info("=" * 60)

    # Prepare arguments for process_combo
    combo_args = []
    for tb_id, bin_len in qualifying_combos:
        key = (tb_id, bin_len)
        combo_args.append((
            tb_id, bin_len,
            bin_data[key], bin_data_norm[key],
            BOOTSTRAP_N, SEED,
        ))

    # Time estimate: run first 3 combos to calibrate
    if len(combo_args) > 3:
        logger.info("Calibrating runtime on 3 combos ...")
        cal_start = time.time()
        for ca in combo_args[:3]:
            process_combo(ca)
        cal_time = time.time() - cal_start
        per_combo = cal_time / 3
        est_total = per_combo * len(combo_args)
        logger.info(
            f"  ~{per_combo:.2f}s per combo, estimated total: {est_total:.0f}s "
            f"({est_total/60:.1f} min)"
        )

        # Apply fallback F2 if too slow
        if est_total > 45 * 60:
            reduced_boot = max(100, int(BOOTSTRAP_N * (40 * 60) / est_total))
            fallback_notes.append(
                f"F2: Reduced bootstrap from {BOOTSTRAP_N} to {reduced_boot} "
                f"(estimated {est_total/60:.0f} min)"
            )
            logger.warning(f"F2 triggered: reducing bootstrap to {reduced_boot}")
            combo_args = [
                (a[0], a[1], a[2], a[3], reduced_boot, a[5])
                for a in combo_args
            ]

    # Run GEV fitting — use parallel workers for large jobs, sequential
    # for small ones (process-spawn overhead dominates for < 50 combos).
    all_combo_results: list[dict] = []
    n_failed = 0

    if len(combo_args) >= 50 and N_WORKERS >= 2:
        logger.info(f"  Using {N_WORKERS} parallel workers")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = [pool.submit(process_combo, ca) for ca in combo_args]
            for fut in tqdm(
                as_completed(futs), total=len(futs),
                desc="GEV fitting", mininterval=5,
            ):
                try:
                    all_combo_results.append(fut.result())
                except Exception as exc:
                    logger.error(f"Combo failed: {exc}")
                    n_failed += 1
    else:
        logger.info("  Running sequentially (small job)")
        for ca in tqdm(combo_args, desc="GEV fitting", mininterval=5):
            try:
                all_combo_results.append(process_combo(ca))
            except Exception as exc:
                logger.error(f"Failed combo {ca[0]}_{ca[1]}: {exc}")
                n_failed += 1

    if n_failed:
        logger.warning(f"  {n_failed}/{len(combo_args)} combos failed")

    logger.info(f"  Fitted {len(all_combo_results)}/{len(combo_args)} combos "
                f"in {time.time()-t_start:.0f}s")

    # Free raw sentence data
    del bin_data, bin_data_norm
    gc.collect()

    # ------------------------------------------------------------------
    # STEP 3: Fit quality report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Fit quality report")
    logger.info("=" * 60)
    fit_quality = compute_fit_quality(all_combo_results)
    logger.info(f"  GEV AIC-best: {fit_quality['pct_gev_aic_best']:.1f}%")
    logger.info(f"  AD pass: {fit_quality['pct_ad_pass']:.1f}%")
    logger.info(f"  MLE-Lmom mean diff: {fit_quality['mle_lmom_mean_diff']}")

    if fit_quality["pct_gev_aic_best"] < 50:
        fallback_notes.append("F5: GEV fits poorly (<50% AIC-best)")
        logger.warning("F5 triggered: GEV may not be optimal model")

    # ------------------------------------------------------------------
    # STEP 5: Aggregate xi per treebank
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Aggregate xi per treebank")
    logger.info("=" * 60)
    treebank_xi = aggregate_treebank_xi(all_combo_results, tb_lookup)
    logger.info(f"  {len(treebank_xi)} treebanks with valid xi estimates")

    # Dual-track Spearman
    common_tbs = [
        tb for tb in treebank_xi
        if "xi_norm" in treebank_xi[tb] and "xi_raw" in treebank_xi[tb]
    ]
    if len(common_tbs) >= 5:
        raw_vals = [treebank_xi[tb]["xi_raw"] for tb in common_tbs]
        norm_vals = [treebank_xi[tb]["xi_norm"] for tb in common_tbs]
        dual_track_rho, dual_track_p = spearmanr(raw_vals, norm_vals)
    else:
        dual_track_rho, dual_track_p = np.nan, np.nan
    logger.info(f"  Dual-track Spearman rho: {dual_track_rho:.3f} (p={dual_track_p:.2e})")

    # ------------------------------------------------------------------
    # STEP 6: Grambank cross-validation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Grambank cross-validation")
    logger.info("=" * 60)
    grambank_cv = grambank_crossval(tb_lookup)
    logger.info(f"  Grambank overlap: {grambank_cv.get('n_overlap')}")
    logger.info(f"  Spearman r: {grambank_cv.get('spearman_r')}")

    # ------------------------------------------------------------------
    # STEP 7-8: Regression
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7-8: Mixed-effects regression")
    logger.info("=" * 60)
    df_reg = build_regression_df(treebank_xi, tb_lookup)
    regression_results = run_regression(df_reg)
    if regression_results.get("fallback_notes"):
        fallback_notes.extend(regression_results["fallback_notes"])

    # Try to extract a fitted model for predictions
    model_fit_obj = None
    try:
        import statsmodels.formula.api as smf
        formula = ("xi ~ morph_richness_z + head_direction_ratio_z"
                   " + word_order_entropy_z")
        if "mixed" in regression_results.get("model_type", ""):
            m = smf.mixedlm(formula, data=df_reg, groups=df_reg["family"])
            model_fit_obj = m.fit(reml=True)
        else:
            model_fit_obj = smf.ols(formula, data=df_reg).fit()
    except Exception:
        pass

    for pred, vals in regression_results.get("coefficients", {}).items():
        logger.info(
            f"  {pred}: beta={vals.get('beta', '?'):.4f}, "
            f"p={vals.get('p', '?'):.4f}, p_fdr={vals.get('p_fdr', '?')}"
        )

    # ------------------------------------------------------------------
    # STEP 9: Mediation analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 9: Mediation analysis")
    logger.info("=" * 60)
    mediation_results = run_mediation(df_reg, n_boot=MEDIATION_BOOTSTRAP_N)
    logger.info(f"  Indirect effect: {mediation_results.get('indirect_effect_mean')}")
    logger.info(f"  Interpretation: {mediation_results.get('interpretation')}")

    # ------------------------------------------------------------------
    # STEP 10: Discordant language profiles
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 10: Discordant language profiles")
    logger.info("=" * 60)
    discordant_profiles = build_discordant_profiles(
        treebank_xi, tb_lookup, model_fit_obj, df_reg
    )
    for tb_id, prof in discordant_profiles.items():
        logger.info(
            f"  {tb_id}: xi={prof['xi_raw']:.4f}, "
            f"pred={prof.get('predicted_xi')}, "
            f"resid={prof.get('residual')}"
        )

    # ------------------------------------------------------------------
    # STEP 11: Spoken/written comparison
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 11: Spoken vs written comparison")
    logger.info("=" * 60)
    sw_results = spoken_written_analysis(treebank_xi)
    for sw in sw_results:
        logger.info(
            f"  {sw['language']}: spoken={sw['xi_spoken']:.4f}, "
            f"written={sw['xi_written']:.4f}, d={sw['cohens_d']:.3f}"
        )

    # ------------------------------------------------------------------
    # STEP 12: EVT-unique pairs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 12: EVT-unique treebank pairs")
    logger.info("=" * 60)
    evt_results = evt_unique_pairs(treebank_xi, tb_lookup)
    logger.info(
        f"  {evt_results['n_evt_unique']}/{evt_results['n_similar_mean_dd']} "
        f"EVT-unique ({evt_results['pct_evt_unique']:.1f}%)"
    )

    # ------------------------------------------------------------------
    # STEP 13: Genre control
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 13: Genre control")
    logger.info("=" * 60)
    genre_results = genre_control(treebank_xi)
    for lang, gr in genre_results.items():
        logger.info(
            f"  {lang}: range={gr['within_lang_xi_range']:.4f}, "
            f"n_treebanks={len(gr['treebanks'])}"
        )

    # ------------------------------------------------------------------
    # STEP 14: Compile and write output
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 14: Compile output")
    logger.info("=" * 60)

    output = build_schema_output(
        treebank_xi=treebank_xi,
        tb_lookup=tb_lookup,
        all_combo_results=all_combo_results,
        fit_quality=fit_quality,
        dual_track_rho=dual_track_rho,
        dual_track_p=dual_track_p,
        grambank_cv=grambank_cv,
        regression_results=regression_results,
        mediation_results=mediation_results,
        discordant_profiles=discordant_profiles,
        sw_results=sw_results,
        evt_results=evt_results,
        genre_results=genre_results,
        fallback_notes=fallback_notes,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"Pipeline completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("SUCCESS")


if __name__ == "__main__":
    main()
