#!/usr/bin/env python3
"""Super-Block GEV Sensitivity Analysis: Validating Single-Sentence Parametric Approach.

Validates whether single-sentence GEV shape parameter (xi) rankings across
treebank-bin combinations agree with super-block (K=20,30,50) GEV estimates
where formal EVT convergence is better justified.
"""

import gc
import json
import math
import os
import resource
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import genextreme, spearmanr, pearsonr, wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# CONFIGURATION & CONSTANTS
# ===================================================================
WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE / "deps" / "data_id3_it1__opus"
DATA_FILES = [
    DATA_DIR / "data_out" / "full_data_out_1.json",
    DATA_DIR / "data_out" / "full_data_out_2.json",
]
FIGURES_DIR = WORKSPACE / "figures"
LOGS_DIR = WORKSPACE / "logs"

K_VALUES = [20, 30, 50]
MIN_SUPER_BLOCKS = 20
BINS = [10, 12, 14, 16, 18, 20]
N_BOOTSTRAP = 1000
N_SHUFFLES = 5
RANDOM_SEED = 42

# Limit: only process up to MAX_EXAMPLES sentences (0 = all)
MAX_EXAMPLES = 0

# GEV validity range: xi outside this is considered a poor/degenerate fit
XI_VALID_RANGE = (-2.0, 2.0)
# Jitter scale added to super-block maxima to prevent degenerate fits
JITTER_SCALE = 1e-6

# ===================================================================
# HARDWARE DETECTION & MEMORY LIMITS
# ===================================================================
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

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# Set memory limit (use 70% of container RAM = ~20GB)
RAM_BUDGET = int(TOTAL_RAM_GB * 0.70 * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
    logger.info(f"RAM budget set to {RAM_BUDGET / 1e9:.1f} GB (virtual limit {RAM_BUDGET * 3 / 1e9:.1f} GB)")
except (ValueError, OSError) as e:
    logger.warning(f"Could not set memory limit: {e}")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")


# ===================================================================
# PHASE 1: DATA LOADING
# ===================================================================
def load_data(data_files: list[Path], max_examples: int = 0) -> tuple[list[dict], list[dict]]:
    """Load data from JSON files, separating treebank and sentence rows."""
    treebank_rows = []
    sentence_rows = []

    for fpath in data_files:
        logger.info(f"Loading {fpath.name}...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]
        for e in examples:
            if e["metadata_row_type"] == "treebank":
                treebank_rows.append(e)
            elif e["metadata_row_type"] == "sentence":
                sentence_rows.append(e)
        del raw, examples
        gc.collect()
        logger.info(f"  Loaded from {fpath.name}: treebanks={len(treebank_rows)}, sentences={len(sentence_rows)}")

    if max_examples > 0 and len(sentence_rows) > max_examples:
        sentence_rows = sentence_rows[:max_examples]
        logger.info(f"  Truncated to {max_examples} sentences")

    logger.info(f"Total: {len(treebank_rows)} treebanks, {len(sentence_rows)} sentences")
    return treebank_rows, sentence_rows


# ===================================================================
# PHASE 2: ORGANIZE BY (TREEBANK, BIN)
# ===================================================================
def organize_data(
    treebank_rows: list[dict], sentence_rows: list[dict]
) -> tuple[dict, dict]:
    """Build combo_data and treebank_info dicts."""
    combo_data: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"raw_max_dd": [], "norm_max_dd": []}
    )

    for s in sentence_rows:
        key = (s["metadata_treebank_id"], int(s["metadata_length_bin"]))
        combo_data[key]["raw_max_dd"].append(s["metadata_max_dd"])
        combo_data[key]["norm_max_dd"].append(s["metadata_max_dd_normalized"])

    treebank_info: dict[str, dict] = {}
    for t in treebank_rows:
        tid = t["metadata_treebank_id"]
        treebank_info[tid] = {
            "language": t.get("metadata_language", ""),
            "iso_code": t.get("metadata_iso_code", ""),
            "morph_richness": t.get("metadata_morph_richness", 0.0),
            "head_direction_ratio": t.get("metadata_head_direction_ratio", 0.0),
            "word_order_entropy": t.get("metadata_word_order_entropy", 0.0),
            "nonprojectivity_rate": t.get("metadata_nonprojectivity_rate", 0.0),
            "genre": t.get("metadata_genre", ""),
            "modality": t.get("metadata_modality", ""),
        }

    logger.info(f"Organized {len(combo_data)} (treebank, bin) combinations from {len(treebank_info)} treebanks")
    return dict(combo_data), treebank_info


def compute_qualification(
    combo_data: dict, k_values: list[int], min_super_blocks: int
) -> dict[int, list[tuple[str, int]]]:
    """For each K, determine which combos qualify (enough sentences for min_super_blocks)."""
    qualifying: dict[int, list[tuple[str, int]]] = {}

    for k in k_values:
        qualified = []
        for key, vals in combo_data.items():
            n = len(vals["raw_max_dd"])
            n_blocks = n // k
            if n_blocks >= min_super_blocks:
                qualified.append(key)
        qualifying[k] = sorted(qualified)

        treebanks_at_k = set(t for t, b in qualified)
        bins_at_k = set(b for t, b in qualified)
        logger.info(
            f"K={k}: {len(qualified)} qualifying combos across "
            f"{len(treebanks_at_k)} treebanks, bins={sorted(bins_at_k)}"
        )

    return qualifying


# ===================================================================
# PHASE 3: GEV FITTING UTILITIES
# ===================================================================
def fit_gev_lmom(data: np.ndarray) -> dict:
    """Fit GEV via L-moments. Returns {xi, mu, sigma, method, converged}."""
    try:
        from lmoments3 import distr
        params = distr.gev.lmom_fit(data)
        c_lmom = params["c"]
        xi = -c_lmom  # SIGN CONVERSION: c = -xi
        return {
            "xi": float(xi),
            "mu": float(params["loc"]),
            "sigma": float(params["scale"]),
            "method": "lmom",
            "converged": True,
        }
    except Exception as e:
        return {
            "xi": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "method": "lmom",
            "converged": False,
            "error": str(e),
        }


def fit_gev_mle(data: np.ndarray) -> dict:
    """Fit GEV via MLE (scipy). Returns {xi, mu, sigma, method, converged}."""
    try:
        c_mle, loc, scale = genextreme.fit(data)
        xi = -c_mle  # SIGN CONVERSION: c = -xi
        return {
            "xi": float(xi),
            "mu": float(loc),
            "sigma": float(scale),
            "method": "mle",
            "converged": True,
        }
    except Exception as e:
        return {
            "xi": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "method": "mle",
            "converged": False,
            "error": str(e),
        }


def bootstrap_xi(
    data: np.ndarray, n_boot: int = 1000, fit_fn=None, seed: int = 42
) -> dict:
    """Bootstrap CI for xi. Returns {xi_mean, xi_std, xi_ci_lo, xi_ci_hi}."""
    if fit_fn is None:
        fit_fn = fit_gev_lmom
    rng = np.random.default_rng(seed)
    xi_samples = []
    for _ in range(n_boot):
        resample = rng.choice(data, size=len(data), replace=True)
        result = fit_fn(resample)
        if result["converged"] and np.isfinite(result["xi"]):
            xi_samples.append(result["xi"])
    if len(xi_samples) < 10:
        return {
            "xi_mean": float("nan"),
            "xi_std": float("nan"),
            "xi_ci_lo": float("nan"),
            "xi_ci_hi": float("nan"),
            "n_valid_boots": len(xi_samples),
        }
    xi_arr = np.array(xi_samples)
    return {
        "xi_mean": float(np.mean(xi_arr)),
        "xi_std": float(np.std(xi_arr)),
        "xi_ci_lo": float(np.percentile(xi_arr, 2.5)),
        "xi_ci_hi": float(np.percentile(xi_arr, 97.5)),
        "n_valid_boots": len(xi_samples),
    }


# ===================================================================
# PHASE 4: SINGLE-SENTENCE GEV BASELINE (worker)
# ===================================================================
def _fit_single_combo(args: tuple) -> tuple:
    """Worker function for parallel single-sentence GEV fitting."""
    key, raw_list, norm_list, n_boot = args
    raw_values = np.array(raw_list, dtype=float)
    norm_values = np.array(norm_list, dtype=float)

    # Fit GEV (L-moments primary, MLE secondary)
    single_raw_lmom = fit_gev_lmom(raw_values)
    single_raw_mle = fit_gev_mle(raw_values)
    single_norm_lmom = fit_gev_lmom(norm_values)
    single_norm_mle = fit_gev_mle(norm_values)

    # Decision rule: use L-moments primary, override if MLE xi < -0.5
    xi_raw = single_raw_lmom["xi"]
    if single_raw_mle["converged"] and single_raw_mle["xi"] < -0.5:
        xi_raw = single_raw_lmom["xi"]  # override to L-moments

    xi_norm = single_norm_lmom["xi"]
    if single_norm_mle["converged"] and single_norm_mle["xi"] < -0.5:
        xi_norm = single_norm_lmom["xi"]

    # Bootstrap CIs (L-moments only)
    boot_raw = bootstrap_xi(raw_values, n_boot, fit_gev_lmom, RANDOM_SEED)
    boot_norm = bootstrap_xi(norm_values, n_boot, fit_gev_lmom, RANDOM_SEED)

    result = {
        "xi_raw_lmom": single_raw_lmom["xi"],
        "xi_raw_mle": single_raw_mle["xi"],
        "xi_norm_lmom": single_norm_lmom["xi"],
        "xi_norm_mle": single_norm_mle["xi"],
        "boot_raw": boot_raw,
        "boot_norm": boot_norm,
        "n_sentences": len(raw_list),
        "raw_lmom_converged": single_raw_lmom["converged"],
        "norm_lmom_converged": single_norm_lmom["converged"],
        "raw_mle_converged": single_raw_mle["converged"],
        "norm_mle_converged": single_norm_mle["converged"],
    }
    return (key, result)


def fit_single_sentence_gev(
    combo_data: dict, qualifying_keys: set, n_boot: int = 1000
) -> dict:
    """Fit GEV to single-sentence max_DD for all qualifying combos."""
    tasks = []
    for key in qualifying_keys:
        vals = combo_data[key]
        tasks.append((key, vals["raw_max_dd"], vals["norm_max_dd"], n_boot))

    logger.info(f"Fitting single-sentence GEV for {len(tasks)} combos (parallel, {NUM_CPUS} workers)...")
    single_results = {}
    n_workers = max(1, NUM_CPUS - 1)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fit_single_combo, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            try:
                key, result = future.result()
                single_results[key] = result
                done += 1
                if done % 50 == 0:
                    logger.info(f"  Single-sentence: {done}/{len(tasks)} done")
            except Exception as e:
                k = futures[future]
                logger.error(f"  Failed for {k}: {e}")

    logger.info(f"Single-sentence GEV fitting complete: {len(single_results)} results")
    return single_results


# ===================================================================
# PHASE 5: SUPER-BLOCK CONSTRUCTION & GEV FITTING (worker)
# ===================================================================
def _xi_is_valid(xi: float) -> bool:
    """Check if xi is within the valid range (not degenerate)."""
    return np.isfinite(xi) and XI_VALID_RANGE[0] <= xi <= XI_VALID_RANGE[1]


def _add_jitter(data: np.ndarray, seed: int) -> np.ndarray:
    """Add tiny jitter to prevent degenerate fits from zero-variance data."""
    rng = np.random.default_rng(seed)
    std = max(np.std(data), 1e-10)
    return data + rng.normal(0, std * JITTER_SCALE, size=len(data))


def _fit_super_block_combo(args: tuple) -> tuple:
    """Worker function for parallel super-block GEV fitting."""
    key, raw_list, norm_list, k, n_shuffles, n_boot, seed = args
    raw_values = np.array(raw_list, dtype=float)
    norm_values = np.array(norm_list, dtype=float)
    n_blocks = len(raw_values) // k

    xi_raw_shuffles = []
    xi_norm_shuffles = []
    first_super_raw = None
    first_super_norm = None

    for shuffle_i in range(n_shuffles):
        rng = np.random.default_rng(seed + shuffle_i)
        perm = rng.permutation(len(raw_values))
        shuffled_raw = raw_values[perm][: n_blocks * k]
        shuffled_norm = norm_values[perm][: n_blocks * k]

        blocks_raw = shuffled_raw.reshape(n_blocks, k)
        blocks_norm = shuffled_norm.reshape(n_blocks, k)
        super_maxima_raw = blocks_raw.max(axis=1)
        super_maxima_norm = blocks_norm.max(axis=1)

        # Add tiny jitter to prevent degenerate fits
        super_maxima_raw = _add_jitter(super_maxima_raw, seed + shuffle_i * 100)
        super_maxima_norm = _add_jitter(super_maxima_norm, seed + shuffle_i * 100 + 1)

        if shuffle_i == 0:
            first_super_raw = super_maxima_raw.copy()
            first_super_norm = super_maxima_norm.copy()

        fit_raw = fit_gev_lmom(super_maxima_raw)
        fit_norm = fit_gev_lmom(super_maxima_norm)
        if fit_raw["converged"] and _xi_is_valid(fit_raw["xi"]):
            xi_raw_shuffles.append(fit_raw["xi"])
        if fit_norm["converged"] and _xi_is_valid(fit_norm["xi"]):
            xi_norm_shuffles.append(fit_norm["xi"])

    xi_super_raw = float(np.mean(xi_raw_shuffles)) if xi_raw_shuffles else float("nan")
    xi_super_norm = float(np.mean(xi_norm_shuffles)) if xi_norm_shuffles else float("nan")
    xi_raw_std = float(np.std(xi_raw_shuffles)) if len(xi_raw_shuffles) > 1 else float("nan")
    xi_norm_std = float(np.std(xi_norm_shuffles)) if len(xi_norm_shuffles) > 1 else float("nan")

    # MLE on first shuffle
    mle_raw = fit_gev_mle(first_super_raw) if first_super_raw is not None else {"xi": float("nan"), "converged": False}
    mle_norm = fit_gev_mle(first_super_norm) if first_super_norm is not None else {"xi": float("nan"), "converged": False}

    # Bootstrap CI on first shuffle
    boot_raw = {"xi_ci_lo": float("nan"), "xi_ci_hi": float("nan"), "n_valid_boots": 0}
    boot_norm = {"xi_ci_lo": float("nan"), "xi_ci_hi": float("nan"), "n_valid_boots": 0}
    if first_super_raw is not None and len(first_super_raw) >= 20:
        boot_raw = bootstrap_xi(first_super_raw, n_boot, fit_gev_lmom, seed)
    if first_super_norm is not None and len(first_super_norm) >= 20:
        boot_norm = bootstrap_xi(first_super_norm, n_boot, fit_gev_lmom, seed)

    result = {
        "xi_raw": xi_super_raw,
        "xi_norm": xi_super_norm,
        "xi_raw_std_shuffles": xi_raw_std,
        "xi_norm_std_shuffles": xi_norm_std,
        "boot_ci_raw": boot_raw,
        "boot_ci_norm": boot_norm,
        "n_blocks": n_blocks,
        "mle_xi_raw": mle_raw["xi"],
        "mle_xi_norm": mle_norm["xi"],
        "n_converged_shuffles_raw": len(xi_raw_shuffles),
        "n_converged_shuffles_norm": len(xi_norm_shuffles),
    }
    return (key, k, result)


def fit_super_block_gev(
    combo_data: dict, qualifying: dict, n_shuffles: int = 5, n_boot: int = 1000
) -> dict:
    """Fit GEV to super-block maxima for each K and qualifying combo."""
    super_results: dict[int, dict] = {k: {} for k in K_VALUES}

    for k in K_VALUES:
        keys = qualifying[k]
        tasks = []
        for key in keys:
            vals = combo_data[key]
            tasks.append((key, vals["raw_max_dd"], vals["norm_max_dd"], k, n_shuffles, n_boot, RANDOM_SEED))

        logger.info(f"Fitting super-block GEV K={k} for {len(tasks)} combos (parallel)...")
        n_workers = max(1, NUM_CPUS - 1)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_fit_super_block_combo, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                try:
                    key, k_val, result = future.result()
                    super_results[k_val][key] = result
                    done += 1
                    if done % 50 == 0:
                        logger.info(f"  Super-block K={k}: {done}/{len(tasks)} done")
                except Exception as e:
                    fkey = futures[future]
                    logger.error(f"  Super-block K={k} failed for {fkey}: {e}")

        logger.info(f"Super-block K={k} complete: {len(super_results[k])} results")

    return super_results


# ===================================================================
# PHASE 6: SPEARMAN RANK CORRELATION ANALYSIS
# ===================================================================
def compute_correlations(
    single_results: dict, super_results: dict, qualifying: dict
) -> dict:
    """Compute Spearman rho between single-sentence and super-block xi rankings."""
    comparison: dict[int, dict] = {}

    for k in K_VALUES:
        comparison[k] = {}
        keys = qualifying[k]

        for track in ["raw", "norm"]:
            xi_single_list = []
            xi_super_list = []

            n_skipped_invalid = 0
            for key in keys:
                if key not in single_results or key not in super_results[k]:
                    continue

                if track == "raw":
                    xs = single_results[key]["xi_raw_lmom"]
                    xb = super_results[k][key]["xi_raw"]
                else:
                    xs = single_results[key]["xi_norm_lmom"]
                    xb = super_results[k][key]["xi_norm"]

                if _xi_is_valid(xs) and _xi_is_valid(xb):
                    xi_single_list.append(xs)
                    xi_super_list.append(xb)
                elif np.isfinite(xs) or np.isfinite(xb):
                    n_skipped_invalid += 1

            if n_skipped_invalid > 0:
                logger.info(f"  K={k}, {track}: skipped {n_skipped_invalid} combos with out-of-range xi")
            n_pairs = len(xi_single_list)
            if n_pairs < 5:
                logger.warning(f"K={k}, track={track}: only {n_pairs} valid pairs, skipping")
                comparison[k][track] = {
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "rho_ci_lo": float("nan"),
                    "rho_ci_hi": float("nan"),
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "n_pairs": n_pairs,
                }
                continue

            xi_s = np.array(xi_single_list)
            xi_b = np.array(xi_super_list)

            rho, pval = spearmanr(xi_s, xi_b)
            r_pearson, p_pearson = pearsonr(xi_s, xi_b)

            # Bootstrap CI for Spearman rho
            rng = np.random.default_rng(RANDOM_SEED)
            rho_boots = []
            for _ in range(1000):
                idx = rng.choice(n_pairs, size=n_pairs, replace=True)
                r, _ = spearmanr(xi_s[idx], xi_b[idx])
                if np.isfinite(r):
                    rho_boots.append(r)

            rho_ci_lo = float(np.percentile(rho_boots, 2.5)) if rho_boots else float("nan")
            rho_ci_hi = float(np.percentile(rho_boots, 97.5)) if rho_boots else float("nan")

            comparison[k][track] = {
                "spearman_rho": float(rho),
                "spearman_p": float(pval),
                "rho_ci_lo": rho_ci_lo,
                "rho_ci_hi": rho_ci_hi,
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "n_pairs": n_pairs,
            }

            passed = "PASS" if rho > 0.8 else "FAIL"
            logger.info(
                f"K={k}, {track}: Spearman rho={rho:.4f} (p={pval:.2e}), "
                f"CI=[{rho_ci_lo:.3f}, {rho_ci_hi:.3f}], n={n_pairs} [{passed}]"
            )

    return comparison


# ===================================================================
# PHASE 7: SYSTEMATIC BIAS ANALYSIS
# ===================================================================
def compute_bias_analysis(
    single_results: dict, super_results: dict, qualifying: dict, combo_data: dict
) -> dict:
    """Analyze systematic bias between single and super-block xi estimates."""
    bias_results: dict[int, dict] = {}

    for k in K_VALUES:
        keys = qualifying[k]
        for track in ["raw", "norm"]:
            diffs = []
            n_sentences_list = []

            for key in keys:
                if key not in single_results or key not in super_results[k]:
                    continue
                if track == "raw":
                    xs = single_results[key]["xi_raw_lmom"]
                    xb = super_results[k][key]["xi_raw"]
                else:
                    xs = single_results[key]["xi_norm_lmom"]
                    xb = super_results[k][key]["xi_norm"]

                if _xi_is_valid(xs) and _xi_is_valid(xb):
                    diffs.append(xb - xs)
                    n_sentences_list.append(len(combo_data[key]["raw_max_dd"]))

            if len(diffs) < 5:
                continue

            diffs_arr = np.array(diffs)
            mean_diff = float(np.mean(diffs_arr))
            median_diff = float(np.median(diffs_arr))

            try:
                stat_w, pval_w = wilcoxon(diffs_arr)
            except (ValueError, ZeroDivisionError):
                stat_w, pval_w = float("nan"), float("nan")

            # Does bias correlate with sample size?
            try:
                rho_bias_n, p_bias_n = spearmanr(diffs_arr, np.array(n_sentences_list))
            except (ValueError, ZeroDivisionError):
                rho_bias_n, p_bias_n = float("nan"), float("nan")

            bkey = f"K{k}_{track}"
            bias_results[bkey] = {
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "std_diff": float(np.std(diffs_arr)),
                "wilcoxon_stat": float(stat_w),
                "wilcoxon_p": float(pval_w),
                "rho_bias_vs_n": float(rho_bias_n),
                "p_bias_vs_n": float(p_bias_n),
                "n_pairs": len(diffs),
                "diffs": diffs,
            }

            direction = "more negative" if mean_diff < 0 else "more positive"
            logger.info(
                f"Bias K={k} {track}: mean_diff={mean_diff:.4f} ({direction}), "
                f"Wilcoxon p={pval_w:.4f}, rho_bias_vs_n={rho_bias_n:.3f}"
            )

    return bias_results


# ===================================================================
# PHASE 8: SENSITIVITY ANALYSES
# ===================================================================
def sensitivity_analyses(
    single_results: dict, super_results: dict, combo_data: dict, qualifying: dict
) -> dict:
    """Run sensitivity checks: min-blocks, MLE agreement, K-effect."""
    checks = {}

    # 8A: Minimum-blocks sensitivity — repeat with MIN_SUPER_BLOCKS=30
    logger.info("Sensitivity 8A: Min-blocks sensitivity (threshold=30)...")
    qualifying_30 = {}
    for k in K_VALUES:
        qualified_30 = []
        for key, vals in combo_data.items():
            n = len(vals["raw_max_dd"])
            n_blocks = n // k
            if n_blocks >= 30:
                qualified_30.append(key)
        qualifying_30[k] = qualified_30

    for k in K_VALUES:
        keys_30 = set(qualifying_30[k])
        xi_s = []
        xi_b = []
        for key in keys_30:
            if key in single_results and key in super_results[k]:
                xs = single_results[key]["xi_raw_lmom"]
                xb = super_results[k][key]["xi_raw"]
                if _xi_is_valid(xs) and _xi_is_valid(xb):
                    xi_s.append(xs)
                    xi_b.append(xb)
        if len(xi_s) >= 5:
            rho_30, _ = spearmanr(xi_s, xi_b)
            checks[f"min_blocks_30_K{k}"] = {
                "rho": float(rho_30),
                "n_pairs": len(xi_s),
            }
            logger.info(f"  Min-blocks=30 K={k}: rho={rho_30:.4f}, n={len(xi_s)}")
        else:
            checks[f"min_blocks_30_K{k}"] = {"rho": float("nan"), "n_pairs": len(xi_s)}

    # 8B: MLE vs L-moments agreement
    logger.info("Sensitivity 8B: MLE vs L-moments agreement...")
    divergence_single = []
    divergence_super = {}
    for key, res in single_results.items():
        if res["raw_lmom_converged"] and res["raw_mle_converged"]:
            diff = abs(res["xi_raw_lmom"] - res["xi_raw_mle"])
            if np.isfinite(diff):
                divergence_single.append(diff)

    for k in K_VALUES:
        divs = []
        for key, res in super_results[k].items():
            if np.isfinite(res["xi_raw"]) and np.isfinite(res["mle_xi_raw"]):
                divs.append(abs(res["xi_raw"] - res["mle_xi_raw"]))
        divergence_super[k] = divs

    if divergence_single:
        pct_within_01 = float(np.mean([d < 0.1 for d in divergence_single]) * 100)
        pct_within_015 = float(np.mean([d < 0.15 for d in divergence_single]) * 100)
        checks["mle_lmom_agreement_single"] = {
            "mean_divergence": float(np.mean(divergence_single)),
            "pct_within_0.1": pct_within_01,
            "pct_within_0.15": pct_within_015,
            "n": len(divergence_single),
        }
        logger.info(
            f"  Single MLE-Lmom: {pct_within_01:.1f}% within 0.1, "
            f"{pct_within_015:.1f}% within 0.15 (n={len(divergence_single)})"
        )

    for k in K_VALUES:
        divs = divergence_super[k]
        if divs:
            pct_01 = float(np.mean([d < 0.1 for d in divs]) * 100)
            checks[f"mle_lmom_agreement_super_K{k}"] = {
                "mean_divergence": float(np.mean(divs)),
                "pct_within_0.1": pct_01,
                "n": len(divs),
            }
            logger.info(f"  Super K={k} MLE-Lmom: {pct_01:.1f}% within 0.1 (n={len(divs)})")

    # 8D: Effect of K value on xi estimates
    logger.info("Sensitivity 8D: Effect of K on xi...")
    all_k_keys = set(qualifying[K_VALUES[0]])
    for k in K_VALUES[1:]:
        all_k_keys &= set(qualifying[k])
    # Filter to those with valid results at all K
    valid_all_k = []
    for key in all_k_keys:
        if key not in single_results:
            continue
        valid = True
        for k in K_VALUES:
            if key not in super_results[k] or not np.isfinite(super_results[k][key]["xi_raw"]):
                valid = False
                break
        if valid:
            valid_all_k.append(key)

    # Filter to valid xi range
    valid_all_k = [k for k in valid_all_k
                   if all(_xi_is_valid(super_results[kk][k]["xi_raw"]) for kk in K_VALUES)]

    checks["k_effect"] = {
        "n_combos_all_k": len(valid_all_k),
        "k_values": K_VALUES,
    }
    if valid_all_k:
        for k in K_VALUES:
            xis = [super_results[k][key]["xi_raw"] for key in valid_all_k]
            checks["k_effect"][f"mean_xi_K{k}"] = float(np.mean(xis))
            checks["k_effect"][f"std_xi_K{k}"] = float(np.std(xis))

    # 8E: Degenerate fit report
    logger.info("Sensitivity 8E: Degenerate fit report...")
    for k in K_VALUES:
        total = len(super_results[k])
        n_valid_raw = sum(1 for r in super_results[k].values() if _xi_is_valid(r["xi_raw"]))
        n_valid_norm = sum(1 for r in super_results[k].values() if _xi_is_valid(r["xi_norm"]))
        pct_raw = n_valid_raw / total * 100 if total > 0 else 0
        pct_norm = n_valid_norm / total * 100 if total > 0 else 0
        checks[f"degenerate_fits_K{k}"] = {
            "total_combos": total,
            "valid_raw": n_valid_raw,
            "valid_norm": n_valid_norm,
            "pct_valid_raw": round(pct_raw, 1),
            "pct_valid_norm": round(pct_norm, 1),
        }
        logger.info(f"  K={k}: {n_valid_raw}/{total} valid raw ({pct_raw:.1f}%), "
                     f"{n_valid_norm}/{total} valid norm ({pct_norm:.1f}%)")
        logger.info(f"  K-effect: {len(valid_all_k)} combos qualify at all K values")

    # 8F: Per-bin Spearman rho (Fallback D investigation)
    logger.info("Sensitivity 8F: Per-bin Spearman rho...")
    for k in [20, 30]:
        for b in BINS:
            xi_s_bin, xi_b_bin = [], []
            for key in qualifying[k]:
                if key[1] != b:
                    continue
                if key not in single_results or key not in super_results[k]:
                    continue
                xs = single_results[key]["xi_norm_lmom"]
                xb = super_results[k][key]["xi_norm"]
                if _xi_is_valid(xs) and _xi_is_valid(xb):
                    xi_s_bin.append(xs)
                    xi_b_bin.append(xb)
            if len(xi_s_bin) >= 5:
                rho_bin, p_bin = spearmanr(xi_s_bin, xi_b_bin)
                checks[f"per_bin_rho_K{k}_bin{b}_norm"] = {
                    "rho": float(rho_bin), "p": float(p_bin), "n": len(xi_s_bin)
                }
                logger.info(f"  K={k} bin={b} norm: rho={rho_bin:.3f} (n={len(xi_s_bin)})")

    # 8G: Non-parametric quantile comparison (Fallback C alternative)
    logger.info("Sensitivity 8G: Non-parametric quantile rankings...")
    for k in [20]:
        for track, pctile in [("raw", 95), ("raw", 99), ("norm", 95), ("norm", 99)]:
            q_single_list, q_super_list = [], []
            for key in qualifying[k]:
                if key not in single_results or key not in super_results[k]:
                    continue
                vals = combo_data[key]
                raw_arr = np.array(vals["raw_max_dd" if track == "raw" else "norm_max_dd"], dtype=float)
                n_blocks = len(raw_arr) // k
                if n_blocks < MIN_SUPER_BLOCKS:
                    continue
                q_s = float(np.percentile(raw_arr, pctile))
                # Super-block quantile: take block max of first shuffle, compute percentile
                rng_q = np.random.default_rng(RANDOM_SEED)
                perm = rng_q.permutation(len(raw_arr))
                blocks = raw_arr[perm][:n_blocks * k].reshape(n_blocks, k)
                block_max = blocks.max(axis=1)
                q_b = float(np.percentile(block_max, pctile))
                q_single_list.append(q_s)
                q_super_list.append(q_b)
            if len(q_single_list) >= 10:
                rho_q, p_q = spearmanr(q_single_list, q_super_list)
                checks[f"quantile_rho_K{k}_{track}_p{pctile}"] = {
                    "rho": float(rho_q), "p": float(p_q), "n": len(q_single_list)
                }
                logger.info(f"  Quantile p{pctile} K={k} {track}: rho={rho_q:.3f} (n={len(q_single_list)})")

    # 8H: Per-treebank analysis for large treebanks
    logger.info("Sensitivity 8H: Large-treebank analysis...")
    treebank_combos: dict[str, list] = defaultdict(list)
    for key in qualifying[20]:
        treebank_combos[key[0]].append(key)
    large_treebanks = [(tid, keys) for tid, keys in treebank_combos.items() if len(keys) >= 4]
    large_treebanks.sort(key=lambda x: -len(x[1]))
    for tid, keys in large_treebanks[:10]:
        xi_s_tb, xi_b_tb = [], []
        for key in keys:
            if key not in single_results or key not in super_results[20]:
                continue
            xs = single_results[key]["xi_norm_lmom"]
            xb = super_results[20][key]["xi_norm"]
            if _xi_is_valid(xs) and _xi_is_valid(xb):
                xi_s_tb.append(xs)
                xi_b_tb.append(xb)
        if len(xi_s_tb) >= 4:
            rho_tb, _ = spearmanr(xi_s_tb, xi_b_tb)
            checks[f"treebank_rho_{tid}"] = {
                "rho": float(rho_tb), "n_bins": len(xi_s_tb)
            }
            logger.info(f"  {tid}: rho={rho_tb:.3f} ({len(xi_s_tb)} bins)")

    return checks


# ===================================================================
# PHASE 9: VISUALIZATIONS
# ===================================================================
def create_visualizations(
    single_results: dict,
    super_results: dict,
    qualifying: dict,
    bias_results: dict,
    combo_data: dict,
    comparison: dict,
    checks: dict,
) -> None:
    """Generate all figures and save to figures/ directory."""
    FIGURES_DIR.mkdir(exist_ok=True)
    sns.set_style("whitegrid")
    bin_colors = {10: "#e41a1c", 12: "#377eb8", 14: "#4daf4a", 16: "#984ea3", 18: "#ff7f00", 20: "#a65628"}

    # FIG 1: Scatter raw track
    logger.info("Creating Figure 1: Scatter raw track...")
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 5), squeeze=False)
    for idx, k in enumerate(K_VALUES):
        ax = axes[0, idx]
        keys = qualifying[k]
        for key in keys:
            if key not in single_results or key not in super_results[k]:
                continue
            xs = single_results[key]["xi_raw_lmom"]
            xb = super_results[k][key]["xi_raw"]
            if _xi_is_valid(xs) and _xi_is_valid(xb):
                b = key[1]
                ax.scatter(xs, xb, c=bin_colors.get(b, "gray"), s=20, alpha=0.6, edgecolors="none")
        # Diagonal
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("xi (single-sentence, L-moments)")
        ax.set_ylabel("xi (super-block, L-moments)")
        rho_val = comparison.get(k, {}).get("raw", {}).get("spearman_rho", float("nan"))
        n_val = comparison.get(k, {}).get("raw", {}).get("n_pairs", 0)
        ax.set_title(f"K={k}  (rho={rho_val:.3f}, n={n_val})")
    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"n={b}")
               for b, c in sorted(bin_colors.items())]
    axes[0, -1].legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle("Single-Sentence vs Super-Block GEV Shape Parameter (Raw)", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig1_scatter_raw.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG 2: Scatter normalized track
    logger.info("Creating Figure 2: Scatter normalized track...")
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 5), squeeze=False)
    for idx, k in enumerate(K_VALUES):
        ax = axes[0, idx]
        keys = qualifying[k]
        for key in keys:
            if key not in single_results or key not in super_results[k]:
                continue
            xs = single_results[key]["xi_norm_lmom"]
            xb = super_results[k][key]["xi_norm"]
            if _xi_is_valid(xs) and _xi_is_valid(xb):
                b = key[1]
                ax.scatter(xs, xb, c=bin_colors.get(b, "gray"), s=20, alpha=0.6, edgecolors="none")
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("xi (single-sentence, L-moments)")
        ax.set_ylabel("xi (super-block, L-moments)")
        rho_val = comparison.get(k, {}).get("norm", {}).get("spearman_rho", float("nan"))
        n_val = comparison.get(k, {}).get("norm", {}).get("n_pairs", 0)
        ax.set_title(f"K={k}  (rho={rho_val:.3f}, n={n_val})")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"n={b}")
               for b, c in sorted(bin_colors.items())]
    axes[0, -1].legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle("Single-Sentence vs Super-Block GEV Shape Parameter (Normalized)", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig2_scatter_norm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG 3: Bias histograms
    logger.info("Creating Figure 3: Bias histograms...")
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 4), squeeze=False)
    for idx, k in enumerate(K_VALUES):
        ax = axes[0, idx]
        bkey = f"K{k}_raw"
        if bkey in bias_results and bias_results[bkey]["diffs"]:
            diffs = np.array(bias_results[bkey]["diffs"])
            ax.hist(diffs, bins=30, color="#377eb8", alpha=0.7, edgecolor="white")
            ax.axvline(0, color="red", linestyle="--", lw=1.5)
            ax.axvline(np.mean(diffs), color="orange", linestyle="-", lw=1.5, label=f"mean={np.mean(diffs):.4f}")
            ax.axvline(np.median(diffs), color="green", linestyle="-", lw=1.5, label=f"median={np.median(diffs):.4f}")
            wp = bias_results[bkey].get("wilcoxon_p", float("nan"))
            ax.set_title(f"K={k} (Wilcoxon p={wp:.4f})")
            ax.legend(fontsize=8)
        ax.set_xlabel("xi_super - xi_single")
        ax.set_ylabel("Count")
    fig.suptitle("Systematic Bias: Super-Block vs Single-Sentence xi (Raw)", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig3_bias_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG 4: Qualification funnel chart
    logger.info("Creating Figure 4: Qualification funnel...")
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(K_VALUES))
    for b in BINS:
        counts = []
        for k in K_VALUES:
            cnt = sum(1 for key in qualifying[k] if key[1] == b)
            counts.append(cnt)
        ax.bar([f"K={k}" for k in K_VALUES], counts, bottom=bottom,
               label=f"n={b}", color=bin_colors[b], alpha=0.8)
        bottom += np.array(counts)
    ax.set_ylabel("Number of Qualifying Combos")
    ax.set_title("Qualifying (Treebank, Bin) Combinations by Super-Block Size K")
    ax.legend(title="Sentence Length")
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig4_qualification_funnel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG 5: xi convergence plot
    logger.info("Creating Figure 5: xi convergence...")
    all_k_keys = set(qualifying[K_VALUES[0]])
    for k in K_VALUES[1:]:
        all_k_keys &= set(qualifying[k])
    valid_all_k = []
    for key in all_k_keys:
        if key not in single_results:
            continue
        valid = all(key in super_results[k] and np.isfinite(super_results[k][key]["xi_raw"]) for k in K_VALUES)
        if valid:
            valid_all_k.append(key)

    fig, ax = plt.subplots(figsize=(8, 5))
    if valid_all_k:
        for key in valid_all_k[:100]:  # Limit lines for readability
            xis = [super_results[k][key]["xi_raw"] for k in K_VALUES]
            b = key[1]
            ax.plot(K_VALUES, xis, color=bin_colors.get(b, "gray"), alpha=0.3, lw=0.8)
        # Mean line
        for k in K_VALUES:
            mean_xi = np.mean([super_results[k][key]["xi_raw"] for key in valid_all_k])
            ax.scatter([k], [mean_xi], color="black", s=80, zorder=5)
    ax.set_xlabel("K (Super-Block Size)")
    ax.set_ylabel("xi (L-moments, raw)")
    ax.set_title(f"xi Convergence Across K Values ({len(valid_all_k)} combos)")
    ax.set_xticks(K_VALUES)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig5_xi_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG 6: MLE vs L-moments agreement
    logger.info("Creating Figure 6: MLE vs L-moments agreement...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Single-sentence
    ax = axes[0]
    for key, res in single_results.items():
        if res["raw_lmom_converged"] and res["raw_mle_converged"]:
            xl = res["xi_raw_lmom"]
            xm = res["xi_raw_mle"]
            if _xi_is_valid(xl) and _xi_is_valid(xm):
                b = key[1]
                ax.scatter(xl, xm, c=bin_colors.get(b, "gray"), s=15, alpha=0.5, edgecolors="none")
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
    ax.set_xlabel("xi (L-moments)")
    ax.set_ylabel("xi (MLE)")
    ax.set_title("Single-Sentence: L-moments vs MLE")

    # Super-block (K=20)
    ax = axes[1]
    if 20 in super_results:
        for key, res in super_results[20].items():
            if _xi_is_valid(res["xi_raw"]) and _xi_is_valid(res["mle_xi_raw"]):
                b = key[1]
                ax.scatter(res["xi_raw"], res["mle_xi_raw"],
                           c=bin_colors.get(b, "gray"), s=15, alpha=0.5, edgecolors="none")
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
    ax.set_xlabel("xi (L-moments)")
    ax.set_ylabel("xi (MLE)")
    ax.set_title("Super-Block K=20: L-moments vs MLE")

    fig.suptitle("L-moments vs MLE Agreement", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig6_lmom_vs_mle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("All 6 figures saved to figures/")


# ===================================================================
# PHASE 10: OUTPUT GENERATION
# ===================================================================
def _s(val: Any) -> str:
    """Convert any value to string for predict_* fields (schema requires string)."""
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return "NaN"
        return f"{val}"
    return str(val)


def generate_output(
    single_results: dict,
    super_results: dict,
    qualifying: dict,
    comparison: dict,
    bias_results: dict,
    checks: dict,
    treebank_info: dict,
    combo_data: dict,
) -> dict:
    """Build output in standard schema."""
    datasets = []

    # Dataset 1: super_block_gev_combos
    examples_combos = []
    for k in K_VALUES:
        for key in sorted(qualifying[k]):
            if key not in single_results or key not in super_results[k]:
                continue
            tid, binn = key
            sr = single_results[key]
            br = super_results[k][key]
            tinfo = treebank_info.get(tid, {})
            xi_diff = br["xi_raw"] - sr["xi_raw_lmom"] if np.isfinite(br["xi_raw"]) and np.isfinite(sr["xi_raw_lmom"]) else float("nan")

            examples_combos.append({
                "input": f"{tid}__{binn}__{k}",
                "output": f"xi_single={sr['xi_raw_lmom']:.4f}, xi_super={br['xi_raw']:.4f}, diff={xi_diff:.4f}",
                "metadata_treebank_id": tid,
                "metadata_length_bin": str(binn),
                "metadata_K": str(k),
                "metadata_n_sentences": str(sr["n_sentences"]),
                "metadata_n_super_blocks": str(br["n_blocks"]),
                "metadata_language": tinfo.get("language", ""),
                "metadata_morph_richness": str(tinfo.get("morph_richness", "")),
                "metadata_head_direction_ratio": str(tinfo.get("head_direction_ratio", "")),
                "metadata_word_order_entropy": str(tinfo.get("word_order_entropy", "")),
                "predict_xi_single_lmom": _s(sr["xi_raw_lmom"]),
                "predict_xi_single_mle": _s(sr["xi_raw_mle"]),
                "predict_xi_super_lmom": _s(br["xi_raw"]),
                "predict_xi_super_mle": _s(br["mle_xi_raw"]),
                "predict_xi_single_boot_ci_lo": _s(sr["boot_raw"]["xi_ci_lo"]),
                "predict_xi_single_boot_ci_hi": _s(sr["boot_raw"]["xi_ci_hi"]),
                "predict_xi_super_boot_ci_lo": _s(br["boot_ci_raw"].get("xi_ci_lo", float("nan"))),
                "predict_xi_super_boot_ci_hi": _s(br["boot_ci_raw"].get("xi_ci_hi", float("nan"))),
                "predict_xi_diff": _s(xi_diff),
                "predict_xi_abs_diff": _s(abs(xi_diff) if np.isfinite(xi_diff) else float("nan")),
            })

    datasets.append({"dataset": "super_block_gev_combos", "examples": examples_combos})

    # Dataset 2: super_block_summary
    examples_summary = []
    for k in K_VALUES:
        for track in ["raw", "norm"]:
            comp = comparison.get(k, {}).get(track, {})
            bkey = f"K{k}_{track}"
            br = bias_results.get(bkey, {})
            n_treebanks = len(set(t for t, b in qualifying[k]))
            passed = comp.get("spearman_rho", 0) > 0.8 if np.isfinite(comp.get("spearman_rho", float("nan"))) else False

            examples_summary.append({
                "input": f"K={k}_track={track}",
                "output": f"spearman_rho={comp.get('spearman_rho', float('nan')):.4f} "
                          f"(CI: [{comp.get('rho_ci_lo', float('nan')):.3f}, {comp.get('rho_ci_hi', float('nan')):.3f}]), "
                          f"n_pairs={comp.get('n_pairs', 0)}",
                "metadata_K": str(k),
                "metadata_track": track,
                "predict_spearman_rho": _s(comp.get("spearman_rho", float("nan"))),
                "predict_spearman_p": _s(comp.get("spearman_p", float("nan"))),
                "predict_rho_ci_lo": _s(comp.get("rho_ci_lo", float("nan"))),
                "predict_rho_ci_hi": _s(comp.get("rho_ci_hi", float("nan"))),
                "predict_pearson_r": _s(comp.get("pearson_r", float("nan"))),
                "predict_pearson_p": _s(comp.get("pearson_p", float("nan"))),
                "predict_n_qualifying_combos": _s(comp.get("n_pairs", 0)),
                "predict_n_qualifying_treebanks": _s(n_treebanks),
                "predict_mean_xi_diff": _s(br.get("mean_diff", float("nan"))),
                "predict_median_xi_diff": _s(br.get("median_diff", float("nan"))),
                "predict_wilcoxon_p": _s(br.get("wilcoxon_p", float("nan"))),
                "predict_validation_passed": _s(passed),
            })

    datasets.append({"dataset": "super_block_summary", "examples": examples_summary})

    # Dataset 3: sensitivity_checks
    examples_checks = []
    for check_name, check_vals in checks.items():
        summary_parts = []
        for ck, cv in check_vals.items():
            summary_parts.append(f"{ck}={cv}")
        examples_checks.append({
            "input": f"check_{check_name}",
            "output": "; ".join(summary_parts[:10]),
            "predict_check_name": _s(check_name),
            "predict_check_result": _s(json.dumps(check_vals, default=str)),
        })

    datasets.append({"dataset": "sensitivity_checks", "examples": examples_checks})

    return {"datasets": datasets}


# ===================================================================
# MAIN
# ===================================================================
@logger.catch
def main():
    import time
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Super-Block GEV Sensitivity Analysis")
    logger.info("=" * 70)

    # Phase 1: Load data
    logger.info("PHASE 1: Loading data...")
    treebank_rows, sentence_rows = load_data(DATA_FILES, MAX_EXAMPLES)

    # Phase 2: Organize
    logger.info("PHASE 2: Organizing data...")
    combo_data, treebank_info = organize_data(treebank_rows, sentence_rows)
    del treebank_rows, sentence_rows
    gc.collect()

    # Qualification
    qualifying = compute_qualification(combo_data, K_VALUES, MIN_SUPER_BLOCKS)

    # Check fallbacks
    for k in K_VALUES:
        if len(qualifying[k]) < 5:
            logger.warning(f"K={k} has only {len(qualifying[k])} combos — insufficient!")

    # All qualifying keys (union across K values)
    all_qualifying_keys = set()
    for k in K_VALUES:
        all_qualifying_keys.update(qualifying[k])
    logger.info(f"Total unique qualifying combos: {len(all_qualifying_keys)}")

    # Phase 4: Single-sentence baseline
    logger.info("PHASE 4: Single-sentence GEV fitting...")
    single_results = fit_single_sentence_gev(combo_data, all_qualifying_keys, N_BOOTSTRAP)

    elapsed = time.time() - t0
    logger.info(f"  Single-sentence done in {elapsed:.1f}s")

    # Phase 5: Super-block fitting
    logger.info("PHASE 5: Super-block GEV fitting...")
    super_results = fit_super_block_gev(combo_data, qualifying, N_SHUFFLES, N_BOOTSTRAP)

    elapsed = time.time() - t0
    logger.info(f"  Super-block done in {elapsed:.1f}s")

    # Phase 6: Correlations
    logger.info("PHASE 6: Spearman rank correlation analysis...")
    comparison = compute_correlations(single_results, super_results, qualifying)

    # Phase 7: Bias analysis
    logger.info("PHASE 7: Systematic bias analysis...")
    bias_results = compute_bias_analysis(single_results, super_results, qualifying, combo_data)

    # Phase 8: Sensitivity
    logger.info("PHASE 8: Sensitivity analyses...")
    checks = sensitivity_analyses(single_results, super_results, combo_data, qualifying)

    # Phase 9: Visualizations
    logger.info("PHASE 9: Creating visualizations...")
    create_visualizations(
        single_results, super_results, qualifying, bias_results,
        combo_data, comparison, checks
    )

    # Phase 10: Output
    logger.info("PHASE 10: Generating output...")
    output = generate_output(
        single_results, super_results, qualifying, comparison,
        bias_results, checks, treebank_info, combo_data
    )

    # Save output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path}")

    total_examples = sum(len(ds["examples"]) for ds in output["datasets"])
    logger.info(f"Total examples in output: {total_examples}")

    elapsed = time.time() - t0
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
