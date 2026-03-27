#!/usr/bin/env python3
"""Bound-Awareness Simulation: Random Projective Linearization GEV Null Distributions.

Generates random projective linearizations of observed dependency trees across 10
typologically diverse UD treebanks at 6 sentence lengths, fits GEV to both null and
observed max_DD distributions, and determines whether raw-track or normalized-track
should serve as the primary analysis based on the empirical xi range under the null model.
"""

import gc
import json
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
from loguru import logger

# ============================================================
# WORKSPACE AND PATHS
# ============================================================
WORKSPACE = Path(__file__).resolve().parent
DEP_DATA = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bto/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
PREVIEW_FILE = DEP_DATA / "preview_data_out.json"
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION (cgroup-aware)
# ============================================================

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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Set memory limits: use at most 80% of container RAM
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except (ValueError, OSError) as e:
    logger.warning(f"Could not set RLIMIT_AS: {e}")

# CPU time limit: 1 hour
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except (ValueError, OSError) as e:
    logger.warning(f"Could not set RLIMIT_CPU: {e}")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_LENGTHS = [10, 12, 14, 16, 18, 20]
MAX_TREES_PER_BIN = 100
N_LINEARIZATIONS = 500
N_BOOTSTRAP = 1000
MIN_SAMPLES_FOR_GEV = 30

SELECTED_TREEBANKS = [
    "ar_padt",     # Arabic: head-initial (hdr~0.63), rich morph (~2.4)
    "eu_bdt",      # Basque: SOV/head-final (hdr~0.32), very rich morph (~4.0)
    "zh_gsdsimp",  # Chinese: mixed (hdr~0.55), poor morph (~0.0)
    "cs_pdt",      # Czech: mixed (hdr~0.47), rich morph (~2.8)
    "en_ewt",      # English: mixed (hdr~0.53), moderate morph (~1.3)
    "fi_tdt",      # Finnish: mixed (hdr~0.42), very rich morph (~3.5)
    "hi_hdtb",     # Hindi: head-final (hdr~0.37), moderate morph (~2.3)
    "id_gsd",      # Indonesian: head-initial (hdr~0.55), poor morph (~0.5)
    "ja_gsd",      # Japanese: strongly head-final (hdr~0.20), minimal morph (~0.0)
    "tr_imst",     # Turkish: head-final (hdr~0.28), rich morph (~3.0)
]

FALLBACKS = {
    "ar_padt": "he_htb",
    "eu_bdt": "eu_bdt",
    "zh_gsdsimp": "zh_gsd",
    "cs_pdt": "cs_cac",
    "en_ewt": "en_gum",
    "fi_tdt": "fi_ftb",
    "hi_hdtb": "ur_udtb",
    "id_gsd": "vi_vtb",
    "ja_gsd": "ko_kaist",
    "tr_imst": "tr_boun",
}

# HuggingFace dataset
HF_DATASET = "commul/universal_dependencies"

# ============================================================
# CORE ALGORITHMS
# ============================================================

def build_tree(heads_int: list[int]) -> tuple[dict, int | None]:
    """Build adjacency list from UD head array.
    heads_int[i] = parent of token i+1 (0=root).
    Returns (children_dict, root_node).
    """
    children: dict[int, list[int]] = defaultdict(list)
    root = None
    for i, h in enumerate(heads_int):
        node = i + 1
        if h == 0:
            root = node
        else:
            children[h].append(node)
    return dict(children), root


def random_projective_linearize(children: dict, node: int) -> list[int]:
    """Recursively produce a random projective linearization.
    At each node: randomly shuffle children, randomly split into
    left-of-head and right-of-head groups, recursively linearize.
    Guarantees projectivity by construction.
    """
    kids = children.get(node, [])
    if not kids:
        return [node]

    shuffled = list(kids)
    random.shuffle(shuffled)

    k = random.randint(0, len(shuffled))
    left_children = shuffled[:k]
    right_children = shuffled[k:]

    result = []
    for child in left_children:
        result.extend(random_projective_linearize(children, child))
    result.append(node)
    for child in right_children:
        result.extend(random_projective_linearize(children, child))
    return result


def compute_max_dd(linearized_order: list[int], heads_int: list[int]) -> int:
    """Compute max dependency distance given new linear positions."""
    pos = {node: idx for idx, node in enumerate(linearized_order)}
    max_dd = 0
    for i, h in enumerate(heads_int):
        if h == 0:
            continue
        node = i + 1
        dd = abs(pos[node] - pos[h])
        max_dd = max(max_dd, dd)
    return max_dd


def is_projective(linearized_order: list[int], heads_int: list[int]) -> bool:
    """Check that all dependency arcs are non-crossing in the linearization."""
    pos = {node: idx for idx, node in enumerate(linearized_order)}
    arcs = []
    for i, h in enumerate(heads_int):
        if h == 0:
            continue
        node = i + 1
        l_pos, r_pos = min(pos[node], pos[h]), max(pos[node], pos[h])
        arcs.append((l_pos, r_pos))
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            a1l, a1r = arcs[i]
            a2l, a2r = arcs[j]
            if (a1l < a2l < a1r < a2r) or (a2l < a1l < a2r < a1r):
                return False
    return True


def fit_gev(data: np.ndarray, method: str = "lmom") -> dict:
    """Fit GEV and return xi (standard convention), loc, scale.
    CRITICAL: negate c to get standard xi.
    """
    data = np.array(data, dtype=float)
    if len(data) < MIN_SAMPLES_FOR_GEV:
        return {"xi": float("nan"), "loc": float("nan"), "scale": float("nan"),
                "method": method, "success": False, "reason": "too few samples"}

    if method == "lmom":
        try:
            from lmoments3 import distr as lm_distr
            params = lm_distr.gev.lmom_fit(data)
            c = params["c"]
            loc = params["loc"]
            scale = params["scale"]
            xi = -c  # NEGATE!
            if not np.isfinite(xi) or not np.isfinite(loc) or not np.isfinite(scale):
                return {"xi": float("nan"), "loc": float("nan"), "scale": float("nan"),
                        "method": "lmom", "success": False, "reason": "non-finite params"}
            return {"xi": float(xi), "loc": float(loc), "scale": float(scale),
                    "method": "lmom", "success": True, "c_raw": float(c)}
        except Exception as e:
            return {"xi": float("nan"), "loc": float("nan"), "scale": float("nan"),
                    "method": "lmom", "success": False, "reason": str(e)[:200]}

    elif method == "mle":
        try:
            from scipy.stats import genextreme
            c, loc, scale = genextreme.fit(data)
            xi = -c  # NEGATE!
            if not np.isfinite(xi) or not np.isfinite(loc) or not np.isfinite(scale):
                return {"xi": float("nan"), "loc": float("nan"), "scale": float("nan"),
                        "method": "mle", "success": False, "reason": "non-finite params"}
            flag = "xi_below_neg05_mle_unreliable" if xi < -0.5 else ""
            return {"xi": float(xi), "loc": float(loc), "scale": float(scale),
                    "method": "mle", "success": True, "c_raw": float(c), "flag": flag}
        except Exception as e:
            return {"xi": float("nan"), "loc": float("nan"), "scale": float("nan"),
                    "method": "mle", "success": False, "reason": str(e)[:200]}

    return {"xi": float("nan"), "loc": float("nan"), "scale": float("nan"),
            "method": method, "success": False, "reason": f"unknown method {method}"}


def bootstrap_xi_ci(data: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     method: str = "lmom") -> tuple[float, float, int]:
    """Compute 95% bootstrap CI for xi."""
    xis = []
    data = np.array(data, dtype=float)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        result = fit_gev(sample, method=method)
        if result["success"] and np.isfinite(result["xi"]):
            xis.append(result["xi"])
    if len(xis) < n_boot * 0.5:
        return (float("nan"), float("nan"), len(xis))
    return (float(np.percentile(xis, 2.5)),
            float(np.percentile(xis, 97.5)),
            len(xis))


def coles_criterion(xi: float, loc: float, scale: float,
                    observed_max: float) -> dict:
    """Check Coles (2001): for xi < 0, GEV-implied upper endpoint
    should exceed observed max by >= 20%.
    Standard convention: upper_endpoint = loc - scale / xi (for xi < 0)
    """
    if xi >= 0 or np.isnan(xi):
        return {"applicable": False, "reason": "xi >= 0 or NaN"}
    upper_endpoint = loc - scale / xi  # xi < 0 so this is loc + scale/|xi|
    ratio = upper_endpoint / observed_max if observed_max > 0 else float("nan")
    return {
        "applicable": True,
        "upper_endpoint": float(upper_endpoint),
        "observed_max": float(observed_max),
        "ratio": float(ratio),
        "passes_20pct": bool(ratio >= 1.2),
    }


# ============================================================
# PARALLEL WORKER: Process one tree for N linearizations
# ============================================================

def _process_tree_batch(args: tuple) -> list[int]:
    """Worker: generate N_LINS random projective linearizations for one tree
    and return the list of null max_DD values.
    args = (heads_int_list, children_dict, root_node, n_lins, seed)
    """
    heads_int, children, root, n_lins, seed = args
    rng_local = random.Random(seed)
    # Override module-level random for this worker
    random.seed(seed)
    null_dds = []
    for _ in range(n_lins):
        lin_order = random_projective_linearize(children, root)
        null_dd = compute_max_dd(lin_order, heads_int)
        null_dds.append(null_dd)
    return null_dds


# ============================================================
# DATASET LOADING
# ============================================================

def load_treebank_from_hf(tb_name: str) -> list | None:
    """Load a treebank from HuggingFace and return combined dataset rows."""
    from datasets import concatenate_datasets, load_dataset
    try:
        logger.info(f"  Loading {tb_name} from HuggingFace...")
        ds = load_dataset(HF_DATASET, tb_name)
        full_tb = concatenate_datasets([ds[s] for s in ds.keys()])
        logger.info(f"  Loaded {tb_name}: {len(full_tb)} sentences")
        return full_tb
    except Exception as e:
        logger.error(f"  Failed to load {tb_name}: {e}")
        return None


def extract_trees_for_length(full_tb, target_len: int,
                             max_trees: int) -> tuple[list, list]:
    """Extract trees of exactly target_len from the treebank.
    Returns (trees, obs_max_dds) where trees is list of (heads_int, children, root).
    """
    trees = []
    obs_max_dds = []

    for idx in range(len(full_tb)):
        row = full_tb[idx]
        tokens = row["tokens"]
        heads_str = row["head"]
        n = len(tokens)
        if n != target_len:
            continue

        # Parse heads to int, validate
        heads_int = []
        valid = True
        for h in heads_str:
            try:
                hi = int(h)
                if hi < 0 or hi > n:
                    valid = False
                    break
                heads_int.append(hi)
            except (ValueError, TypeError):
                valid = False
                break
        if not valid or len(heads_int) != n:
            continue

        # Must have exactly one root
        root_count = sum(1 for h in heads_int if h == 0)
        if root_count != 1:
            continue

        # Compute observed max_DD
        obs_dd = 0
        for i, h in enumerate(heads_int):
            if h == 0:
                continue
            obs_dd = max(obs_dd, abs((i + 1) - h))
        if obs_dd == 0:
            continue
        obs_max_dds.append(obs_dd)

        # Build tree
        children, root = build_tree(heads_int)
        if root is None:
            continue
        trees.append((heads_int, dict(children), root))

        if len(trees) >= max_trees:
            break

    return trees, obs_max_dds


# ============================================================
# PROCESS ONE TREEBANK-LENGTH COMBINATION
# ============================================================

def process_treebank_length(
    tb_name: str,
    trees: list,
    obs_max_dds: list,
    target_len: int,
    n_lins: int,
    n_boot: int,
    phase_label: str = "full",
) -> dict | None:
    """Process one treebank-length combination:
    generate null linearizations, fit GEV to null and observed, compute diagnostics.
    """
    if len(trees) < 10:
        logger.warning(f"  Only {len(trees)} trees for {tb_name} len={target_len}, skipping")
        return None

    bound = target_len - 1
    logger.info(f"  [{phase_label}] {tb_name} len={target_len}: {len(trees)} trees, "
                f"{len(obs_max_dds)} observed, generating {len(trees)*n_lins} null lins")

    # --- Generate null linearizations using multiprocessing ---
    n_workers = max(1, NUM_CPUS - 1)
    tasks = []
    for tree_idx, (heads_int, children, root) in enumerate(trees):
        seed = hash((tb_name, target_len, tree_idx)) & 0xFFFFFFFF
        tasks.append((heads_int, children, root, n_lins, seed))

    null_max_dds = []
    # Use ProcessPoolExecutor for CPU-bound linearization work
    if len(tasks) > 5 and n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_tree_batch, t): i for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    null_max_dds.extend(result)
                except Exception as e:
                    logger.error(f"  Worker error: {e}")
    else:
        # Sequential for small batches or single CPU
        for t in tasks:
            null_max_dds.extend(_process_tree_batch(t))

    # Validation on first tree
    if trees:
        heads_int_0, children_0, root_0 = trees[0]
        for check_idx in range(min(5, n_lins)):
            random.seed(hash((tb_name, target_len, 0, "validate", check_idx)) & 0xFFFFFFFF)
            lin_order = random_projective_linearize(children_0, root_0)
            assert len(lin_order) == target_len, \
                f"Linearization length {len(lin_order)} != {target_len}"
            if check_idx < 3:
                assert is_projective(lin_order, heads_int_0), \
                    f"Non-projective linearization detected!"

    logger.info(f"  Generated {len(null_max_dds)} null max_DDs")

    # --- Descriptive statistics ---
    obs_arr = np.array(obs_max_dds, dtype=float)
    null_arr = np.array(null_max_dds, dtype=float)
    obs_range_occupancy = float(np.mean(obs_arr)) / bound
    null_range_occupancy = float(np.mean(null_arr)) / bound
    obs_pct_near_bound = float(np.mean(obs_arr >= 0.9 * bound))
    null_pct_near_bound = float(np.mean(null_arr >= 0.9 * bound))

    # --- GEV fitting: NULL distribution ---
    null_lmom = fit_gev(null_arr, method="lmom")
    null_mle = fit_gev(null_arr, method="mle")
    null_ci = bootstrap_xi_ci(null_arr, n_boot=min(500, n_boot), method="lmom")

    # --- GEV fitting: OBSERVED distribution ---
    obs_lmom = fit_gev(obs_arr, method="lmom")
    obs_mle = fit_gev(obs_arr, method="mle")
    obs_ci = bootstrap_xi_ci(obs_arr, n_boot=n_boot, method="lmom")

    # --- Coles criterion (on observed) ---
    if obs_lmom["success"]:
        coles = coles_criterion(
            obs_lmom["xi"], obs_lmom["loc"], obs_lmom["scale"],
            float(np.max(obs_arr))
        )
    else:
        coles = {"applicable": False, "reason": "obs fitting failed"}

    # --- Also fit normalized track ---
    obs_norm = obs_arr / bound
    null_norm = null_arr / bound
    obs_norm_lmom = fit_gev(obs_norm, method="lmom")
    null_norm_lmom = fit_gev(null_norm, method="lmom")

    # --- Baseline: fit GEV to uniform random integers in [1, bound] ---
    rng = np.random.default_rng(42)
    uniform_samples = rng.integers(1, bound + 1, size=len(null_max_dds))
    uniform_lmom = fit_gev(uniform_samples.astype(float), method="lmom")

    # --- Method agreement ---
    if null_lmom["success"] and null_mle["success"]:
        xi_agreement = abs(null_lmom["xi"] - null_mle["xi"]) < 0.1
    else:
        xi_agreement = None

    return {
        "treebank": tb_name,
        "sentence_length": target_len,
        "n_trees": len(trees),
        "n_observed_sentences": len(obs_max_dds),
        "n_null_linearizations": len(null_max_dds),
        "theoretical_bound": bound,
        # Descriptive
        "obs_mean_maxdd": round(float(np.mean(obs_arr)), 4),
        "obs_std_maxdd": round(float(np.std(obs_arr)), 4),
        "obs_max_maxdd": int(np.max(obs_arr)),
        "obs_min_maxdd": int(np.min(obs_arr)),
        "obs_median_maxdd": float(np.median(obs_arr)),
        "obs_range_occupancy": round(obs_range_occupancy, 4),
        "obs_pct_near_bound": round(obs_pct_near_bound, 4),
        "null_mean_maxdd": round(float(np.mean(null_arr)), 4),
        "null_std_maxdd": round(float(np.std(null_arr)), 4),
        "null_max_maxdd": int(np.max(null_arr)),
        "null_median_maxdd": float(np.median(null_arr)),
        "null_range_occupancy": round(null_range_occupancy, 4),
        "null_pct_near_bound": round(null_pct_near_bound, 4),
        # GEV fits - NULL (raw track)
        "null_xi_lmom": null_lmom["xi"] if null_lmom["success"] else None,
        "null_loc_lmom": null_lmom["loc"] if null_lmom["success"] else None,
        "null_scale_lmom": null_lmom["scale"] if null_lmom["success"] else None,
        "null_xi_mle": null_mle["xi"] if null_mle["success"] else None,
        "null_xi_lmom_ci": list(null_ci[:2]) if np.isfinite(null_ci[0]) else None,
        "null_fit_lmom_success": null_lmom["success"],
        "null_fit_mle_success": null_mle["success"],
        # GEV fits - OBSERVED (raw track)
        "obs_xi_lmom": obs_lmom["xi"] if obs_lmom["success"] else None,
        "obs_loc_lmom": obs_lmom["loc"] if obs_lmom["success"] else None,
        "obs_scale_lmom": obs_lmom["scale"] if obs_lmom["success"] else None,
        "obs_xi_mle": obs_mle["xi"] if obs_mle["success"] else None,
        "obs_xi_lmom_ci": list(obs_ci[:2]) if np.isfinite(obs_ci[0]) else None,
        "obs_fit_lmom_success": obs_lmom["success"],
        "obs_fit_mle_success": obs_mle["success"],
        # GEV fits - normalized track
        "obs_xi_norm_lmom": obs_norm_lmom["xi"] if obs_norm_lmom["success"] else None,
        "null_xi_norm_lmom": null_norm_lmom["xi"] if null_norm_lmom["success"] else None,
        # Coles criterion
        "coles_criterion": coles,
        # Baseline: uniform random
        "baseline_uniform_xi_lmom": uniform_lmom["xi"] if uniform_lmom["success"] else None,
        # MLE vs L-moments agreement
        "xi_method_agreement": xi_agreement,
    }


# ============================================================
# LOAD TREEBANK METADATA FROM data_id3
# ============================================================

def load_treebank_metadata() -> dict:
    """Load treebank-level metadata from the dependency preview file."""
    treebank_meta = {}
    if not PREVIEW_FILE.exists():
        logger.warning(f"Preview file not found: {PREVIEW_FILE}")
        return treebank_meta

    data = json.loads(PREVIEW_FILE.read_text())
    for ex in data["datasets"][0]["examples"]:
        if ex.get("metadata_row_type") == "treebank":
            tb_id = ex["metadata_treebank_id"]
            treebank_meta[tb_id] = {
                "language": ex.get("metadata_language"),
                "morph_richness": ex.get("metadata_morph_richness"),
                "head_direction_ratio": ex.get("metadata_head_direction_ratio"),
                "word_order_entropy": ex.get("metadata_word_order_entropy"),
                "nonprojectivity_rate": ex.get("metadata_nonprojectivity_rate"),
            }
    # Also load from full data files for more treebanks
    full_data_dir = DEP_DATA / "data_out"
    if full_data_dir.exists():
        import glob as glob_mod
        for fp in sorted(glob_mod.glob(str(full_data_dir / "full_data_out_*.json"))):
            try:
                with open(fp) as f:
                    fd = json.load(f)
                for ex in fd["datasets"][0]["examples"]:
                    if ex.get("metadata_row_type") == "treebank":
                        tb_id = ex["metadata_treebank_id"]
                        if tb_id not in treebank_meta:
                            treebank_meta[tb_id] = {
                                "language": ex.get("metadata_language"),
                                "morph_richness": ex.get("metadata_morph_richness"),
                                "head_direction_ratio": ex.get("metadata_head_direction_ratio"),
                                "word_order_entropy": ex.get("metadata_word_order_entropy"),
                                "nonprojectivity_rate": ex.get("metadata_nonprojectivity_rate"),
                            }
            except Exception as e:
                logger.warning(f"Could not load {fp}: {e}")
    logger.info(f"Loaded metadata for {len(treebank_meta)} treebanks")
    return treebank_meta


# ============================================================
# AGGREGATE ANALYSIS AND TRACK RECOMMENDATION
# ============================================================

def aggregate_analysis(
    results_by_treebank_length: dict,
    all_null_xis: dict,
    all_obs_xis: dict,
) -> tuple[dict, dict, str]:
    """Compute summary by length and track recommendations."""

    track_recommendation = {}
    summary_by_length = {}

    for slen in TARGET_LENGTHS:
        null_xis = all_null_xis.get(slen, [])
        obs_xis = all_obs_xis.get(slen, [])

        if len(null_xis) < 2 or len(obs_xis) < 2:
            summary_by_length[str(slen)] = {"status": "insufficient_data",
                                             "n_treebanks": len(null_xis)}
            continue

        null_range = max(null_xis) - min(null_xis)
        obs_range = max(obs_xis) - min(obs_xis)
        null_min, null_max = min(null_xis), max(null_xis)
        obs_min, obs_max_val = min(obs_xis), max(obs_xis)

        # DECISION CRITERION:
        # If null xi range > 0.25: raw track viable
        # If null xi uniformly below -0.3: normalized track is primary
        raw_track_viable = null_range > 0.25
        all_strongly_negative = null_max < -0.3

        if raw_track_viable:
            recommendation = "raw_track_primary"
            rationale = (f"Null xi range = {null_range:.3f} (from {null_min:.3f} to "
                         f"{null_max:.3f}) exceeds 0.25 threshold, supporting meaningful "
                         f"cross-treebank variation in the raw track.")
        elif all_strongly_negative:
            recommendation = "normalized_track_primary"
            rationale = (f"Null xi uniformly strongly negative (max = {null_max:.3f}), "
                         f"indicating bound domination. Normalized track is more defensible.")
        else:
            recommendation = "both_tracks_report"
            rationale = (f"Null xi range = {null_range:.3f}, ambiguous. "
                         f"Report both tracks with explicit caveats.")

        # Coles pass rate
        coles_pass_count = sum(
            1 for k, v in results_by_treebank_length.items()
            if v["sentence_length"] == slen
            and v.get("coles_criterion", {}).get("passes_20pct", False)
        )
        total_at_len = sum(
            1 for k, v in results_by_treebank_length.items()
            if v["sentence_length"] == slen
        )

        summary_by_length[str(slen)] = {
            "n_treebanks": len(null_xis),
            "null_xi_range": round(null_range, 4),
            "null_xi_min": round(null_min, 4),
            "null_xi_max": round(null_max, 4),
            "null_xi_mean": round(float(np.mean(null_xis)), 4),
            "null_xi_std": round(float(np.std(null_xis)), 4),
            "obs_xi_range": round(obs_range, 4),
            "obs_xi_min": round(obs_min, 4),
            "obs_xi_max": round(obs_max_val, 4),
            "obs_xi_mean": round(float(np.mean(obs_xis)), 4),
            "coles_pass_rate": f"{coles_pass_count}/{total_at_len}",
            "recommendation": recommendation,
            "rationale": rationale,
        }
        track_recommendation[str(slen)] = recommendation

    # Overall recommendation: majority vote
    raw_votes = sum(1 for r in track_recommendation.values() if r == "raw_track_primary")
    norm_votes = sum(1 for r in track_recommendation.values() if r == "normalized_track_primary")
    if raw_votes > norm_votes:
        overall_recommendation = "raw_track_primary"
    elif norm_votes > raw_votes:
        overall_recommendation = "normalized_track_primary"
    else:
        overall_recommendation = "both_tracks_report"

    return summary_by_length, track_recommendation, overall_recommendation


# ============================================================
# FORMAT OUTPUT TO SCHEMA
# ============================================================

def format_output(
    results_by_treebank_length: dict,
    summary_by_length: dict,
    track_recommendation: dict,
    overall_recommendation: str,
    treebank_meta: dict,
    wall_time_sec: float,
) -> dict:
    """Format results into exp_gen_sol_out.json schema format."""
    examples = []

    # One example per treebank-length combination
    for key, result in sorted(results_by_treebank_length.items()):
        tb = result["treebank"]
        slen = result["sentence_length"]
        meta = treebank_meta.get(tb, {})

        # Build a descriptive output string
        output_parts = []
        output_parts.append(f"Treebank: {tb}, Length: {slen}")
        output_parts.append(f"Trees: {result['n_trees']}, Null linearizations: {result['n_null_linearizations']}")
        output_parts.append(f"Observed xi(L-mom): {result.get('obs_xi_lmom', 'N/A')}")
        output_parts.append(f"Null xi(L-mom): {result.get('null_xi_lmom', 'N/A')}")
        if result.get("obs_xi_norm_lmom") is not None:
            output_parts.append(f"Normalized obs xi: {result['obs_xi_norm_lmom']:.4f}")
        if result.get("coles_criterion", {}).get("applicable"):
            output_parts.append(f"Coles passes: {result['coles_criterion']['passes_20pct']}")

        example = {
            "input": f"{tb}__{slen}",
            "output": " | ".join(output_parts),
            # Treebank metadata
            "metadata_treebank": tb,
            "metadata_sentence_length": slen,
            "metadata_language": meta.get("language", "unknown"),
            "metadata_morph_richness": meta.get("morph_richness"),
            "metadata_head_direction_ratio": meta.get("head_direction_ratio"),
            "metadata_word_order_entropy": meta.get("word_order_entropy"),
            "metadata_nonprojectivity_rate": meta.get("nonprojectivity_rate"),
            "metadata_n_trees": result["n_trees"],
            "metadata_n_observed": result["n_observed_sentences"],
            "metadata_n_null_lins": result["n_null_linearizations"],
            "metadata_theoretical_bound": result["theoretical_bound"],
            # Descriptive stats
            "metadata_obs_mean_maxdd": result["obs_mean_maxdd"],
            "metadata_obs_std_maxdd": result["obs_std_maxdd"],
            "metadata_obs_max_maxdd": result["obs_max_maxdd"],
            "metadata_obs_range_occupancy": result["obs_range_occupancy"],
            "metadata_obs_pct_near_bound": result["obs_pct_near_bound"],
            "metadata_null_mean_maxdd": result["null_mean_maxdd"],
            "metadata_null_range_occupancy": result["null_range_occupancy"],
            "metadata_null_pct_near_bound": result["null_pct_near_bound"],
            # GEV fits
            "metadata_null_xi_lmom": result.get("null_xi_lmom"),
            "metadata_null_xi_mle": result.get("null_xi_mle"),
            "metadata_obs_xi_lmom": result.get("obs_xi_lmom"),
            "metadata_obs_xi_mle": result.get("obs_xi_mle"),
            "metadata_obs_xi_norm_lmom": result.get("obs_xi_norm_lmom"),
            "metadata_null_xi_norm_lmom": result.get("null_xi_norm_lmom"),
            "metadata_coles_passes": result.get("coles_criterion", {}).get("passes_20pct"),
            "metadata_xi_method_agreement": result.get("xi_method_agreement"),
            "metadata_baseline_uniform_xi": result.get("baseline_uniform_xi_lmom"),
            # Predictions: our method vs baseline
            "predict_null_gev_xi": str(round(result["null_xi_lmom"], 4)) if result.get("null_xi_lmom") is not None else "N/A",
            "predict_observed_gev_xi": str(round(result["obs_xi_lmom"], 4)) if result.get("obs_xi_lmom") is not None else "N/A",
            "predict_baseline_uniform_xi": str(round(result["baseline_uniform_xi_lmom"], 4)) if result.get("baseline_uniform_xi_lmom") is not None else "N/A",
        }
        examples.append(example)

    output = {
        "metadata": {
            "experiment": "bound_awareness_simulation",
            "description": ("Random projective linearization GEV null distributions for "
                            "bound-awareness validation of max dependency distance analysis"),
            "overall_track_recommendation": overall_recommendation,
            "n_treebanks_processed": len(set(r["treebank"] for r in results_by_treebank_length.values())),
            "n_lengths": len(TARGET_LENGTHS),
            "n_total_combinations": len(results_by_treebank_length),
            "total_null_linearizations": sum(
                v["n_null_linearizations"] for v in results_by_treebank_length.values()
            ),
            "wall_time_seconds": round(wall_time_sec, 1),
            "summary_by_length": summary_by_length,
            "track_recommendation_by_length": track_recommendation,
            "methodology": {
                "linearization_algorithm": "Futrell et al. 2015 recursive child-order randomization",
                "gev_primary_method": "L-moments (Hosking 1990)",
                "gev_secondary_method": "MLE (scipy.stats.genextreme)",
                "sign_convention": "xi = -c for both scipy and lmoments3",
                "bootstrap_resamples": N_BOOTSTRAP,
                "coles_threshold": "GEV upper endpoint >= 1.2 * observed max",
                "null_range_threshold": "0.25 for raw track viability",
                "baseline": "Uniform random integers in [1, n-1] (no tree structure)",
            },
        },
        "datasets": [
            {
                "dataset": "bound_awareness_gev_null",
                "examples": examples,
            }
        ],
    }
    return output


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("BOUND-AWARENESS SIMULATION: Random Projective Linearization")
    logger.info("=" * 60)

    # Load metadata
    treebank_meta = load_treebank_metadata()
    for tb in SELECTED_TREEBANKS:
        if tb in treebank_meta:
            m = treebank_meta[tb]
            logger.info(f"  {tb}: morph={m.get('morph_richness')}, "
                        f"hdr={m.get('head_direction_ratio')}, "
                        f"woe={m.get('word_order_entropy')}")
        else:
            logger.warning(f"  {tb} not in preview metadata")

    results_by_treebank_length = {}
    all_null_xis = defaultdict(list)
    all_obs_xis = defaultdict(list)

    # ================================================================
    # PHASE 1: Quick validation (2 treebanks x 2 lengths)
    # ================================================================
    logger.info("=" * 50)
    logger.info("=== PHASE 1: Quick validation ===")
    logger.info("=" * 50)
    PHASE1_TBS = ["en_ewt", "tr_imst"]
    PHASE1_LENS = [10, 14]
    PHASE1_TREES = 20
    PHASE1_LINS = 50

    phase1_ok = True
    for tb_name in PHASE1_TBS:
        full_tb = load_treebank_from_hf(tb_name)
        if full_tb is None:
            fb = FALLBACKS.get(tb_name)
            if fb and fb != tb_name:
                logger.info(f"  Trying fallback: {fb}")
                full_tb = load_treebank_from_hf(fb)
                if full_tb is not None:
                    tb_name = fb
            if full_tb is None:
                logger.error(f"  PHASE 1 FAIL: Could not load {tb_name}")
                phase1_ok = False
                continue

        for target_len in PHASE1_LENS:
            trees, obs_max_dds = extract_trees_for_length(full_tb, target_len, PHASE1_TREES)
            logger.info(f"  Phase1: {tb_name} len={target_len}: {len(trees)} trees")

            if len(trees) < 5:
                logger.warning(f"  Phase1: Too few trees for {tb_name} len={target_len}")
                continue

            result = process_treebank_length(
                tb_name, trees, obs_max_dds, target_len,
                n_lins=PHASE1_LINS, n_boot=100, phase_label="phase1"
            )
            if result is None:
                logger.error(f"  Phase1: Failed for {tb_name} len={target_len}")
                phase1_ok = False
                continue

            # Validate results
            if result.get("null_xi_lmom") is None:
                logger.error(f"  Phase1: null_xi_lmom is None for {tb_name} len={target_len}")
                phase1_ok = False
            elif not np.isfinite(result["null_xi_lmom"]):
                logger.error(f"  Phase1: null_xi_lmom not finite")
                phase1_ok = False
            else:
                logger.info(f"  Phase1 OK: null_xi={result['null_xi_lmom']:.4f}, "
                            f"obs_xi={result.get('obs_xi_lmom', 'N/A')}")

        del full_tb
        gc.collect()

    if not phase1_ok:
        logger.warning("Phase 1 had some issues, but continuing to Phase 2...")

    elapsed = time.time() - start_time
    logger.info(f"Phase 1 completed in {elapsed:.1f}s")

    # ================================================================
    # PHASE 2: Full simulation (10 treebanks x 6 lengths)
    # ================================================================
    logger.info("=" * 50)
    logger.info("=== PHASE 2: Full simulation ===")
    logger.info("=" * 50)

    tb_times = []
    for tb_idx, tb_name in enumerate(SELECTED_TREEBANKS):
        tb_start = time.time()
        logger.info(f"\n--- [{tb_idx+1}/{len(SELECTED_TREEBANKS)}] Treebank: {tb_name} ---")

        # Check time budget
        elapsed = time.time() - start_time
        remaining = 3000 - elapsed  # 50 min budget
        if remaining < 120:
            logger.warning(f"Time budget low ({remaining:.0f}s remaining), saving partial results")
            break

        # Adaptive: if previous treebanks took too long, reduce N_LINEARIZATIONS
        current_n_lins = N_LINEARIZATIONS
        if tb_times and np.mean(tb_times) > 420:  # > 7 min per TB
            current_n_lins = 200
            logger.warning(f"Reducing N_LINEARIZATIONS to {current_n_lins} due to time pressure")

        # Load treebank
        full_tb = load_treebank_from_hf(tb_name)
        actual_tb_name = tb_name
        if full_tb is None:
            fb = FALLBACKS.get(tb_name)
            if fb and fb != tb_name:
                logger.info(f"  Trying fallback: {fb}")
                full_tb = load_treebank_from_hf(fb)
                if full_tb is not None:
                    actual_tb_name = fb
            if full_tb is None:
                logger.error(f"  Skipping {tb_name} entirely")
                continue

        for target_len in TARGET_LENGTHS:
            trees, obs_max_dds = extract_trees_for_length(
                full_tb, target_len, MAX_TREES_PER_BIN
            )

            if len(trees) < 10:
                logger.warning(f"  Only {len(trees)} trees for {actual_tb_name} "
                               f"len={target_len}, skipping")
                continue

            result = process_treebank_length(
                actual_tb_name, trees, obs_max_dds, target_len,
                n_lins=current_n_lins, n_boot=N_BOOTSTRAP, phase_label="phase2"
            )
            if result is None:
                continue

            key = f"{actual_tb_name}__{target_len}"
            results_by_treebank_length[key] = result

            # Accumulate for cross-treebank analysis
            if result.get("null_xi_lmom") is not None and np.isfinite(result["null_xi_lmom"]):
                all_null_xis[target_len].append(result["null_xi_lmom"])
            if result.get("obs_xi_lmom") is not None and np.isfinite(result["obs_xi_lmom"]):
                all_obs_xis[target_len].append(result["obs_xi_lmom"])

        # Free memory
        del full_tb
        gc.collect()

        tb_elapsed = time.time() - tb_start
        tb_times.append(tb_elapsed)
        logger.info(f"  Treebank {actual_tb_name} done in {tb_elapsed:.1f}s")

        # Log memory
        try:
            usage_bytes = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
            logger.info(f"  Memory usage: {usage_bytes / 1e9:.2f} GB / {TOTAL_RAM_GB:.1f} GB")
        except (FileNotFoundError, ValueError):
            pass

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    logger.info("=" * 50)
    logger.info("=== AGGREGATE ANALYSIS ===")
    logger.info("=" * 50)

    summary_by_length, track_recommendation, overall_recommendation = aggregate_analysis(
        results_by_treebank_length, dict(all_null_xis), dict(all_obs_xis)
    )

    logger.info(f"Overall track recommendation: {overall_recommendation}")
    for slen, summary in summary_by_length.items():
        logger.info(f"  Length {slen}: {summary.get('recommendation', 'N/A')} "
                    f"(null xi range: {summary.get('null_xi_range', 'N/A')})")

    # ================================================================
    # WRITE OUTPUT
    # ================================================================
    wall_time = time.time() - start_time
    logger.info(f"Total wall time: {wall_time:.1f}s")

    output = format_output(
        results_by_treebank_length, summary_by_length,
        track_recommendation, overall_recommendation,
        treebank_meta, wall_time,
    )

    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")

    # Summary stats
    logger.info(f"Results: {len(results_by_treebank_length)} treebank-length combinations")
    logger.info(f"Total null linearizations: "
                f"{sum(v['n_null_linearizations'] for v in results_by_treebank_length.values())}")
    logger.info(f"Overall recommendation: {overall_recommendation}")

    return output


if __name__ == "__main__":
    main()
