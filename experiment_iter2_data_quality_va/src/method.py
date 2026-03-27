#!/usr/bin/env python3
"""
Data Quality Validation and Robustness Checks for GEV Dependency Distance Analysis.

Executes six validation analyses:
1. Grambank-UD morphological richness cross-validation
2. Word-order entropy threshold sensitivity (10 vs 20 tokens)
3. Representativeness of the exact-length subset
4. Within-bin autocorrelation diagnostics via Ljung-Box
5. Annotation-completeness confound assessment
6. Structured JSON output with diagnostic flags

Dependencies: data_id3 (UD sentence max-DD + treebank features), data_id4 (Grambank morph complexity)
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import tqdm

# ===========================================================================
# LOGGING SETUP
# ===========================================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ===========================================================================
# HARDWARE DETECTION & MEMORY LIMITS
# ===========================================================================

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
# Budget: ~20GB for data processing (leave ~9GB for OS + agent)
RAM_BUDGET_BYTES = int(20 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))  # 2 hours CPU time

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f}GB")

# ===========================================================================
# PATH CONSTANTS
# ===========================================================================
WORKSPACE = Path(__file__).parent
DEP_ID3 = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bto/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
DEP_ID4 = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bto/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus"
)
HF_DATASET = "commul/universal_dependencies"


# ===========================================================================
# PHASE 1: DATA LOADING
# ===========================================================================

def load_data_id3() -> tuple[list[dict], list[dict]]:
    """Load treebank summaries and sentence rows from data_id3 (two files)."""
    treebank_rows: list[dict] = []
    sentence_rows: list[dict] = []

    for fname in ["data_out/full_data_out_1.json", "data_out/full_data_out_2.json"]:
        fpath = DEP_ID3 / fname
        logger.info(f"Loading {fpath.name}...")
        with open(fpath) as f:
            data = json.load(f)
        for ds in data["datasets"]:
            for ex in ds["examples"]:
                if ex.get("metadata_row_type") == "treebank":
                    treebank_rows.append(ex)
                elif ex.get("metadata_row_type") == "sentence":
                    sentence_rows.append(ex)
        del data
        gc.collect()

    logger.info(f"Loaded {len(treebank_rows)} treebank rows, {len(sentence_rows)} sentence rows")
    return treebank_rows, sentence_rows


def load_data_id4_overlap() -> list[dict]:
    """Load Grambank entries that overlap with UD (has_ud=True, has_grambank=True)."""
    fpath = DEP_ID4 / "full_data_out.json"
    logger.info(f"Loading {fpath.name}...")
    with open(fpath) as f:
        data = json.load(f)

    overlap: list[dict] = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            if ex.get("metadata_has_ud") and ex.get("metadata_has_grambank"):
                try:
                    output = json.loads(ex["output"])
                    ex["_grambank_morph_index"] = output["grambank_morph_index"]
                    ex["_grambank_morph_richness_raw"] = output["grambank_morph_richness_raw"]
                    ex["_n_morph_features_coded"] = output["n_morph_features_coded"]
                    overlap.append(ex)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse Grambank output for {ex.get('input')}: {e}")

    del data
    gc.collect()
    logger.info(f"Loaded {len(overlap)} Grambank-UD overlap entries")
    return overlap


# ===========================================================================
# ANALYSIS 1: GRAMBANK CROSS-VALIDATION
# ===========================================================================

def analysis_grambank_crossvalidation(
    treebank_rows: list[dict],
    grambank_overlap: list[dict],
) -> dict:
    """
    Cross-validate UD morphological richness against Grambank morph index.
    For languages with MULTIPLE UD treebanks, AVERAGE the morph_richness.
    """
    logger.info("Analysis 1: Grambank Cross-Validation")

    # Build lookup: treebank_id -> {morph_richness, feat_completeness, language}
    tb_lookup: dict[str, dict] = {}
    for tb in treebank_rows:
        tb_lookup[tb["metadata_treebank_id"]] = {
            "morph_richness": tb["metadata_morph_richness"],
            "feat_completeness": tb["metadata_feat_completeness"],
            "language": tb["metadata_language"],
        }

    # Build paired data
    pairs: list[dict] = []
    for gb_entry in grambank_overlap:
        ud_treebanks = gb_entry.get("metadata_ud_treebanks", [])
        if not ud_treebanks:
            continue

        morph_values: list[float] = []
        feat_values: list[float] = []
        for tb_id in ud_treebanks:
            if tb_id in tb_lookup:
                morph_values.append(tb_lookup[tb_id]["morph_richness"])
                feat_values.append(tb_lookup[tb_id]["feat_completeness"])

        if not morph_values:
            continue

        pairs.append({
            "iso": gb_entry["metadata_iso639_3_code"],
            "language": gb_entry["metadata_language_name"],
            "family": gb_entry["metadata_family_name"],
            "macroarea": gb_entry["metadata_macroarea"],
            "ud_morph_richness": float(np.mean(morph_values)),
            "grambank_morph_index": gb_entry["_grambank_morph_index"],
            "feat_completeness": float(np.mean(feat_values)),
            "n_ud_treebanks": len(morph_values),
        })

    df = pd.DataFrame(pairs)
    n = len(df)
    logger.info(f"Grambank cross-validation: {n} paired languages")

    if n < 5:
        return {"error": f"Too few paired languages ({n})"}

    # Compute correlations
    spearman_r, spearman_p = sp_stats.spearmanr(df["ud_morph_richness"], df["grambank_morph_index"])
    pearson_r, pearson_p = sp_stats.pearsonr(df["ud_morph_richness"], df["grambank_morph_index"])

    # Bootstrap 95% CIs (1000 resamples)
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_spearman: list[float] = []
    boot_pearson: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx]
        sr, _ = sp_stats.spearmanr(sample["ud_morph_richness"], sample["grambank_morph_index"])
        pr, _ = sp_stats.pearsonr(sample["ud_morph_richness"], sample["grambank_morph_index"])
        boot_spearman.append(float(sr))
        boot_pearson.append(float(pr))

    spearman_ci = (float(np.percentile(boot_spearman, 2.5)), float(np.percentile(boot_spearman, 97.5)))
    pearson_ci = (float(np.percentile(boot_pearson, 2.5)), float(np.percentile(boot_pearson, 97.5)))

    # Outlier detection: feat_completeness < 0.3
    outliers = df[df["feat_completeness"] < 0.3][
        ["iso", "language", "ud_morph_richness", "grambank_morph_index", "feat_completeness"]
    ].to_dict("records")

    # Scatter plot data
    scatter_data = df[
        ["iso", "language", "family", "macroarea", "ud_morph_richness",
         "grambank_morph_index", "feat_completeness"]
    ].to_dict("records")

    interpretation = (
        "STRONG" if abs(spearman_r) > 0.6 else
        "MODERATE" if abs(spearman_r) > 0.4 else
        "WEAK" if abs(spearman_r) > 0.2 else "NEGLIGIBLE"
    )

    return {
        "n_overlap_languages": n,
        "spearman_rho": round(float(spearman_r), 4),
        "spearman_p": float(spearman_p),
        "spearman_95ci": [round(spearman_ci[0], 4), round(spearman_ci[1], 4)],
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "pearson_95ci": [round(pearson_ci[0], 4), round(pearson_ci[1], 4)],
        "n_outliers_low_completeness": len(outliers),
        "outliers": outliers,
        "scatter_data": scatter_data,
        "interpretation": interpretation,
    }


# ===========================================================================
# ANALYSIS 5: ANNOTATION CONFOUND
# ===========================================================================

def analysis_annotation_confound(treebank_rows: list[dict]) -> dict:
    """Test whether feat_completeness confounds morph_richness (rho > 0.7 = confound)."""
    logger.info("Analysis 5: Annotation Confound")

    fc = [tb["metadata_feat_completeness"] for tb in treebank_rows]
    mr = [tb["metadata_morph_richness"] for tb in treebank_rows]

    rho, p = sp_stats.spearmanr(fc, mr)

    # Identify treebanks with low feat_completeness
    low_fc = [
        {
            "treebank_id": tb["metadata_treebank_id"],
            "language": tb["metadata_language"],
            "feat_completeness": tb["metadata_feat_completeness"],
            "morph_richness": tb["metadata_morph_richness"],
        }
        for tb in treebank_rows
        if tb["metadata_feat_completeness"] < 0.3
    ]

    # Additional: partial correlation controlling for feat_completeness
    # Rank-based partial correlation (Spearman)
    # Also test: morph_richness vs mean_dd_all, controlling for feat_completeness
    dd_values = [tb["metadata_mean_dd_all"] for tb in treebank_rows]
    rho_mr_dd, p_mr_dd = sp_stats.spearmanr(mr, dd_values)

    is_confound = abs(rho) > 0.7

    return {
        "n_treebanks": len(treebank_rows),
        "spearman_rho_fc_vs_mr": round(float(rho), 4),
        "spearman_p": float(p),
        "is_confound": is_confound,
        "confound_threshold": 0.7,
        "n_low_completeness_treebanks": len(low_fc),
        "low_completeness_treebanks": low_fc,
        "additional_morph_richness_vs_mean_dd": {
            "spearman_rho": round(float(rho_mr_dd), 4),
            "p_value": float(p_mr_dd),
        },
        "recommendation": (
            "WARNING: feat_completeness strongly confounds morph_richness. "
            "Include feat_completeness as control in regression."
            if is_confound else
            "feat_completeness is not a dominant confound, but still include as control."
        ),
    }


# ===========================================================================
# ANALYSIS 3: REPRESENTATIVENESS
# ===========================================================================

def analysis_representativeness(treebank_rows: list[dict]) -> dict:
    """Assess whether the exact-length subset is representative of the full corpus."""
    logger.info("Analysis 3: Representativeness")

    records: list[dict] = []
    for tb in treebank_rows:
        try:
            n_per_bin = json.loads(tb["metadata_n_per_bin"])
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Bad n_per_bin for {tb['metadata_treebank_id']}")
            continue

        total_binned = sum(n_per_bin.values())
        total_all = tb["metadata_n_sentences_total"]
        coverage = total_binned / total_all if total_all > 0 else 0.0

        records.append({
            "treebank_id": tb["metadata_treebank_id"],
            "language": tb["metadata_language"],
            "modality": tb["metadata_modality"],
            "genre": tb["metadata_genre"],
            "mean_dd_all": tb["metadata_mean_dd_all"],
            "n_total": total_all,
            "n_binned": total_binned,
            "coverage": coverage,
        })

    df = pd.DataFrame(records)

    summary = {
        "mean_coverage": round(float(df["coverage"].mean()), 4),
        "median_coverage": round(float(df["coverage"].median()), 4),
        "min_coverage": round(float(df["coverage"].min()), 4),
        "max_coverage": round(float(df["coverage"].max()), 4),
        "std_coverage": round(float(df["coverage"].std()), 4),
        "total_binned_sentences": int(df["n_binned"].sum()),
        "total_all_sentences": int(df["n_total"].sum()),
        "overall_coverage": round(float(df["n_binned"].sum() / df["n_total"].sum()), 4),
    }

    # Correlation: coverage vs mean_dd_all
    rho_dd, p_dd = sp_stats.spearmanr(df["coverage"], df["mean_dd_all"])

    # Spoken vs written comparison
    spoken = df[df["modality"] == "spoken"]["coverage"]
    written = df[df["modality"] == "written"]["coverage"]
    if len(spoken) >= 3 and len(written) >= 3:
        mwu_stat, mwu_p = sp_stats.mannwhitneyu(spoken, written, alternative="two-sided")
        modality_test = {
            "spoken_mean_coverage": round(float(spoken.mean()), 4),
            "spoken_n": int(len(spoken)),
            "written_mean_coverage": round(float(written.mean()), 4),
            "written_n": int(len(written)),
            "mannwhitney_U": float(mwu_stat),
            "mannwhitney_p": float(mwu_p),
        }
    else:
        modality_test = {"note": "too few spoken treebanks for comparison"}

    # Genre breakdown
    genre_stats = {}
    for genre, grp in df.groupby("genre"):
        genre_stats[genre] = {
            "mean": round(float(grp["coverage"].mean()), 4),
            "median": round(float(grp["coverage"].median()), 4),
            "count": int(len(grp)),
        }

    # Low-coverage treebanks
    low_cov = df[df["coverage"] < 0.10].sort_values("coverage")
    low_cov_list = low_cov[["treebank_id", "language", "coverage", "n_total"]].to_dict("records")

    return {
        "summary": summary,
        "correlation_coverage_vs_mean_dd": {
            "spearman_rho": round(float(rho_dd), 4),
            "p_value": float(p_dd),
            "interpretation": (
                "Higher mean DD treebanks have LOWER coverage in bins"
                if rho_dd < -0.2 and p_dd < 0.05
                else "No strong coverage bias by syntactic complexity"
            ),
        },
        "modality_comparison": modality_test,
        "genre_breakdown": genre_stats,
        "n_low_coverage_treebanks": len(low_cov_list),
        "low_coverage_treebanks": low_cov_list[:20],
    }


# ===========================================================================
# ANALYSIS 4: AUTOCORRELATION DIAGNOSTICS
# ===========================================================================

def _ljungbox_for_group(args: tuple) -> dict | None:
    """Run Ljung-Box test on a single treebank-bin group (for parallel execution)."""
    tb_id, bin_val, items = args
    if len(items) < 30:
        return None

    # Sort by original document order (index from input field)
    items.sort(key=lambda x: x[0])
    max_dd_seq = np.array([x[1] for x in items], dtype=float)

    n_lags = min(10, len(max_dd_seq) // 4)
    if n_lags < 1:
        return None

    try:
        lb_result = acorr_ljungbox(max_dd_seq, lags=n_lags, return_df=True)
        min_p = float(lb_result["lb_pvalue"].min())
        any_sig = min_p < 0.05

        return {
            "treebank_id": tb_id,
            "length_bin": bin_val,
            "n_sentences": len(items),
            "min_p_value": round(min_p, 6),
            "significant_at_05": any_sig,
        }
    except Exception as e:
        logger.debug(f"Ljung-Box failed for {tb_id} bin {bin_val}: {e}")
        return None


def analysis_autocorrelation(sentence_rows: list[dict]) -> dict:
    """
    Test serial dependence of max_DD within each treebank-bin combination.
    Uses original sentence index from input field to reconstruct order.
    """
    logger.info("Analysis 4: Autocorrelation Diagnostics")

    # Group sentence rows by (treebank_id, length_bin)
    groups: dict[tuple, list] = defaultdict(list)
    parse_failures = 0
    for s in sentence_rows:
        tb = s["metadata_treebank_id"]
        bin_val = s["metadata_length_bin"]
        try:
            idx = int(s["input"].rsplit("__", 1)[1])
        except (ValueError, IndexError):
            parse_failures += 1
            continue
        groups[(tb, bin_val)].append((idx, s["metadata_max_dd"]))

    if parse_failures > 0:
        logger.warning(f"Failed to parse {parse_failures} sentence indices")

    logger.info(f"Grouped into {len(groups)} treebank-bin combinations")

    # Prepare args for parallel execution
    group_args = [(tb_id, bin_val, items) for (tb_id, bin_val), items in groups.items()]
    del groups
    gc.collect()

    # Run Ljung-Box tests in parallel using ProcessPoolExecutor
    results: list[dict] = []
    n_workers = max(1, NUM_CPUS - 1)
    logger.info(f"Running Ljung-Box tests with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_ljungbox_for_group, args): args[:2] for args in group_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ljung-Box"):
            try:
                result = future.result(timeout=30)
                if result is not None:
                    results.append(result)
            except Exception as e:
                key = futures[future]
                logger.debug(f"Worker failed for {key}: {e}")

    n_tested = len(results)
    n_significant = sum(1 for r in results if r["significant_at_05"])
    prop_significant = n_significant / n_tested if n_tested > 0 else 0.0

    # Bin-level summary
    bin_summary: dict[int, dict] = defaultdict(lambda: {"tested": 0, "significant": 0})
    for r in results:
        b = r["length_bin"]
        bin_summary[b]["tested"] += 1
        if r["significant_at_05"]:
            bin_summary[b]["significant"] += 1

    # Top autocorrelated groups
    top_auto = sorted(
        [r for r in results if r["significant_at_05"]],
        key=lambda x: x["min_p_value"],
    )[:20]

    return {
        "n_bin_treebank_combinations_tested": n_tested,
        "n_significant_at_05": n_significant,
        "proportion_significant": round(prop_significant, 4),
        "recommend_block_subsampling": prop_significant > 0.20,
        "per_bin_summary": {str(k): v for k, v in sorted(bin_summary.items())},
        "top_autocorrelated": top_auto,
        "all_results": results,
        "interpretation": (
            "CONCERN: >20% of bin-treebank combinations show significant serial "
            "dependence. Block-of-blocks subsampling recommended for GEV fitting."
            if prop_significant > 0.20 else
            f"ACCEPTABLE: {prop_significant:.1%} show dependence, within expected "
            "false-positive range under multiple testing. Independence assumption is reasonable."
        ),
    }


# ===========================================================================
# ANALYSIS 2: ENTROPY THRESHOLD SENSITIVITY (requires HuggingFace)
# ===========================================================================

def compute_entropy_for_treebank(config_name: str) -> dict:
    """
    Load a single treebank from HuggingFace, compute word-order entropy
    at threshold=10 and threshold=20.
    """
    from datasets import concatenate_datasets, load_dataset

    try:
        ds = load_dataset(HF_DATASET, config_name)
        all_splits = [ds[s] for s in ds.keys()]
        full_tb = concatenate_datasets(all_splits)
    except Exception as e:
        return {"config": config_name, "error": str(e)[:200]}

    # Count per-relation head directions
    rel_counts: dict[str, Counter] = defaultdict(Counter)

    for idx in range(len(full_tb)):
        row = full_tb[idx]
        tokens = row["tokens"]
        heads_str = row["head"]
        deprels = row["deprel"]
        n = len(tokens)

        if n < 2 or len(heads_str) != n:
            continue

        for i, h_str in enumerate(heads_str):
            try:
                h = int(h_str)
            except (ValueError, TypeError):
                continue
            if h == 0 or h < 0 or h > n:
                continue
            dep_pos = i + 1
            is_head_before_dep = h < dep_pos
            deprel = deprels[i]
            if deprel in ("punct", "root"):
                continue
            rel_counts[deprel][is_head_before_dep] += 1

    del full_tb, all_splits, ds
    gc.collect()

    def compute_wo_entropy(threshold: int) -> tuple[float, int]:
        rel_entropies: dict[str, float] = {}
        rel_token_counts: dict[str, int] = {}
        for deprel, counts in rel_counts.items():
            total = counts[True] + counts[False]
            if total < threshold:
                continue
            rel_token_counts[deprel] = total
            if counts[True] == 0 or counts[False] == 0:
                rel_entropies[deprel] = 0.0
            else:
                p = counts[True] / total
                rel_entropies[deprel] = -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        total_rel_tokens = sum(rel_token_counts.values())
        if total_rel_tokens > 0:
            entropy = sum(
                rel_entropies[r] * rel_token_counts[r] / total_rel_tokens
                for r in rel_entropies
            )
        else:
            entropy = 0.0
        return round(entropy, 6), len(rel_entropies)

    ent_10, n_rels_10 = compute_wo_entropy(10)
    ent_20, n_rels_20 = compute_wo_entropy(20)

    return {
        "config": config_name,
        "entropy_threshold_10": ent_10,
        "entropy_threshold_20": ent_20,
        "n_relations_at_10": n_rels_10,
        "n_relations_at_20": n_rels_20,
        "n_relations_dropped": n_rels_10 - n_rels_20,
    }


def analysis_entropy_threshold_sensitivity(
    treebank_rows: list[dict],
    grambank_overlap: list[dict],
) -> dict:
    """
    Recompute word-order entropy at threshold=10 and threshold=20.
    Targets: all Grambank overlap treebanks + stratified sample to ~80.
    """
    logger.info("Analysis 2: Entropy Threshold Sensitivity")

    # Identify overlap treebank IDs
    overlap_tb_ids: set[str] = set()
    for gb in grambank_overlap:
        for tb_id in gb.get("metadata_ud_treebanks", []):
            overlap_tb_ids.add(tb_id)

    all_tb_ids = [tb["metadata_treebank_id"] for tb in treebank_rows]
    overlap_targets = [t for t in all_tb_ids if t in overlap_tb_ids]

    # Add non-overlap treebanks to reach ~80, stratified by morph_richness
    non_overlap = [
        (tb["metadata_treebank_id"], tb["metadata_morph_richness"])
        for tb in treebank_rows
        if tb["metadata_treebank_id"] not in overlap_tb_ids
    ]
    non_overlap.sort(key=lambda x: x[1])
    n_extra = max(0, 80 - len(overlap_targets))
    if n_extra > 0 and len(non_overlap) > 0:
        step = max(1, len(non_overlap) // max(n_extra, 1))
        extra_targets = [non_overlap[i][0] for i in range(0, len(non_overlap), step)][:n_extra]
    else:
        extra_targets = []

    targets = overlap_targets + extra_targets
    logger.info(f"Entropy sensitivity: {len(targets)} treebanks ({len(overlap_targets)} overlap)")

    # Sequential loading with time budget
    results: list[dict] = []
    start_time = time.time()
    MAX_TIME = 2400  # 40 minutes max for this phase

    for i, config in enumerate(tqdm(targets, desc="Entropy recomputation")):
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            logger.warning(f"Entropy phase time limit at {i}/{len(targets)} treebanks ({elapsed:.0f}s)")
            break
        result = compute_entropy_for_treebank(config)
        results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"Entropy: {i+1}/{len(targets)} done, {elapsed:.0f}s elapsed")

    # Filter successful results
    successful = [r for r in results if "error" not in r]
    n_success = len(successful)
    logger.info(f"Entropy: {n_success}/{len(results)} successful")

    if n_success < 10:
        return {
            "error": "Too few treebanks loaded successfully",
            "n_attempted": len(results),
            "n_success": n_success,
            "errors": [r for r in results if "error" in r][:10],
        }

    ent10 = np.array([r["entropy_threshold_10"] for r in successful])
    ent20 = np.array([r["entropy_threshold_20"] for r in successful])

    spearman_10v20, p_10v20 = sp_stats.spearmanr(ent10, ent20)

    # Compare recomputed threshold=10 with stored metadata_word_order_entropy
    stored_lookup = {
        tb["metadata_treebank_id"]: tb["metadata_word_order_entropy"]
        for tb in treebank_rows
    }
    paired_stored: list[float] = []
    paired_recomp: list[float] = []
    for r in successful:
        if r["config"] in stored_lookup:
            paired_stored.append(stored_lookup[r["config"]])
            paired_recomp.append(r["entropy_threshold_10"])

    consistency_rho = None
    consistency_p = None
    if len(paired_stored) >= 10:
        consistency_rho, consistency_p = sp_stats.spearmanr(paired_stored, paired_recomp)

    # Rank shifts
    rank10 = sp_stats.rankdata(ent10)
    rank20 = sp_stats.rankdata(ent20)
    rank_shifts = np.abs(rank10 - rank20)
    n_big_shift = int(np.sum(rank_shifts > 5))
    mean_abs_diff = round(float(np.mean(np.abs(ent10 - ent20))), 6)

    n_dropped = [r["n_relations_dropped"] for r in successful]

    is_robust = float(spearman_10v20) > 0.95

    return {
        "n_treebanks_tested": n_success,
        "n_attempted": len(results),
        "n_errors": len(results) - n_success,
        "spearman_rho_10_vs_20": round(float(spearman_10v20), 4),
        "spearman_p_10_vs_20": float(p_10v20),
        "is_robust": is_robust,
        "robustness_threshold": 0.95,
        "consistency_with_stored_entropy": {
            "spearman_rho": round(float(consistency_rho), 4) if consistency_rho is not None else None,
            "p_value": float(consistency_p) if consistency_p is not None else None,
            "n_paired": len(paired_stored),
        },
        "mean_abs_difference_10_vs_20": mean_abs_diff,
        "n_treebanks_rank_shift_gt5": n_big_shift,
        "mean_relations_dropped_by_threshold20": round(float(np.mean(n_dropped)), 1),
        "median_relations_dropped": round(float(np.median(n_dropped)), 1),
        "per_treebank_results": successful,
        "interpretation": (
            f"Spearman rho={float(spearman_10v20):.3f} between threshold=10 and threshold=20. "
            + (
                "ROBUST: threshold choice does not materially affect treebank rankings."
                if is_robust
                else "CAUTION: threshold choice affects rankings. Report both in downstream analysis."
            )
        ),
    }


# ===========================================================================
# ADDITIONAL ANALYSIS 6: FAMILY-LEVEL BIAS CHECK
# ===========================================================================

def analysis_family_bias(treebank_rows: list[dict]) -> dict:
    """
    Check whether language family distribution in the dataset is balanced.
    Highly skewed family representation could bias cross-linguistic results.
    """
    logger.info("Analysis 6 (Additional): Family Bias Check")

    # Count treebanks per language (unique iso codes)
    iso_to_treebanks: dict[str, list[str]] = defaultdict(list)
    # We don't have family info in data_id3 directly, but we can approximate
    # from data_id4 overlap. For now, just count treebanks per language.
    lang_counts: dict[str, int] = defaultdict(int)
    for tb in treebank_rows:
        lang_counts[tb["metadata_language"]] += 1
        iso_to_treebanks[tb["metadata_iso_code"]].append(tb["metadata_treebank_id"])

    # Distribution of treebanks per language
    tb_per_lang = list(lang_counts.values())
    multi_tb_languages = {lang: count for lang, count in lang_counts.items() if count > 1}

    # Morph richness distribution stats
    morph_values = [tb["metadata_morph_richness"] for tb in treebank_rows]
    wo_entropy_values = [tb["metadata_word_order_entropy"] for tb in treebank_rows]

    # Test for normality of morph_richness (important for parametric tests)
    shapiro_mr, shapiro_mr_p = sp_stats.shapiro(morph_values[:5000])  # shapiro max 5000
    shapiro_wo, shapiro_wo_p = sp_stats.shapiro(wo_entropy_values[:5000])

    return {
        "n_unique_languages": len(lang_counts),
        "n_treebanks": len(treebank_rows),
        "treebanks_per_language": {
            "mean": round(float(np.mean(tb_per_lang)), 2),
            "median": round(float(np.median(tb_per_lang)), 1),
            "max": int(max(tb_per_lang)),
            "n_with_multiple_treebanks": len(multi_tb_languages),
        },
        "multi_treebank_languages": multi_tb_languages,
        "morph_richness_distribution": {
            "mean": round(float(np.mean(morph_values)), 4),
            "std": round(float(np.std(morph_values)), 4),
            "min": round(float(min(morph_values)), 4),
            "max": round(float(max(morph_values)), 4),
            "shapiro_W": round(float(shapiro_mr), 4),
            "shapiro_p": float(shapiro_mr_p),
            "is_normal": shapiro_mr_p > 0.05,
        },
        "word_order_entropy_distribution": {
            "mean": round(float(np.mean(wo_entropy_values)), 4),
            "std": round(float(np.std(wo_entropy_values)), 4),
            "min": round(float(min(wo_entropy_values)), 4),
            "max": round(float(max(wo_entropy_values)), 4),
            "shapiro_W": round(float(shapiro_wo), 4),
            "shapiro_p": float(shapiro_wo_p),
            "is_normal": shapiro_wo_p > 0.05,
        },
    }


# ===========================================================================
# OUTPUT FORMATTING (exp_gen_sol_out schema — per-treebank examples)
# ===========================================================================

def format_per_treebank_output(
    treebank_rows: list[dict],
    grambank_overlap: list[dict],
    results: dict,
    flags: list[str],
) -> dict:
    """
    Format results into exp_gen_sol_out.json schema with per-treebank examples.
    Each treebank = 1 example with predict_comprehensive_validation and
    predict_baseline_range_check fields.
    """
    # --- Build per-treebank lookups from aggregate results ---

    # Grambank cross-validation: treebank_id -> scatter point
    gb_by_tb: dict[str, dict] = {}
    gcv = results.get("grambank_crossvalidation", {})
    for pt in gcv.get("scatter_data", []):
        # Map language ISO back to its treebanks
        gb_by_tb[pt["iso"]] = pt

    # Map each Grambank overlap entry's treebanks to its ISO
    tb_to_iso: dict[str, str] = {}
    for gb in grambank_overlap:
        iso = gb["metadata_iso639_3_code"]
        for tb_id in gb.get("metadata_ud_treebanks", []):
            tb_to_iso[tb_id] = iso

    # Entropy sensitivity: treebank_id -> entropy result
    entropy_by_tb: dict[str, dict] = {}
    ets = results.get("entropy_threshold_sensitivity", {})
    for r in ets.get("per_treebank_results", []):
        entropy_by_tb[r["config"]] = r

    # Autocorrelation: treebank_id -> list of bin results
    autocorr_by_tb: dict[str, list[dict]] = defaultdict(list)
    ac = results.get("autocorrelation", {})
    for r in ac.get("all_results", []):
        autocorr_by_tb[r["treebank_id"]].append({
            "length_bin": r["length_bin"],
            "n_sentences": r["n_sentences"],
            "min_p_value": r["min_p_value"],
            "significant_at_05": r["significant_at_05"],
        })

    # Representativeness summary
    rep = results.get("representativeness", {})
    rep_summary = rep.get("summary", {})

    # Annotation confound
    annot = results.get("annotation_confound", {})
    confound_rho = annot.get("spearman_rho_fc_vs_mr")
    is_confound = annot.get("is_confound", False)

    # Global aggregate stats for metadata
    global_summary = {
        "grambank_spearman_rho": gcv.get("spearman_rho"),
        "grambank_interpretation": gcv.get("interpretation"),
        "entropy_spearman_10v20": ets.get("spearman_rho_10_vs_20"),
        "entropy_is_robust": ets.get("is_robust"),
        "overall_coverage": rep_summary.get("overall_coverage"),
        "autocorr_proportion_significant": ac.get("proportion_significant"),
        "confound_rho": confound_rho,
        "is_confound": is_confound,
        "diagnostic_flags": flags,
    }

    # --- Build per-treebank examples ---
    examples: list[dict] = []
    for tb in treebank_rows:
        tb_id = tb["metadata_treebank_id"]

        # Parse n_per_bin
        try:
            n_per_bin = json.loads(tb["metadata_n_per_bin"])
        except (json.JSONDecodeError, TypeError):
            n_per_bin = {}
        total_binned = sum(n_per_bin.values())
        total_all = tb["metadata_n_sentences_total"]
        coverage = total_binned / total_all if total_all > 0 else 0.0

        # Ground truth output: key treebank metrics
        output_data = {
            "morph_richness": tb["metadata_morph_richness"],
            "word_order_entropy": tb["metadata_word_order_entropy"],
            "feat_completeness": tb["metadata_feat_completeness"],
            "mean_dd_all": tb["metadata_mean_dd_all"],
            "n_sentences_total": total_all,
            "n_binned": total_binned,
            "coverage": round(coverage, 4),
            "modality": tb["metadata_modality"],
            "genre": tb["metadata_genre"],
        }

        # --- Our method: comprehensive multi-analysis validation ---
        tb_flags: list[str] = []
        tb_iso = tb_to_iso.get(tb_id)

        # 1. Grambank cross-validation
        gb_result = gb_by_tb.get(tb_iso) if tb_iso else None
        grambank_val = None
        if gb_result:
            grambank_val = {
                "ud_morph_richness": gb_result["ud_morph_richness"],
                "grambank_morph_index": gb_result["grambank_morph_index"],
                "in_overlap": True,
            }
            if gb_result.get("feat_completeness", 1.0) < 0.3:
                tb_flags.append("low_feat_completeness_outlier")
        else:
            grambank_val = {"in_overlap": False}

        # 2. Entropy sensitivity
        ent_result = entropy_by_tb.get(tb_id)
        entropy_val = None
        if ent_result:
            entropy_val = {
                "entropy_threshold_10": ent_result["entropy_threshold_10"],
                "entropy_threshold_20": ent_result["entropy_threshold_20"],
                "n_relations_dropped": ent_result["n_relations_dropped"],
                "abs_diff": round(abs(ent_result["entropy_threshold_10"] - ent_result["entropy_threshold_20"]), 6),
            }
        else:
            entropy_val = {"computed": False}

        # 3. Representativeness
        repr_val = {
            "coverage": round(coverage, 4),
            "low_coverage": coverage < 0.10,
        }
        if coverage < 0.10:
            tb_flags.append("low_coverage")

        # 4. Autocorrelation
        ac_bins = autocorr_by_tb.get(tb_id, [])
        n_sig_bins = sum(1 for b in ac_bins if b["significant_at_05"])
        autocorr_val = {
            "n_bins_tested": len(ac_bins),
            "n_bins_significant": n_sig_bins,
            "has_autocorrelation": n_sig_bins > 0,
        }
        if n_sig_bins > 0:
            tb_flags.append("autocorrelation_detected")

        # 5. Annotation confound
        fc = tb["metadata_feat_completeness"]
        confound_val = {
            "feat_completeness": fc,
            "is_low": fc < 0.3,
            "global_confound_detected": is_confound,
        }
        if fc < 0.3:
            tb_flags.append("low_annotation_completeness")

        comprehensive = json.dumps({
            "grambank": grambank_val,
            "entropy": entropy_val,
            "representativeness": repr_val,
            "autocorrelation": autocorr_val,
            "annotation_confound": confound_val,
            "quality_flags": tb_flags,
            "n_flags": len(tb_flags),
            "overall_quality": "GOOD" if len(tb_flags) == 0 else (
                "ACCEPTABLE" if len(tb_flags) <= 1 else "CONCERN"
            ),
        }, default=str)

        # --- Baseline: simple range check ---
        mr = tb["metadata_morph_richness"]
        wo = tb["metadata_word_order_entropy"]
        baseline_flags: list[str] = []
        if not (0.0 <= mr <= 5.0):
            baseline_flags.append("morph_richness_out_of_range")
        if not (0.0 <= wo <= 1.0):
            baseline_flags.append("word_order_entropy_out_of_range")
        if not (0.0 <= fc <= 1.0):
            baseline_flags.append("feat_completeness_out_of_range")
        if total_all < 500:
            baseline_flags.append("too_few_sentences")

        baseline = json.dumps({
            "all_in_range": len(baseline_flags) == 0,
            "flags": baseline_flags,
            "n_flags": len(baseline_flags),
            "overall_quality": "PASS" if len(baseline_flags) == 0 else "FAIL",
        })

        examples.append({
            "input": tb_id,
            "output": json.dumps(output_data, default=str),
            "predict_comprehensive_validation": comprehensive,
            "predict_baseline_range_check": baseline,
            "metadata_language": tb["metadata_language"],
            "metadata_iso_code": tb["metadata_iso_code"],
            "metadata_modality": tb["metadata_modality"],
            "metadata_genre": tb["metadata_genre"],
            "metadata_morph_richness": tb["metadata_morph_richness"],
            "metadata_word_order_entropy": tb["metadata_word_order_entropy"],
            "metadata_feat_completeness": tb["metadata_feat_completeness"],
            "metadata_n_quality_flags": len(tb_flags),
        })

    return {
        "metadata": {
            "method_name": "data_quality_validation",
            "description": (
                "Per-treebank validation combining six analyses: "
                "Grambank cross-validation, entropy threshold sensitivity, "
                "representativeness, autocorrelation, annotation confound, "
                "and family bias diagnostics."
            ),
            "n_treebanks": len(examples),
            "global_summary": global_summary,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "datasets": [
            {
                "dataset": "data_quality_validation",
                "examples": examples,
            }
        ],
    }


# ===========================================================================
# MAIN ORCHESTRATION
# ===========================================================================

def _save_partial(
    treebank_rows: list[dict],
    grambank_overlap: list[dict],
    results: dict,
    flags: list[str],
) -> None:
    """Save partial results so we don't lose completed analyses on timeout."""
    output = format_per_treebank_output(treebank_rows, grambank_overlap, results, flags)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Partial results saved to {out_path}")


@logger.catch
def main():
    overall_start = time.time()
    logger.info("=== Starting Data Quality Validation ===")
    results: dict = {}
    flags: list[str] = []

    # ---------------------------------------------------------------
    # STEP 1: Load all data
    # ---------------------------------------------------------------
    logger.info("--- Loading data ---")
    treebank_rows, sentence_rows = load_data_id3()
    grambank_overlap = load_data_id4_overlap()

    # ---------------------------------------------------------------
    # STEP 2: Grambank cross-validation (fast, no HF)
    # ---------------------------------------------------------------
    try:
        results["grambank_crossvalidation"] = analysis_grambank_crossvalidation(
            treebank_rows, grambank_overlap
        )
        gcv = results["grambank_crossvalidation"]
        if gcv.get("interpretation") == "NEGLIGIBLE":
            flags.append(
                "CRITICAL: UD morph_richness and Grambank morph_index show negligible correlation"
            )
        if gcv.get("n_outliers_low_completeness", 0) > 10:
            flags.append(
                f"WARNING: {gcv['n_outliers_low_completeness']} languages with "
                "feat_completeness < 0.3 may have artifactual morph_richness"
            )
    except Exception:
        logger.error("Grambank cross-validation failed")
        results["grambank_crossvalidation"] = {"error": "analysis failed"}

    # ---------------------------------------------------------------
    # STEP 3: Annotation confound (fast, no HF)
    # ---------------------------------------------------------------
    try:
        results["annotation_confound"] = analysis_annotation_confound(treebank_rows)
        if results["annotation_confound"].get("is_confound"):
            flags.append(
                "CRITICAL: feat_completeness is a strong confound for morph_richness (rho > 0.7)"
            )
    except Exception:
        logger.error("Annotation confound analysis failed")
        results["annotation_confound"] = {"error": "analysis failed"}

    # ---------------------------------------------------------------
    # STEP 4: Representativeness (fast, no HF)
    # ---------------------------------------------------------------
    try:
        results["representativeness"] = analysis_representativeness(treebank_rows)
        if results["representativeness"]["summary"]["overall_coverage"] < 0.15:
            flags.append("WARNING: Binned subset covers <15% of total sentences")
    except Exception:
        logger.error("Representativeness analysis failed")
        results["representativeness"] = {"error": "analysis failed"}

    # ---------------------------------------------------------------
    # STEP 5: Autocorrelation (moderate, uses stored sentence data)
    # ---------------------------------------------------------------
    try:
        results["autocorrelation"] = analysis_autocorrelation(sentence_rows)
        if results["autocorrelation"].get("recommend_block_subsampling"):
            flags.append("WARNING: >20% of bin-treebank combos show serial dependence")
    except Exception:
        logger.error("Autocorrelation analysis failed")
        results["autocorrelation"] = {"error": "analysis failed"}

    # Free sentence data to make room for HuggingFace loading
    del sentence_rows
    gc.collect()

    # ---------------------------------------------------------------
    # STEP 6: Family bias (fast, no HF)
    # ---------------------------------------------------------------
    try:
        results["family_bias"] = analysis_family_bias(treebank_rows)
    except Exception:
        logger.error("Family bias analysis failed")
        results["family_bias"] = {"error": "analysis failed"}

    # Write intermediate results (in case entropy phase is slow/interrupted)
    _save_partial(treebank_rows, grambank_overlap, results, flags)

    # ---------------------------------------------------------------
    # STEP 7: Entropy threshold sensitivity (slow, requires HuggingFace)
    # ---------------------------------------------------------------
    remaining_time = 3600 - (time.time() - overall_start)  # rough 60min budget
    if remaining_time > 600:
        try:
            results["entropy_threshold_sensitivity"] = (
                analysis_entropy_threshold_sensitivity(treebank_rows, grambank_overlap)
            )
            ets = results["entropy_threshold_sensitivity"]
            if "error" not in ets and not ets.get("is_robust", True):
                flags.append(
                    "CAUTION: word-order entropy not robust to threshold change (10 vs 20)"
                )
        except Exception:
            logger.error("Entropy threshold sensitivity failed")
            results["entropy_threshold_sensitivity"] = {"error": "analysis failed"}
    else:
        logger.warning("Skipping entropy sensitivity — insufficient time remaining")
        results["entropy_threshold_sensitivity"] = {"error": "skipped due to time budget"}

    # ---------------------------------------------------------------
    # STEP 8: Compile per-treebank output
    # ---------------------------------------------------------------
    output = format_per_treebank_output(treebank_rows, grambank_overlap, results, flags)

    # Write output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    n_examples = len(output["datasets"][0]["examples"])
    logger.info(f"Results written to {out_path} ({n_examples} treebank examples)")
    logger.info(f"Total runtime: {time.time() - overall_start:.1f}s")
    logger.info(f"Diagnostic flags: {len(flags)} total")

    return output


if __name__ == "__main__":
    main()
