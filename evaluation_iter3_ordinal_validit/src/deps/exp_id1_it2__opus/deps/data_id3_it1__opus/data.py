# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "numpy",
#   "tqdm",
#   "loguru",
# ]
# ///
"""Extract sentence-level max-DD and treebank-level typological features from Universal Dependencies.

Source: commul/universal_dependencies on HuggingFace (UD v2.17, Parquet format).
Output: full_data_out.json in exp_sel_data_out.json schema.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
OUTPUT_FILE = WORKSPACE / "full_data_out.json"
LOG_FILE = WORKSPACE / "logs" / "data_extraction.log"

HF_DATASET = "commul/universal_dependencies"
TARGET_BINS = frozenset({10, 12, 14, 16, 18, 20})
MIN_TOTAL_SENTENCES = 500
MIN_PER_BIN = 50

# Controls: set via env or defaults
MAX_CONFIGS: int | None = None  # None = all; set small for testing

# ---------------------------------------------------------------------------
# ISO-639 lookup (prefix → language name)
# ---------------------------------------------------------------------------
ISO_MAP: dict[str, str] = {
    "abq": "Abaza", "ab": "Abkhaz", "af": "Afrikaans", "akk": "Akkadian",
    "aqz": "Akuntsu", "sq": "Albanian", "aln": "Albanian-Gheg", "gsw": "Swiss-German",
    "am": "Amharic", "grc": "Ancient-Greek", "hbo": "Ancient-Hebrew",
    "apu": "Apurinã", "ar": "Arabic", "hy": "Armenian", "aii": "Assyrian",
    "az": "Azerbaijani", "bm": "Bambara", "eu": "Basque", "bar": "Bavarian",
    "bej": "Beja", "be": "Belarusian", "bn": "Bengali", "bho": "Bhojpuri",
    "sab": "Burushaski", "bor": "Bororo", "br": "Breton", "bg": "Bulgarian",
    "bxr": "Buryat", "yue": "Cantonese", "cpg": "Cappadocian",
    "ca": "Catalan", "ceb": "Cebuano", "ckb": "Central-Kurdish",
    "zh": "Chinese", "ctn": "Choctaw", "ckt": "Chukchi",
    "xcl": "Classical-Armenian", "lzh": "Classical-Chinese",
    "cop": "Coptic", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "egy": "Egyptian", "en": "English", "myv": "Erzya",
    "eo": "Esperanto", "et": "Estonian", "fo": "Faroese", "fi": "Finnish",
    "fr": "French", "qfn": "Frisian-Northern", "gl": "Galician",
    "ka": "Georgian", "de": "German", "got": "Gothic", "el": "Greek",
    "gub": "Guajajara", "gn": "Guarani", "gu": "Gujarati", "gwi": "Gwich'in",
    "ht": "Haitian-Creole", "ha": "Hausa", "he": "Hebrew", "azz": "Highland-Puebla-Nahuatl",
    "hi": "Hindi", "hit": "Hittite", "hu": "Hungarian",
    "is": "Icelandic", "arh": "Ika", "id": "Indonesian", "ga": "Irish",
    "it": "Italian", "ja": "Japanese", "jv": "Javanese", "urb": "Kaapor",
    "xnr": "Kangri", "krl": "Karelian", "arr": "Karo", "kk": "Kazakh",
    "naq": "Khoekhoe", "kfm": "Khufi", "quc": "K'iche'",
    "koi": "Komi-Permyak", "kpv": "Komi-Zyrian", "ko": "Korean",
    "ky": "Kyrgyz", "ltg": "Latgalian", "la": "Latin", "lv": "Latvian",
    "lij": "Ligurian", "lt": "Lithuanian", "olo": "Livvi",
    "nds": "Low-Saxon", "lb": "Luxembourgish", "mk": "Macedonian",
    "jaa": "Jarawara", "qaf": "Arabizi", "mpu": "Makuráp",
    "ml": "Malayalam", "mt": "Maltese", "gv": "Manx", "mr": "Marathi",
    "gun": "Mbyá-Guaraní", "frm": "Middle-French", "mdf": "Moksha",
    "myu": "Mundurukú", "nmf": "Namonuito", "pcm": "Naija",
    "nyq": "Nayini", "nap": "Neapolitan", "yrk": "Nenets",
    "yrl": "Nheengatu", "sme": "North-Sámi", "kmr": "Kurmanji",
    "gya": "Northwest-Gbaya", "no": "Norwegian", "oc": "Occitan",
    "or": "Odia", "cu": "Old-Church-Slavonic", "orv": "Old-Russian",
    "ang": "Old-English", "fro": "Old-French", "sga": "Old-Irish",
    "pro": "Old-Provençal", "otk": "Old-Turkish", "ota": "Ottoman-Turkish",
    "ps": "Pashto", "pad": "Paumarí", "fa": "Persian", "pay": "Pech",
    "xpg": "Phrygian", "pl": "Polish", "qpm": "Pomak", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "gd": "Scottish-Gaelic",
    "sr": "Serbian", "wuu": "Wu-Chinese", "scn": "Sicilian",
    "sd": "Sindhi", "si": "Sinhala", "sms": "Skolt-Sámi",
    "sk": "Slovak", "sl": "Slovenian", "soj": "Soi",
    "ajp": "South-Levantine-Arabic", "sdh": "Southern-Kurdish",
    "es": "Spanish", "ssp": "Spanish-Sign-Language", "sv": "Swedish",
    "swl": "Swedish-Sign-Language", "tl": "Tagalog", "ta": "Tamil",
    "tt": "Tatar", "eme": "Teko", "te": "Telugu", "qte": "Teko-Tupi",
    "th": "Thai", "tn": "Tswana", "tpn": "Tupinambá", "tr": "Turkish",
    "qti": "Turkish-Informal", "qtd": "Turkish-German",
    "uk": "Ukrainian", "xum": "Umbrian", "hsb": "Upper-Sorbian",
    "ur": "Urdu", "ug": "Uyghur", "uz": "Uzbek", "vep": "Veps",
    "vi": "Vietnamese", "wbp": "Warlpiri", "cy": "Welsh",
    "hyw": "Western-Armenian", "nhi": "Western-Sierra-Puebla-Nahuatl",
    "wo": "Wolof", "xav": "Xavante", "sjo": "Xibe",
    "sah": "Yakut", "yi": "Yiddish", "yo": "Yoruba", "ess": "Yupik",
    "say": "Sayula-Popoluca",
}

# Spoken treebank identifiers (from Dobrovoljc 2022 + plan list)
SPOKEN_TREEBANKS = frozenset({
    "fr_spoken", "fr_rhapsodie", "fr_parisstories",
    "sl_sst", "no_nynorsklia",
    "it_postwita", "it_twittiro", "it_kiparlaforest",
    "tr_gb", "es_coser",
    "pcm_nsc", "qtd_sagt", "swl_sslc",
    "gun_dooley", "gun_thomas",
    "en_childes", "en_eslspok",
    "hy_bsut",
})

# Genre hints from treebank name suffixes
GENRE_MAP: dict[str, str] = {
    "ewt": "web", "gsd": "wiki", "gsdsimp": "wiki",
    "pdt": "news", "pdtc": "news", "cac": "academic",
    "pud": "news", "bosque": "news", "set": "news",
    "ancora": "news", "htb": "news", "hdtb": "mixed",
    "sst": "spoken", "spoken": "spoken", "rhapsodie": "spoken",
    "parisstories": "spoken", "postwita": "social-media",
    "twittiro": "social-media", "gumreddit": "social-media",
    "taiga": "social-media", "atis": "spoken",
    "isdt": "mixed", "vit": "legal", "partut": "legal",
    "sequoia": "medical", "ftb": "news",
}


# ---------------------------------------------------------------------------
# Sentence-level processing
# ---------------------------------------------------------------------------
def process_sentence(
    tokens: list[str],
    heads_str: list[str],
    deprels: list[str],
    feats_list: list[str | None],
) -> dict[str, Any] | None:
    """Extract features from a single sentence. Returns None if invalid."""
    n = len(tokens)
    if n < 2 or len(heads_str) != n:
        return None

    # Convert heads to int, validate
    heads: list[int] = []
    for h in heads_str:
        if h is None or "-" in str(h):
            return None  # MWT contamination
        try:
            heads.append(int(h))
        except (ValueError, TypeError):
            return None

    # Validate head range
    for h in heads:
        if h < 0 or h > n:
            return None

    # Dependency distances & tree edges
    dep_distances: list[int] = []
    tree_edges: list[tuple[int, int]] = []
    head_directions: list[tuple[str, bool]] = []

    for i, h in enumerate(heads):
        dep_pos = i + 1
        tree_edges.append((h, dep_pos))
        if h == 0:
            continue  # root — no dependency distance
        dd = abs(dep_pos - h)
        dep_distances.append(dd)
        head_directions.append((deprels[i], h < dep_pos))

    if not dep_distances:
        return None

    max_dd = max(dep_distances)
    max_dd_norm = round(max_dd / (n - 1), 6)

    # Morphological feature counts per token
    feat_counts: list[int] = []
    for f in feats_list:
        if f is None or f == "_" or f == "":
            feat_counts.append(0)
        else:
            feat_counts.append(len(f.split("|")))

    # Non-projectivity check (inline for speed)
    is_nonproj = False
    arcs = [(min(h, d), max(h, d)) for h, d in tree_edges if h != 0]
    n_arcs = len(arcs)
    for i in range(n_arcs):
        a1_l, a1_r = arcs[i]
        for j in range(i + 1, n_arcs):
            a2_l, a2_r = arcs[j]
            if (a1_l < a2_l < a1_r < a2_r) or (a2_l < a1_l < a2_r < a1_r):
                is_nonproj = True
                break
        if is_nonproj:
            break

    return {
        "n": n,
        "max_dd": max_dd,
        "max_dd_normalized": max_dd_norm,
        "dep_distances": dep_distances,
        "tree_edges": tree_edges,
        "feat_counts": feat_counts,
        "head_directions": head_directions,
        "is_nonprojective": is_nonproj,
    }


# ---------------------------------------------------------------------------
# Treebank-level feature computation
# ---------------------------------------------------------------------------
def compute_treebank_features(
    all_feat_counts: list[int],
    all_head_directions: list[tuple[str, bool]],
    n_nonproj_sentences: int,
    n_total_sentences: int,
    tokens_with_feats: int,
    total_tokens: int,
) -> dict[str, Any]:
    """Compute treebank-level typological features from aggregated sentence data."""

    # 5a. Morphological richness
    morph_richness = round(sum(all_feat_counts) / max(total_tokens, 1), 4)

    # 5b. Feature annotation completeness
    feat_completeness = round(tokens_with_feats / max(total_tokens, 1), 4)

    # 5c. Head-direction ratio
    if all_head_directions:
        hb_count = sum(1 for _, is_hbd in all_head_directions if is_hbd)
        head_direction_ratio = round(hb_count / len(all_head_directions), 4)
    else:
        head_direction_ratio = 0.5

    # 5d. Word-order entropy (Levshina 2019)
    rel_counts: dict[str, Counter] = defaultdict(Counter)
    for deprel, is_hbd in all_head_directions:
        rel_counts[deprel][is_hbd] += 1

    rel_entropies: dict[str, float] = {}
    rel_token_counts: dict[str, int] = {}
    for deprel, counts in rel_counts.items():
        # Exclude punct and root relations
        if deprel in ("punct", "root"):
            continue
        total = counts[True] + counts[False]
        # Exclude relations with < 10 tokens
        if total < 10:
            continue
        rel_token_counts[deprel] = total
        if counts[True] == 0 or counts[False] == 0:
            rel_entropies[deprel] = 0.0
        else:
            p = counts[True] / total
            rel_entropies[deprel] = round(-p * math.log2(p) - (1 - p) * math.log2(1 - p), 6)

    total_rel_tokens = sum(rel_token_counts.values())
    if total_rel_tokens > 0:
        word_order_entropy = round(
            sum(rel_entropies[r] * rel_token_counts[r] / total_rel_tokens for r in rel_entropies),
            4,
        )
    else:
        word_order_entropy = 0.0

    # 5e. Non-projectivity rate
    nonprojectivity_rate = round(n_nonproj_sentences / max(n_total_sentences, 1), 4)

    # Per-relation entropy (top relations for output)
    top_rels = dict(sorted(rel_entropies.items(), key=lambda x: -rel_token_counts.get(x[0], 0))[:20])

    # Mean DD across all sentences (for metadata)
    return {
        "morph_richness": morph_richness,
        "feat_completeness": feat_completeness,
        "head_direction_ratio": head_direction_ratio,
        "word_order_entropy": word_order_entropy,
        "nonprojectivity_rate": nonprojectivity_rate,
        "per_relation_entropy": top_rels,
    }


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def get_iso_code(config_name: str) -> str:
    """Extract ISO code from config name like 'en_ewt' → 'en'."""
    return config_name.split("_")[0]


def get_language(config_name: str) -> str:
    """Map config name to language name."""
    iso = get_iso_code(config_name)
    return ISO_MAP.get(iso, iso.upper())


def get_genre(config_name: str) -> str:
    """Infer genre from treebank name suffix."""
    parts = config_name.split("_", 1)
    if len(parts) < 2:
        return "mixed"
    suffix = parts[1]
    return GENRE_MAP.get(suffix, "mixed")


def get_modality(config_name: str) -> str:
    """Binary spoken/written classification."""
    return "spoken" if config_name in SPOKEN_TREEBANKS else "written"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_treebank(config_name: str) -> tuple[list[dict], dict | None, dict]:
    """Process a single treebank config. Returns (sentence_rows, treebank_row, stats)."""
    stats: dict[str, Any] = {"config": config_name, "status": "ok", "reason": ""}
    iso = get_iso_code(config_name)
    lang = get_language(config_name)

    try:
        ds = load_dataset(HF_DATASET, config_name)
    except Exception as exc:
        stats["status"] = "load_error"
        stats["reason"] = str(exc)[:200]
        return [], None, stats

    # Concatenate all splits
    all_splits = [ds[s] for s in ds.keys()]
    full_tb = concatenate_datasets(all_splits)
    n_total = len(full_tb)

    if n_total < MIN_TOTAL_SENTENCES:
        stats["status"] = "disqualified"
        stats["reason"] = f"too few sentences: {n_total} < {MIN_TOTAL_SENTENCES}"
        del full_tb, all_splits, ds
        gc.collect()
        return [], None, stats

    # Aggregators for treebank-level features
    all_feat_counts: list[int] = []
    all_head_directions: list[tuple[str, bool]] = []
    n_nonproj = 0
    tokens_with_feats = 0
    total_tokens = 0
    all_dd_sums: list[float] = []  # for mean DD

    # Sentence-level bins
    bin_sentences: dict[int, list[dict]] = {b: [] for b in TARGET_BINS}
    n_per_bin: dict[str, int] = {str(b): 0 for b in sorted(TARGET_BINS)}

    for idx in range(n_total):
        row = full_tb[idx]
        tokens = row["tokens"]
        heads_str = row["head"]
        deprels = row["deprel"]
        feats_list = row["feats"]

        result = process_sentence(tokens, heads_str, deprels, feats_list)
        if result is None:
            continue

        n = result["n"]

        # Aggregate treebank-level data
        all_feat_counts.extend(result["feat_counts"])
        all_head_directions.extend(result["head_directions"])
        total_tokens += n
        tokens_with_feats += sum(1 for fc in result["feat_counts"] if fc > 0)
        if result["is_nonprojective"]:
            n_nonproj += 1
        if result["dep_distances"]:
            all_dd_sums.append(sum(result["dep_distances"]) / len(result["dep_distances"]))

        # Bin check
        if n in TARGET_BINS:
            n_per_bin[str(n)] += 1
            bin_sentences[n].append({
                "idx": idx,
                "max_dd": result["max_dd"],
                "max_dd_normalized": result["max_dd_normalized"],
                "dep_distances": ",".join(str(d) for d in result["dep_distances"]),
                "tree_edges": ",".join(f"{h}-{d}" for h, d in result["tree_edges"]),
                "is_nonprojective": result["is_nonprojective"],
                "sentence_length": n,
            })

    # Free dataset memory
    del full_tb, all_splits, ds
    gc.collect()

    # Compute treebank-level features
    tb_features = compute_treebank_features(
        all_feat_counts, all_head_directions, n_nonproj, n_total,
        tokens_with_feats, total_tokens,
    )

    # Qualification per bin
    qualifies_per_bin = {str(b): (n_per_bin[str(b)] >= MIN_PER_BIN) for b in sorted(TARGET_BINS)}
    mean_dd_all = round(float(np.mean(all_dd_sums)), 4) if all_dd_sums else 0.0

    genre = get_genre(config_name)
    modality = get_modality(config_name)

    # Build treebank summary row
    per_rel_str = json.dumps(tb_features["per_relation_entropy"])
    treebank_row = {
        "input": config_name,
        "output": "",
        "metadata_row_type": "treebank",
        "metadata_treebank_id": config_name,
        "metadata_language": lang,
        "metadata_iso_code": iso,
        "metadata_morph_richness": tb_features["morph_richness"],
        "metadata_head_direction_ratio": tb_features["head_direction_ratio"],
        "metadata_word_order_entropy": tb_features["word_order_entropy"],
        "metadata_nonprojectivity_rate": tb_features["nonprojectivity_rate"],
        "metadata_feat_completeness": tb_features["feat_completeness"],
        "metadata_genre": genre,
        "metadata_modality": modality,
        "metadata_n_sentences_total": n_total,
        "metadata_n_per_bin": json.dumps(n_per_bin),
        "metadata_qualifies_per_bin": json.dumps(qualifies_per_bin),
        "metadata_mean_dd_all": mean_dd_all,
        "metadata_per_relation_entropy": per_rel_str,
        "metadata_fold": "treebank_summary",
    }

    # Build sentence-level rows (only qualifying bins)
    sentence_rows: list[dict] = []
    for b in sorted(TARGET_BINS):
        if not qualifies_per_bin[str(b)]:
            continue
        for s in bin_sentences[b]:
            sentence_rows.append({
                "input": f"{config_name}__{s['idx']}",
                "output": "",
                "metadata_row_type": "sentence",
                "metadata_treebank_id": config_name,
                "metadata_language": lang,
                "metadata_iso_code": iso,
                "metadata_sentence_length": s["sentence_length"],
                "metadata_length_bin": s["sentence_length"],
                "metadata_max_dd": s["max_dd"],
                "metadata_max_dd_normalized": s["max_dd_normalized"],
                "metadata_dep_distances": s["dep_distances"],
                "metadata_tree_edges": s["tree_edges"],
                "metadata_is_nonprojective": s["is_nonprojective"],
                "metadata_fold": f"bin_{s['sentence_length']}",
            })

    stats["n_total"] = n_total
    stats["n_sentence_rows"] = len(sentence_rows)
    stats["n_per_bin"] = n_per_bin
    stats["qualifies_per_bin"] = qualifies_per_bin

    return sentence_rows, treebank_row, stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_output(examples: list[dict]) -> None:
    """Run sanity checks on the output."""
    sentence_rows = [e for e in examples if e.get("metadata_row_type") == "sentence"]
    treebank_rows = [e for e in examples if e.get("metadata_row_type") == "treebank"]

    logger.info(f"Validation: {len(sentence_rows)} sentence rows, {len(treebank_rows)} treebank rows")

    # Sentence-level checks
    bad_dd = 0
    bad_norm = 0
    for s in sentence_rows:
        n = s["metadata_sentence_length"]
        dd = s["metadata_max_dd"]
        norm = s["metadata_max_dd_normalized"]
        if not (1 <= dd <= n - 1):
            bad_dd += 1
        if not (0 < norm <= 1.0):
            bad_norm += 1
    if bad_dd:
        logger.warning(f"  {bad_dd} sentences with max_dd out of [1, n-1]")
    if bad_norm:
        logger.warning(f"  {bad_norm} sentences with normalized max_dd out of (0, 1]")

    # Treebank-level checks
    for t in treebank_rows:
        hdr = t["metadata_head_direction_ratio"]
        woe = t["metadata_word_order_entropy"]
        mr = t["metadata_morph_richness"]
        if not (0 <= hdr <= 1):
            logger.warning(f"  {t['metadata_treebank_id']}: head_direction_ratio={hdr} out of [0,1]")
        if not (0 <= woe <= 1):
            logger.warning(f"  {t['metadata_treebank_id']}: word_order_entropy={woe} out of [0,1]")
        if mr < 0:
            logger.warning(f"  {t['metadata_treebank_id']}: morph_richness={mr} < 0")

    # Bin distribution
    bin_counts: Counter = Counter()
    for s in sentence_rows:
        bin_counts[s["metadata_length_bin"]] += 1
    for b in sorted(bin_counts):
        logger.info(f"  Bin {b}: {bin_counts[b]} sentences")

    logger.info("Validation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(LOG_FILE), rotation="50 MB", level="DEBUG")
    logger.info("=" * 70)
    logger.info("Starting UD Max-DD & Typological Features Extraction")
    logger.info("=" * 70)

    t0 = time.time()

    # Step 1: Enumerate configs
    logger.info(f"Loading configs from {HF_DATASET}...")
    configs = get_dataset_config_names(HF_DATASET)
    logger.info(f"Found {len(configs)} treebank configurations")

    if MAX_CONFIGS:
        configs = configs[:MAX_CONFIGS]
        logger.info(f"  (Limited to first {MAX_CONFIGS} for testing)")

    # Step 2-5: Process each treebank sequentially
    all_examples: list[dict] = []
    all_stats: list[dict] = []
    qualifying_count = 0

    for i, config_name in enumerate(tqdm(configs, desc="Processing treebanks")):
        logger.info(f"[{i+1}/{len(configs)}] Processing {config_name}...")
        t1 = time.time()

        sentence_rows, treebank_row, stats = process_treebank(config_name)

        elapsed = time.time() - t1
        logger.info(
            f"  → {stats['status']} | "
            f"{stats.get('n_total', 0)} total sents | "
            f"{stats.get('n_sentence_rows', 0)} binned rows | "
            f"{elapsed:.1f}s"
        )
        if stats["status"] == "disqualified":
            logger.info(f"  → Reason: {stats['reason']}")

        all_stats.append(stats)

        if treebank_row is not None:
            qualifying_count += 1
            all_examples.append(treebank_row)
            all_examples.extend(sentence_rows)

        # Log progress every 20 treebanks
        if (i + 1) % 20 == 0:
            elapsed_total = time.time() - t0
            rate = (i + 1) / elapsed_total
            remaining = (len(configs) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  Progress: {i+1}/{len(configs)} | "
                f"{qualifying_count} qualifying | "
                f"{len(all_examples)} total examples | "
                f"ETA: {remaining/60:.1f}min"
            )

    elapsed_total = time.time() - t0
    logger.info(f"\nProcessing complete in {elapsed_total/60:.1f} minutes")
    logger.info(f"Qualifying treebanks: {qualifying_count}")
    logger.info(f"Total examples: {len(all_examples)}")

    # Step 6: Validation
    validate_output(all_examples)

    # Step 7: Write output in exp_sel_data_out.json schema
    output = {
        "datasets": [
            {
                "dataset": "ud_maxdd_typology",
                "examples": all_examples,
            }
        ]
    }

    # Write initial full output
    logger.info(f"Writing output to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")

    # --- File size management ---
    MAX_FILE_MB = 100.0  # aii_file_size_limit threshold

    if file_size_mb > MAX_FILE_MB:
        logger.info(f"File exceeds {MAX_FILE_MB}MB — applying size reduction...")

        # Step A: strip dep_distances + tree_edges from sentence rows → supplementary file
        treebank_rows = [e for e in all_examples if e.get("metadata_row_type") == "treebank"]
        sentence_rows = [e for e in all_examples if e.get("metadata_row_type") == "sentence"]

        supplementary = []
        slim_sentences = []
        for s in sentence_rows:
            supplementary.append({
                "input": s["input"],
                "metadata_dep_distances": s.get("metadata_dep_distances", ""),
                "metadata_tree_edges": s.get("metadata_tree_edges", ""),
            })
            slim = {k: v for k, v in s.items()
                    if k not in ("metadata_dep_distances", "metadata_tree_edges")}
            slim_sentences.append(slim)

        # Write supplementary
        sup_path = WORKSPACE / "supplementary_distances.json"
        with open(sup_path, "w", encoding="utf-8") as f:
            json.dump(supplementary, f, ensure_ascii=False)
        logger.info(f"  Supplementary file: {sup_path.stat().st_size / (1024*1024):.1f} MB")

        slim_examples = treebank_rows + slim_sentences

        # Rewrite slim version
        slim_output = {"datasets": [{"dataset": "ud_maxdd_typology", "examples": slim_examples}]}
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(slim_output, f, ensure_ascii=False)
        slim_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"  Slim full_data_out.json: {slim_mb:.1f} MB")

        # Step B: split into parts if still > MAX_FILE_MB
        if slim_mb > MAX_FILE_MB:
            data_out_dir = WORKSPACE / "data_out"
            data_out_dir.mkdir(exist_ok=True)
            n_parts = int(slim_mb // MAX_FILE_MB) + 1
            chunk_size = len(slim_sentences) // n_parts

            for i in range(n_parts):
                start = i * chunk_size
                end = start + chunk_size if i < n_parts - 1 else len(slim_sentences)
                part_examples = (treebank_rows + slim_sentences[start:end]) if i == 0 else slim_sentences[start:end]
                part_data = {"datasets": [{"dataset": "ud_maxdd_typology", "examples": part_examples}]}
                part_path = data_out_dir / f"full_data_out_{i+1}.json"
                with open(part_path, "w", encoding="utf-8") as f:
                    json.dump(part_data, f, ensure_ascii=False)
                logger.info(f"  Part {i+1}: {len(part_examples)} examples, {part_path.stat().st_size/(1024*1024):.1f} MB")

            OUTPUT_FILE.unlink()
            logger.info(f"  Split into {n_parts} parts in data_out/")

    # --- Generate mini & preview ---
    logger.info("Generating mini and preview versions...")
    first_5_tbs = [t["metadata_treebank_id"] for t in
                   [e for e in all_examples if e.get("metadata_row_type") == "treebank"][:5]]
    mini_examples = [e for e in all_examples if e.get("metadata_treebank_id") in first_5_tbs]
    # Strip dep_distances/tree_edges from mini too for consistency
    mini_clean = []
    for e in mini_examples:
        clean = {k: v for k, v in e.items()
                 if k not in ("metadata_dep_distances", "metadata_tree_edges")}
        mini_clean.append(clean)
    mini_data = {"datasets": [{"dataset": "ud_maxdd_typology", "examples": mini_clean}]}
    mini_path = WORKSPACE / "mini_data_out.json"
    with open(mini_path, "w", encoding="utf-8") as f:
        json.dump(mini_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  mini_data_out.json: {len(mini_clean)} examples, {mini_path.stat().st_size/(1024*1024):.1f} MB")

    def truncate_strings(obj: Any, max_len: int = 200) -> Any:
        if isinstance(obj, str):
            return obj[:max_len] + "..." if len(obj) > max_len else obj
        if isinstance(obj, dict):
            return {k: truncate_strings(v, max_len) for k, v in obj.items()}
        if isinstance(obj, list):
            return [truncate_strings(i, max_len) for i in obj]
        return obj

    preview_examples = [truncate_strings(e) for e in all_examples[:50]]
    # Also strip dep_distances/tree_edges
    preview_clean = []
    for e in preview_examples:
        clean = {k: v for k, v in e.items()
                 if k not in ("metadata_dep_distances", "metadata_tree_edges")}
        preview_clean.append(clean)
    preview_data = {"datasets": [{"dataset": "ud_maxdd_typology", "examples": preview_clean}]}
    preview_path = WORKSPACE / "preview_data_out.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  preview_data_out.json: {len(preview_clean)} examples, {preview_path.stat().st_size/(1024*1024):.1f} MB")

    # Summary stats
    disqualified = [s for s in all_stats if s["status"] == "disqualified"]
    load_errors = [s for s in all_stats if s["status"] == "load_error"]
    logger.info(f"Disqualified: {len(disqualified)} treebanks")
    logger.info(f"Load errors: {len(load_errors)} treebanks")
    for s in load_errors:
        logger.warning(f"  Load error: {s['config']} — {s['reason']}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
