#!/usr/bin/env python3
"""Download Grambank v1.0.3 morphosyntactic features, compute per-language
morphological complexity index, and map to Universal Dependencies treebanks.

Produces full_data_out.json following the exp_sel_data_out schema.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import csv
import ssl
import math
import resource
import urllib.request

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Memory limits (cgroup-aware)
# ---------------------------------------------------------------------------
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(min(4, TOTAL_RAM_GB * 0.5) * 1024**3)  # 4 GB max for this task
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

WORKSPACE = Path(__file__).parent.resolve()
TEMP_DIR = WORKSPACE / "temp" / "datasets"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
(WORKSPACE / "logs").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SSL context for GitHub raw downloads
# ---------------------------------------------------------------------------
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE
_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
urllib.request.install_opener(_opener)


def download_file(url: str, dest: Path) -> Path:
    """Download a file if not already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  Cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    logger.info(f"  Downloading {url}")
    urllib.request.urlretrieve(url, str(dest))
    logger.info(f"  Saved: {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def read_csv(path: Path) -> list[dict]:
    """Read CSV file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ===========================================================================
# STEP 1: Download Grambank CLDF v1.0.3
# ===========================================================================
@logger.catch
def step1_download_grambank() -> dict[str, Path]:
    logger.info("STEP 1: Downloading Grambank v1.0.3 CLDF data...")
    BASE = "https://raw.githubusercontent.com/grambank/grambank/v1.0.3/cldf/"
    files = {}
    for f in ["values.csv", "parameters.csv", "languages.csv", "codes.csv"]:
        files[f] = download_file(BASE + f, TEMP_DIR / f"grambank_{f}")
    return files


# ===========================================================================
# STEP 2: Download Feature Domain Classification
# ===========================================================================
@logger.catch
def step2_download_feature_grouping() -> Path:
    logger.info("STEP 2: Downloading feature domain classification...")
    url = "https://raw.githubusercontent.com/grambank/grambank-analysed/main/R_grambank/feature_grouping_for_analysis.csv"
    return download_file(url, TEMP_DIR / "feature_grouping_for_analysis.csv")


# ===========================================================================
# STEP 3: Download Glottolog CLDF for ISO mapping
# ===========================================================================
@logger.catch
def step3_download_glottolog() -> Path:
    logger.info("STEP 3: Downloading Glottolog CLDF languages...")
    url = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv"
    return download_file(url, TEMP_DIR / "glottolog_languages.csv")


# ===========================================================================
# STEP 4: UD Treebank Language List (hardcoded mapping)
# ===========================================================================
# Comprehensive UD 2.x language code → ISO 639-3 mapping
# Built from universaldependencies.org and UD 2.14 release
UD_CODE_TO_ISO3: dict[str, str] = {
    "ab": "abk", "abq": "abq", "af": "afr", "ajp": "ajp", "akk": "akk",
    "am": "amh", "apu": "apu", "aqz": "aqz", "ar": "ara", "be": "bel",
    "bg": "bul", "bho": "bho", "bm": "bam", "bn": "ben", "bo": "bod",
    "br": "bre", "bxr": "bxr", "ca": "cat", "ceb": "ceb", "ckt": "ckt",
    "cop": "cop", "cs": "ces", "cu": "chu", "cy": "cym", "da": "dan",
    "de": "deu", "el": "ell", "en": "eng", "es": "spa", "et": "est",
    "eu": "eus", "fa": "fas", "fi": "fin", "fo": "fao", "fr": "fra",
    "fro": "fro", "ga": "gle", "gd": "gla", "gl": "glg", "got": "got",
    "grc": "grc", "gsw": "gsw", "gub": "gub", "gv": "glv", "ha": "hau",
    "he": "heb", "hi": "hin", "hr": "hrv", "hsb": "hsb", "hu": "hun",
    "hy": "hye", "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "ka": "kat", "kaa": "kaa", "kfm": "kfm", "kk": "kaz", "kmr": "kmr",
    "ko": "kor", "koi": "koi", "kpv": "kpv", "krl": "krl", "ku": "kur",
    "la": "lat", "lij": "lij", "lt": "lit", "lv": "lav", "lzh": "lzh",
    "mdf": "mdf", "mg": "mlg", "mk": "mkd", "ml": "mal", "mn": "mon",
    "mr": "mar", "mt": "mlt", "myu": "myu", "myv": "myv", "nds": "nds",
    "ne": "nep", "nl": "nld", "nn": "nno", "no": "nor", "nqo": "nqo",
    "nyq": "nyq", "olo": "olo", "orv": "orv", "os": "oss", "pcm": "pcm",
    "pl": "pol", "pt": "por", "qaf": "qaf", "qfn": "qfn", "qhe": "heb",
    "qpm": "qpm", "qtd": "deu", "ro": "ron", "ru": "rus", "sa": "san",
    "sk": "slk", "sl": "slv", "sme": "sme", "sms": "sms", "soj": "soj",
    "sq": "sqi", "sr": "srp", "sv": "swe", "swl": "swl", "ta": "tam",
    "te": "tel", "tg": "tgk", "th": "tha", "tl": "tgl", "tr": "tur",
    "ug": "uig", "uk": "ukr", "ur": "urd", "uz": "uzb", "vi": "vie",
    "wbp": "wbp", "wo": "wol", "xnr": "xnr", "yo": "yor", "yue": "yue",
    "zh": "zho", "zu": "zul",
}

# Known UD configs (fetched from commul/universal_dependencies)
# Format: lang_code + "_" + treebank_name
UD_CONFIGS: list[str] = []


def step4_get_ud_configs() -> dict[str, list[str]]:
    """Get UD treebank configs and map language codes to ISO 639-3.

    Returns dict: iso639_3 → list of treebank config names.
    """
    logger.info("STEP 4: Getting Universal Dependencies treebank configs...")

    # Try loading from HuggingFace
    global UD_CONFIGS
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info("commul/universal_dependencies")
        if info.config_names:
            UD_CONFIGS = list(info.config_names)
    except Exception:
        pass

    if not UD_CONFIGS:
        try:
            url = "https://datasets-server.huggingface.co/splits?dataset=commul%2Funiversal_dependencies"
            req = urllib.request.Request(url, headers={"User-Agent": "Python"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                configs_set = {s["config"] for s in data.get("splits", [])}
                UD_CONFIGS = sorted(configs_set)
        except Exception as e:
            logger.warning(f"HF API fallback failed: {e}")

    if not UD_CONFIGS:
        # Hardcoded fallback: well-known UD 2.14 treebanks
        UD_CONFIGS = [
            "af_afribooms", "akk_pisandub", "akk_riao", "aqz_tudet",
            "ar_nyuad", "ar_padt", "be_hse", "bg_btb", "bho_bhtb",
            "bm_crb", "bn_bru", "bo_ricta", "br_keb", "bxr_bdt",
            "ca_ancora", "ceb_gja", "ckt_hse", "cop_scriptorium",
            "cs_cac", "cs_cltt", "cs_fictree", "cs_pdt", "cu_proiel",
            "cy_ccg", "da_ddt", "de_gsd", "de_hdt", "de_lit", "de_pud",
            "el_gdt", "en_ewt", "en_gum", "en_lines", "en_partut", "en_pud",
            "es_ancora", "es_gsd", "es_pud", "et_edt", "et_ewt",
            "eu_bdt", "fa_perdt", "fa_seraji", "fi_ftb", "fi_pud", "fi_tdt",
            "fo_farpahc", "fo_oft", "fr_ftb", "fr_gsd", "fr_partut",
            "fr_pud", "fr_sequoia", "fro_srcmf", "ga_idt", "gd_arcosg",
            "gl_ctg", "gl_treegal", "got_proiel", "grc_perseus", "grc_proiel",
            "gsw_uzh", "gub_tudet", "gv_cadhan", "ha_northernautogramm",
            "he_htb", "he_iahltwiki", "hi_hdtb", "hi_pud",
            "hr_set", "hsb_ufal", "hu_szeged", "hy_armtdp", "hy_bsut",
            "id_csui", "id_gsd", "id_pud", "is_icepahc", "is_modern",
            "it_isdt", "it_partut", "it_pud", "it_twittiro", "it_vit",
            "ja_gsd", "ja_gsdluw", "ja_modern", "ja_pud",
            "ka_glc", "kaa_jsst", "kk_ktb", "kmr_mg", "ko_gsd",
            "ko_kaist", "ko_pud", "koi_uh", "kpv_ikdp", "kpv_lattice",
            "krl_kkpp", "la_ittb", "la_llct", "la_perseus", "la_proiel",
            "la_udante", "lij_glt", "lt_alksnis", "lt_hse", "lv_lvtb",
            "lzh_kyoto", "mdf_jr", "mg_lsd", "mk_mtb", "ml_ufal",
            "mn_mhr", "mr_ufal", "mt_mudt", "myu_tudet", "myv_jr",
            "nds_lsdc", "ne_cdtb", "nl_alpino", "nl_lassysmall",
            "nn_nynorsk", "nn_nynorsklia", "no_bokmaal", "no_nynorsk",
            "no_nynorsklia", "olo_kkpp", "orv_birchbark", "orv_rnc",
            "orv_torot", "os_ossetia", "pcm_nsc", "pl_lfg", "pl_pdb",
            "pl_pud", "pt_bosque", "pt_gsd", "pt_pud",
            "ro_nonstandard", "ro_rrt", "ro_simonero", "ru_gsd",
            "ru_pud", "ru_syntagrus", "ru_taiga", "sa_ufal", "sa_vedic",
            "sk_snk", "sl_ssj", "sl_sst", "sme_giella", "sms_giellagas",
            "sq_tsa", "sr_set", "sv_lines", "sv_pud", "sv_talbanken",
            "swl_sslc", "ta_mwtt", "ta_ttb", "te_mtg", "tg_grammar",
            "th_pud", "tl_trg", "tl_ugnayan", "tr_atis", "tr_boun",
            "tr_framenet", "tr_gb", "tr_imst", "tr_kenet", "tr_pud",
            "tr_tourism", "ug_udt", "uk_iu", "ur_udtb", "uz_ut",
            "vi_vtb", "wo_wtb", "yo_ytb", "yue_hk",
            "zh_cfl", "zh_gsd", "zh_gsdsimp", "zh_hk", "zh_pud", "zu_zulu",
        ]
        logger.warning(f"Using hardcoded UD config list: {len(UD_CONFIGS)} configs")

    logger.info(f"Total UD configs: {len(UD_CONFIGS)}")

    # Map: iso639_3 → list of treebank configs
    iso3_to_treebanks: dict[str, list[str]] = {}
    for config in UD_CONFIGS:
        parts = config.split("_", 1)
        if len(parts) < 2:
            continue
        lang_code = parts[0]
        iso3 = UD_CODE_TO_ISO3.get(lang_code, lang_code if len(lang_code) == 3 else None)
        if iso3:
            iso3_to_treebanks.setdefault(iso3, []).append(config)

    logger.info(f"UD languages mapped to ISO 639-3: {len(iso3_to_treebanks)}")
    return iso3_to_treebanks


# ===========================================================================
# STEP 5: Identify Morphology-Relevant Grambank Features
# ===========================================================================
# Known morphology-relevant GB codes from the plan
KNOWN_MORPH_CODES = {
    "GB039", "GB041", "GB042", "GB043", "GB044", "GB046",
    "GB047", "GB048", "GB049", "GB051",
    "GB070", "GB071", "GB072", "GB073",
    "GB084", "GB116", "GB170",
    "GB400", "GB408",
}

MORPH_KEYWORDS = [
    "morphological", "inflect", "affix", "suffix", "prefix",
    "bound morpheme", "suppletive", "allomorph", "conjugat", "declens",
    "derivat", "case marking", "case system", "noun class",
    "agreement with", "agree with",
]

MORPH_DOMAINS = {"verbal domain", "nominal domain", "pronoun"}
MORPH_GROUPINGS = {
    "argument marking (core)", "argument marking (non-core)",
    "TAME", "number", "class",
}


def step5_identify_morph_features(
    parameters_path: Path, feature_grouping_path: Path
) -> list[str]:
    """Identify morphology-relevant features using domain + keyword + known codes."""
    logger.info("STEP 5: Identifying morphology-relevant features...")

    # Load parameters
    params = read_csv(parameters_path)
    param_lookup = {p["ID"]: p for p in params}
    logger.info(f"  Total Grambank parameters: {len(params)}")

    # Set A: Domain-based selection
    domain_features = set()
    try:
        grouping = read_csv(feature_grouping_path)
        for row in grouping:
            fid = row.get("Feature_ID", "")
            domain = row.get("Main_domain", "")
            finer = row.get("Finer_grouping", "")
            if domain in MORPH_DOMAINS and finer in MORPH_GROUPINGS:
                domain_features.add(fid)
        logger.info(f"  Set A (domain-based): {len(domain_features)} features")
    except Exception as e:
        logger.warning(f"  Feature grouping unavailable, skipping Set A: {e}")

    # Set C: Known codes (plan-verified morphological features)
    logger.info(f"  Set C (known codes): {len(KNOWN_MORPH_CODES)} features")

    # Union of A + C only (authoritative domain classification + known codes)
    all_morph = domain_features | KNOWN_MORPH_CODES
    # Filter to only codes that actually exist in parameters
    valid_ids = {p["ID"] for p in params}
    all_morph = sorted(all_morph & valid_ids)
    logger.info(f"  Final morph feature set (union): {len(all_morph)} features")
    return all_morph


# ===========================================================================
# STEP 6: Compute Per-Language Morphological Complexity Index
# ===========================================================================
def step6_compute_morph_index(
    values_path: Path, morph_features: list[str]
) -> dict[str, dict]:
    """Compute morphological complexity index for each language.

    Returns: lang_id → {morph_index, n_coded, n_present, feature_values, richness_raw}
    """
    logger.info("STEP 6: Computing per-language morphological complexity...")

    morph_set = set(morph_features)
    # Accumulate per language
    lang_data: dict[str, dict[str, int]] = {}  # lang_id → {feature_id: value}

    values = read_csv(values_path)
    logger.info(f"  Total value rows: {len(values):,}")

    for row in values:
        param_id = row["Parameter_ID"]
        if param_id not in morph_set:
            continue
        val = row["Value"]
        if val not in ("0", "1"):
            continue
        lang_id = row["Language_ID"]
        if lang_id not in lang_data:
            lang_data[lang_id] = {}
        lang_data[lang_id][param_id] = int(val)

    logger.info(f"  Languages with morph data: {len(lang_data):,}")

    # Compute indices
    results = {}
    excluded = 0
    for lang_id, fvals in lang_data.items():
        n_coded = len(fvals)
        n_present = sum(fvals.values())
        if n_coded < 15:
            excluded += 1
            continue
        morph_index = round(n_present / n_coded, 4) if n_coded > 0 else None
        results[lang_id] = {
            "grambank_morph_index": morph_index,
            "grambank_morph_richness_raw": n_present,
            "n_morph_features_coded": n_coded,
            "n_morph_features_present": n_present,
            "individual_feature_values": fvals,
        }

    logger.info(f"  Languages with index (≥15 coded): {len(results):,}")
    logger.info(f"  Languages excluded (<15 coded): {excluded}")
    return results


# ===========================================================================
# STEP 7 + 8: Map to UD and Assemble Output
# ===========================================================================
def step7_8_assemble(
    grambank_langs_path: Path,
    glottolog_path: Path,
    morph_data: dict[str, dict],
    ud_map: dict[str, list[str]],
    morph_feature_ids: list[str],
) -> list[dict]:
    """Map Grambank languages to UD and assemble final records."""
    logger.info("STEP 7-8: Mapping to UD and assembling output...")

    # Load Grambank languages
    gb_langs = read_csv(grambank_langs_path)
    logger.info(f"  Grambank languages: {len(gb_langs):,}")

    # Load Glottolog for supplementary ISO mapping
    glotto_iso_map: dict[str, str] = {}
    try:
        glotto_rows = read_csv(glottolog_path)
        for row in glotto_rows:
            gc = row.get("Glottocode", "")
            iso = row.get("ISO639P3code", "")
            closest = row.get("Closest_ISO369P3code", "")
            if gc:
                if iso:
                    glotto_iso_map[gc] = iso
                elif closest:
                    glotto_iso_map[gc] = closest
        logger.info(f"  Glottolog ISO mappings loaded: {len(glotto_iso_map):,}")
    except Exception as e:
        logger.warning(f"  Glottolog load failed, using Grambank ISO only: {e}")

    # ISO 639-3 specific → macro-language mapping
    # UD uses macro-language codes; Grambank uses specific codes
    SPECIFIC_TO_MACRO: dict[str, str] = {
        "arb": "ara",  # Standard Arabic → Arabic
        "cmn": "zho",  # Mandarin Chinese → Chinese
        "pes": "fas",  # Iranian Persian → Persian
        "zsm": "msa",  # Standard Malay → Malay
        "nob": "nor",  # Norwegian Bokmål → Norwegian
        "nno": "nor",  # Norwegian Nynorsk → Norwegian (keep both)
        "swh": "swa",  # Swahili → Swahili (macro)
        "lvs": "lav",  # Standard Latvian → Latvian
        "ekk": "est",  # Standard Estonian → Estonian
        "khk": "mon",  # Halh Mongolian → Mongolian
        "ydd": "yid",  # Eastern Yiddish → Yiddish
        "uzn": "uzb",  # Northern Uzbek → Uzbek
        "azj": "aze",  # North Azerbaijani → Azerbaijani
        "kmr": "kur",  # Kurmanji → Kurdish
        "ckb": "kur",  # Central Kurdish → Kurdish
        "prs": "fas",  # Dari → Persian
        "zlm": "msa",  # Malay (macro) individual
        "plt": "mlg",  # Plateau Malagasy → Malagasy
    }

    # Build per-language records
    # Track best per ISO code (most features coded)
    iso_to_records: dict[str, dict] = {}
    gb_only_count = 0

    for lang in gb_langs:
        lang_id = lang["ID"]
        glottocode = lang.get("Glottocode", "")
        iso3 = lang.get("ISO639P3code", "").strip()

        # Fallback to Glottolog if ISO missing
        if not iso3 and glottocode:
            iso3 = glotto_iso_map.get(glottocode, "")

        if not iso3:
            continue  # Skip languages without any ISO code

        # Also register under macro-language code if applicable
        macro = SPECIFIC_TO_MACRO.get(iso3)

        # Get morphological data
        mdata = morph_data.get(lang_id)
        if mdata is None:
            continue  # No morph data (excluded or not enough features)

        name = lang.get("Name", "")
        family = lang.get("Family_name", "")
        macroarea = lang.get("Macroarea", "")
        lat_str = lang.get("Latitude", "")
        lon_str = lang.get("Longitude", "")
        lat = round(float(lat_str), 2) if lat_str else None
        lon = round(float(lon_str), 2) if lon_str else None

        # Collect UD treebanks from both specific and macro codes
        ud_treebanks = list(ud_map.get(iso3, []))
        if macro:
            ud_treebanks.extend(ud_map.get(macro, []))
        has_ud = len(ud_treebanks) > 0

        # Register under both specific ISO and macro code
        codes_to_register = [iso3]
        if macro:
            codes_to_register.append(macro)

        for code in codes_to_register:
            # Collect treebanks for this code
            code_treebanks = list(ud_map.get(code, []))
            if code == iso3 and macro:
                code_treebanks.extend(ud_map.get(macro, []))
            elif code == macro:
                code_treebanks.extend(ud_map.get(iso3, []))
            # Deduplicate
            code_treebanks = sorted(set(code_treebanks))
            code_has_ud = len(code_treebanks) > 0

            # If we already have this code, keep the one with more features coded
            if code in iso_to_records:
                existing = iso_to_records[code]
                if mdata["n_morph_features_coded"] <= existing["_n_coded"]:
                    continue

            record = {
                "iso3": code,
                "glottocode": glottocode,
                "language_name": name,
                "family_name": family,
                "macroarea": macroarea,
                "latitude": lat,
                "longitude": lon,
                "ud_treebanks": code_treebanks,
                "has_grambank": True,
                "has_ud": code_has_ud,
                "morph": mdata,
                "_n_coded": mdata["n_morph_features_coded"],
            }
            iso_to_records[code] = record

    # Add UD-only languages (no Grambank data)
    ud_only = 0
    for iso3, treebanks in ud_map.items():
        if iso3 not in iso_to_records:
            iso_to_records[iso3] = {
                "iso3": iso3,
                "glottocode": "",
                "language_name": "",
                "family_name": "",
                "macroarea": "",
                "latitude": None,
                "longitude": None,
                "ud_treebanks": treebanks,
                "has_grambank": False,
                "has_ud": True,
                "morph": None,
                "_n_coded": 0,
            }
            ud_only += 1

    logger.info(f"  Total unique ISO languages: {len(iso_to_records):,}")
    both = sum(1 for r in iso_to_records.values() if r["has_grambank"] and r["has_ud"])
    gb_only = sum(1 for r in iso_to_records.values() if r["has_grambank"] and not r["has_ud"])
    logger.info(f"  Both Grambank + UD: {both}")
    logger.info(f"  Grambank only: {gb_only}")
    logger.info(f"  UD only: {ud_only}")

    # Format into schema examples
    examples = []
    for iso3, rec in sorted(iso_to_records.items()):
        morph = rec["morph"]

        # Build output dict
        if morph is not None:
            output_dict = {
                "grambank_morph_index": morph["grambank_morph_index"],
                "grambank_morph_richness_raw": morph["grambank_morph_richness_raw"],
                "n_morph_features_coded": morph["n_morph_features_coded"],
                "n_morph_features_present": morph["n_morph_features_present"],
                "individual_feature_values": morph["individual_feature_values"],
            }
        else:
            output_dict = {
                "grambank_morph_index": None,
                "grambank_morph_richness_raw": None,
                "n_morph_features_coded": 0,
                "n_morph_features_present": 0,
                "individual_feature_values": {},
            }

        example = {
            "input": iso3,
            "output": json.dumps(output_dict, ensure_ascii=False),
            "metadata_fold": "overlap" if (rec["has_grambank"] and rec["has_ud"]) else "all",
            "metadata_iso639_3_code": iso3,
            "metadata_glottocode": rec["glottocode"],
            "metadata_language_name": rec["language_name"],
            "metadata_family_name": rec["family_name"],
            "metadata_macroarea": rec["macroarea"],
            "metadata_ud_treebanks": rec["ud_treebanks"],
            "metadata_has_grambank": rec["has_grambank"],
            "metadata_has_ud": rec["has_ud"],
            "metadata_grambank_morph_feature_ids_used": morph_feature_ids,
            "metadata_latitude": rec["latitude"],
            "metadata_longitude": rec["longitude"],
        }
        examples.append(example)

    return examples


# ===========================================================================
# MAIN
# ===========================================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Grambank Morphosyntactic Complexity + UD Mapping Pipeline")
    logger.info("=" * 60)

    # STEP 1: Download Grambank
    gb_files = step1_download_grambank()

    # STEP 2: Download feature grouping
    fg_path = step2_download_feature_grouping()

    # STEP 3: Download Glottolog
    glotto_path = step3_download_glottolog()

    # STEP 4: Get UD configs
    ud_map = step4_get_ud_configs()

    # STEP 5: Identify morph features
    morph_features = step5_identify_morph_features(
        gb_files["parameters.csv"], fg_path
    )

    # STEP 6: Compute morph index
    morph_data = step6_compute_morph_index(gb_files["values.csv"], morph_features)

    # STEP 7+8: Assemble output
    examples = step7_8_assemble(
        gb_files["languages.csv"],
        glotto_path,
        morph_data,
        ud_map,
        morph_features,
    )

    # STEP 9: Save output
    logger.info("STEP 9: Saving output...")
    output = {
        "datasets": [
            {
                "dataset": "grambank_morph_complexity_ud",
                "examples": examples,
            }
        ]
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(examples)} examples to {out_path}")
    logger.info(f"File size: {out_path.stat().st_size:,} bytes")

    # Summary stats
    overlap = [e for e in examples if e["metadata_has_grambank"] and e["metadata_has_ud"]]
    logger.info(f"Languages with both Grambank + UD: {len(overlap)}")

    if overlap:
        indices = [json.loads(e["output"])["grambank_morph_index"] for e in overlap if json.loads(e["output"])["grambank_morph_index"] is not None]
        if indices:
            logger.info(f"  Morph index range: [{min(indices):.3f}, {max(indices):.3f}]")
            logger.info(f"  Morph index mean: {sum(indices)/len(indices):.3f}")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
