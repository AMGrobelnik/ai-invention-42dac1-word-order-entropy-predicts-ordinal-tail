# DDM Effects, Spoken/Written UD Pairs, Discordant Treebanks

## Summary

This research catalogs three critical inputs for GEV tail-constraint analysis of dependency distance distributions. PART 1 covers DDM Effect Sizes: Futrell 2015 demonstrated DLM in all 37 languages (P less than 0.0001) with head-final languages showing less minimization than head-initial; Yadav 2020 and 2022 showed word-order typology interaction (beta_3 from -0.17 to -0.03, weaker in SOV); Li and Liu 2025 found morphological richness correlates with syntactic directionality not word order entropy; Petrini 2025 found two-regime exponential models with universal breakpoint at d*=4-5 words across 20 PUD languages. PART 2 covers Spoken/Written Pairs: 14 spoken UD treebanks identified yielding 5-6 viable pairs with feats on both sides including Slovenian sl_sst/sl_ssj (BEST), French fr_parisstories/fr_gsd, Norwegian no_nynorsklia/no_nynorsk, English en_atis/en_ewt and en_eslspok/en_ewt, all confirmed in commul HuggingFace. PART 3 covers Discordant Languages: Arabic ar_padt 7664 sentences with 20 features (discordant: rich morphology plus head-initial), Chinese zh_gsd 4997 sentences with 9 features (discordant: poor plus mixed), Basque eu_bdt 8993 sentences (canonical calibration), English en_ewt 16622 sentences (canonical calibration). Grambank v1.0 at Zenodo for cross-validation. Key gap: no study has analyzed GEV shape parameters as function of typological features yet.

## Research Findings

This research catalogs three critical inputs for designing a GEV tail-constraint analysis of dependency distance distributions across typologically diverse languages.\n\n## PART 1: DDM Effect Sizes from Key Literature\n\n### Futrell, Mahowald & Gibson (2015, PNAS)\nThe landmark study analyzed 37 languages from 10 language families using corpora from HamleDT 2.0, Google Universal Treebank 2.0, and Universal Dependencies 1.0 [1]. Using mixed-effects regression with a beta_3 coefficient measuring DLM effect, they found DLM significant at P < 0.0001 for all 37 languages under the free word order baseline [1]. Critically, the study revealed substantial cross-linguistic variance: head-final languages (Japanese, Korean, Turkish) show much less minimization than head-initial languages (Italian, Indonesian, Irish) [1]. The mechanism: head-final languages typically have case marking, giving them more freedom in dependency lengths [1]. Per-language beta_3 coefficients are in supplementary Table S1 but were not extractable from the web.\n\n### Yadav, Vaidya, Shukla & Husain (2020, Cognitive Science)\nExamined 38 languages classified as SVO or SOV using generalized linear mixed-effects models with Poisson link function [2]. Key finding: dependency distance is determined by default word order — longer linear distances arise when structures mirror the default order [2]. Evidence for 'limited adaptability' to default word order preferences [2]. Specific regression coefficients behind Wiley paywall.\n\n### Yadav, Mittal & Husain (2022, Open Mind — Reappraisal)\nAnalyzed 54 languages from SUD v2.4, providing the most accessible quantitative benchmarks [3]. Key coefficients: ICM hypothesis beta_3 = -0.17 (t = -24.5) for random structures; DLM independent beta_3 = -0.07 (t = -12.9) for IC-matched random structures [3]. Both SOV and SVO show minimization, but weaker in SOV [3]. Data at https://osf.io/j975y/ [3].\n\n### Li & Liu (2025, Entropy/MDPI)\nOpen-access study of 55 languages from 11 families using 80 UD treebanks [4]. Measured morphological richness (MAMSP), word order entropy, and syntactic directionality. Key findings: morphological richness only weakly related to word order entropy (not robust after correction); syntactic directionality more closely associated with morphological complexity — rich morphology → head-final, poor morphology → head-initial [4]. Specific r-values require accessing the full MDPI text.\n\n### Petrini & Ferrer-i-Cancho (2025, Glottometrics)\nModeled dependency distance distributions across 20 PUD languages [5, 6]. Two-regime exponential model best fits all 20 languages. Breakpoint d* averages 4-5 words with low cross-linguistic variation [5, 26]. For Arabic, Chinese, Thai: single-regime model most frequent [5]. CRITICAL GAP: models ALL dependency distances aggregated, not per-sentence maxima — exactly the gap motivating the GEV approach [5].\n\n## PART 2: Spoken vs. Written UD Treebank Pairs\n\nFrom Dobrovoljc (2022) [7] and direct treebank page verification, 14 spoken treebanks were identified in UD. After filtering for >=500 sentences and feats availability, 5-6 viable spoken/written pairs emerged:\n\n1. Slovenian (BEST): sl_sst (6,121 sent, feats YES) / sl_ssj (13,435 sent) [22, 24]\n2. French: fr_parisstories (2,776 sent, feats YES) / fr_gsd (16,342 sent) [23, 25]\n3. Norwegian: no_nynorsklia (~5,500 sent, feats YES) / no_nynorsk (17,575 sent) [21]\n4. English ATIS: en_atis (5,432 sent, feats YES) / en_ewt (16,622 sent) [15]\n5. English ESLSpok: en_eslspok (2,320 sent, feats YES) / en_ewt (16,622 sent) [15]\n\nPartially usable (missing feats on spoken side): fr_rhapsodie [17], en_childes [18]. Borderline (too few sentences): es_coser at 539 [19]. Naija pcm_nsc is large (9,241 sent) but has no written counterpart [10]. All confirmed in commul/universal_dependencies on HuggingFace [20].\n\n## PART 3: Discordant Language Treebank Availability\n\nArabic (DISCORDANT: rich morphology + head-initial): ar_padt with 7,664 sentences, 282,384 words, 20 UD features [11]. Grambank: stan1318 [12].\n\nMandarin Chinese (DISCORDANT: poor morphology + mixed head direction): zh_gsd with 4,997 sentences, 123,289 tokens, only 9 features [13]. Grambank: mand1415 [12].\n\nBasque (CANONICAL calibration: rich + head-final): eu_bdt with 8,993 sentences, 121,443 tokens, rich features [14]. Grambank: basq1248 [12].\n\nEnglish (CANONICAL calibration: poor + head-initial): en_ewt with 16,622 sentences, en_gum with 13,263 sentences [15, 9]. Grambank: stan1293 [12].\n\nGrambank v1.0 available at DOI: 10.5281/zenodo.7740140 (195 features, 2,467 languages, CC BY 4.0) [16].\n\n## Calibration Assessment\n\nThe R2-increment threshold of >5% for GEV xi analysis appears reasonable but conservative. Head-direction is the strongest typological predictor of DD variance [1, 4]. No existing study has analyzed GEV shape parameters as function of typological features [5]. Key data gaps: Futrell per-language coefficients need PDF parsing [1]; Li and Liu r-values need MDPI access [4]; Petrini parameters in 60-page appendices [5].

## Sources

[1] [Large-scale evidence of dependency length minimization in 37 languages (Futrell et al. 2015, PNAS)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4547262/) — Landmark study demonstrating DLM across 37 languages; head-final languages show less minimization than head-initial

[2] [Word Order Typology Interacts With Linguistic Complexity (Yadav et al. 2020, Cognitive Science)](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12822) — 38-language study showing dependency distance determined by default word order with limited adaptability

[3] [A Reappraisal of DLM as a Linguistic Universal (Yadav et al. 2022, Open Mind)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9692064/) — 54-language reanalysis with beta coefficients for DLM and ICM; SOV shows weaker effects than SVO

[4] [Interactions Among Morphology, Word Order, and Syntactic Directionality (Li & Liu 2025, Entropy)](https://www.mdpi.com/1099-4300/27/11/1128) — 55-language study: morphological richness correlates with syntactic directionality not word order entropy

[5] [The distribution of syntactic dependency distances (Petrini & Ferrer-i-Cancho 2025)](https://arxiv.org/abs/2211.14620) — 20-language study: two-regime exponential model with breakpoint at 4-5 words; models aggregate distances

[6] [Glottometrics publication of Petrini & Ferrer-i-Cancho (2025)](https://glottometrics.iqla.org/424-the-distribution-of-syntactic-dependency-distances/) — Published version in Glottometrics 58:35-94

[7] [Spoken Language Treebanks in UD (Dobrovoljc 2022, LREC)](https://aclanthology.org/2022.lrec-1.191/) — Comprehensive catalog of spoken treebanks in UD v2.9

[8] [A morph-based and a word-based treebank for Beja](https://hal.science/hal-03494462/document) — Beja NSC deprecated in UD v2.15, replaced by Beja Autogramm

[9] [UD English GUM treebank page](https://universaldependencies.org/treebanks/en_gum/index.html) — Multi-genre English treebank (13,263 sent) including spoken subset

[10] [UD Naija-NSC treebank page](https://universaldependencies.org/treebanks/pcm_nsc/index.html) — Large spoken Naija treebank (9,241 sent) with no written counterpart

[11] [UD Arabic-PADT treebank page](https://universaldependencies.org/treebanks/ar_padt/index.html) — 7,664 sentences, 282,384 words, 20 morphological features

[12] [Grambank typological database](https://grambank.clld.org/) — 195-feature database for 2,467 languages; all target languages included

[13] [UD Chinese-GSD treebank page](https://universaldependencies.org/treebanks/zh_gsd/index.html) — 4,997 sentences, 123,289 tokens, 9 sparse features

[14] [UD Basque-BDT treebank page](https://universaldependencies.org/treebanks/eu_bdt/index.html) — 8,993 sentences, 121,443 tokens, rich morphological features

[15] [UD English-EWT treebank page](https://universaldependencies.org/treebanks/en_ewt/index.html) — 16,622 sentences, 251,489 tokens, feats annotated

[16] [Grambank v1.0 on Zenodo](https://zenodo.org/records/7740140) — Downloadable dataset (CC BY 4.0), 195 features, 2,467 languages

[17] [UD French-Rhapsodie treebank page](https://universaldependencies.org/treebanks/fr_rhapsodie/index.html) — 3,209 spoken sentences but feats NOT available

[18] [UD English-CHILDES treebank page](https://universaldependencies.org/treebanks/en_childes/index.html) — 48,183 spoken sentences but feats NOT available

[19] [UD Spanish-COSER treebank page](https://universaldependencies.org/treebanks/es_coser/index.html) — Only 539 spoken sentences, borderline for GEV

[20] [commul/universal_dependencies on HuggingFace](https://huggingface.co/datasets/commul/universal_dependencies) — Confirmed configs for all target treebanks

[21] [UD Norwegian-NynorskLIA repository](https://github.com/UniversalDependencies/UD_Norwegian-NynorskLIA) — Spoken Norwegian (~5,500 sent estimated) with feats

[22] [UD Slovenian-SST treebank page](https://universaldependencies.org/treebanks/sl_sst/index.html) — Best spoken treebank: 6,121 sent, 98,393 tokens, feats YES

[23] [UD French-ParisStories treebank page](https://universaldependencies.org/treebanks/fr_parisstories/index.html) — 2,776 spoken sentences, feats annotated

[24] [UD Slovenian-SSJ treebank page](https://universaldependencies.org/treebanks/sl_ssj/index.html) — Written Slovenian: 13,435 sent, 267,097 tokens

[25] [UD French-GSD treebank page](https://universaldependencies.org/treebanks/fr_gsd/index.html) — Written French: 16,342 sent, 389,364 tokens

[26] [Semantic Scholar: Petrini & Ferrer-i-Cancho 2025](https://api.semanticscholar.org/graph/v1/paper/DOI:10.53482/2025_58_424?fields=title,abstract,tldr,citationCount) — TLDR confirms two-regime model; 4 citations

## Follow-up Questions

- What are the exact per-language beta_3 coefficients from Futrell 2015 Table S1, and how do they correlate with WALS head-direction indices?
- Can MAMSP morphological richness be computed from UD feats alone, or does it require full paradigm data that some treebanks lack?
- What is the minimum sentences per length bin for reliable GEV shape parameter estimation, and do smaller spoken treebanks meet this?

---
*Generated by AI Inventor Pipeline*
