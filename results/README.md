# Experiment evidence

## Optional run: fixed 150-document sample

`inspec_150_seed2026_current/` contains the optional 150-document experiment. It was generated with `configs/experiment_150.json` and `data/fixed_subset.jsonl`; the sample is fixed with seed 2026. `run_metadata.json` records the input/config/code hashes and confirms `document_count: 150`. The run includes TF-IDF, TextRank windows 2/3/5/10, and two window-2 ablations.

| Method | F1@5 | F1@10 | All-prediction micro F1 |
|---|---:|---:|---:|
| TF-IDF | 0.2058 | 0.1975 | 0.1939 |
| TextRank, window 2 | 0.1377 | 0.1354 | 0.1429 |
| TextRank, window 3 | 0.1234 | 0.1210 | 0.1174 |
| TextRank, window 5 | 0.1602 | 0.1538 | 0.1516 |
| TextRank, window 10 | 0.1686 | 0.1642 | 0.1641 |

These scores describe only the fixed 150-document sample and differ from the default full-split experiment below.

## Default run: paper-aligned experiment on all 500 documents

The command-line defaults run the complete split from `data/inspec_paper_test.jsonl` using `configs/experiment_paper_aligned.json`. The fresh default run is saved to `inspec_500_uncontrolled_default_current/`; `inspec_500_uncontrolled_paper_aligned_updated_v4/` contains the previously analyzed full-split evidence. Both use all Inspec test IDs, the original uncontrolled labels, TextRank windows 2/3/5/10, and two independent window-2 ablations: no POS filter and no single-character tokens. The source/label check is recorded in `data/inspec_paper_metadata.json`; run configuration, code/config/input hashes, package versions, and POS tagger resource checksums are copied into each `run_metadata.json`. The implementation uses stable first-occurrence tie-breaking for word nodes; `_final_v2/` is the pre-fix comparison.

### All-prediction corpus micro scores

This reproduces the paper's all-assigned-keyphrase aggregation shape. The local implementation and tagger can still differ from the original implementation, so these values are not expected to equal Table 1.

| Method | Assigned mean | Correct mean | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF (local course baseline) | 13.330 | 2.096 | 0.1572 | 0.2142 | 0.1813 |
| TextRank, window 2 | 11.752 | 1.482 | 0.1261 | 0.1514 | 0.1376 |
| TextRank, window 3 | 12.878 | 1.342 | 0.1042 | 0.1371 | 0.1184 |
| TextRank, window 5 | 12.990 | 1.722 | 0.1326 | 0.1760 | 0.1512 |
| TextRank, window 10 | 12.882 | 1.860 | 0.1444 | 0.1901 | 0.1641 |

The official paper reports TextRank window 2 at P=0.312, R=0.431, F1=0.362; windows 3/5/10 at F1=0.326/0.322/0.322. The local setup matches the 500-test-document scope, uncontrolled references, and those four window values, but its scores remain materially different. The updated run has window-2 corpus F1=0.1376 versus 0.1094 in `_final_v2/`; this is consistent with the tie fix, but Python differs (3.14 vs 3.10), so do not treat it as a fully controlled paired comparison. It remains far below the paper. Treat this as a setting-aligned partial reproduction, not a numerical reproduction. TF-IDF is an added course baseline and is not one of the paper's Table 1 systems.

### Fixed-cutoff macro scores

| Method | F1@5 | F1@10 |
|---|---:|---:|
| TF-IDF | 0.1856 | 0.1798 |
| TextRank, window 2 | 0.1377 | 0.1327 |
| TextRank, window 3 | 0.1232 | 0.1183 |
| TextRank, window 5 | 0.1612 | 0.1519 |
| TextRank, window 10 | 0.1689 | 0.1626 |

The values above are document-macro F1 with a fixed denominator of 5 or 10 predictions per abstract. The previous table accidentally listed precision@5 under the F1@5 heading; this table is corrected directly from the current `metrics.csv`.

### Single-factor ablations

The ablation removes only the noun/adjective POS filter and holds the dataset, window=2, damping, convergence threshold, and top-node fraction fixed.

| Method | Candidate count mean | Candidate coverage macro | F1@5 | F1@10 | All-prediction micro F1 |
|---|---:|---:|---:|---:|---:|
| TextRank, window 2, POS filter on | 66.554 | 0.6714 | 0.1377 | 0.1327 | 0.1376 |
| TextRank, window 2, POS filter off | 943.986 | 0.7760 | 0.0085 | 0.0139 | 0.0346 |
| TextRank, window 2, exclude 1-character tokens | 65.322 | 0.6690 | 0.1433 | 0.1360 | 0.1411 |

Removing the POS filter increases candidate coverage but sharply reduces F1; it should remain enabled. Excluding one-character tokens slightly improves window-2 F1 (0.1376 to 0.1411) while decreasing candidate coverage (0.6714 to 0.6690); this modest gain does not explain the gap to the paper and may discard legitimate one-character labels. Keep it as an ablation, not the paper-aligned main method. The experiment contains 3,000 converged TextRank rankings (500 documents × six graph conditions); the maximum iteration count was 69. The updated full run took about 43 seconds in Python 3.14.0 / NLTK 3.9.1. The source tagger resource hashes match `_final_v2/`. See `metrics.csv`, `predictions.jsonl`, `cases_10.jsonl`, `run.log`, and `run_metadata.json` for the complete evidence.

### Miss classification

`miss_analysis.json` classifies references using the same normalized labels, POS-filtered contiguous candidate set, and saved predictions as the current run. Counts are cross-checked against `metrics.csv`. Of 4,893 distinct references, 3,254 are generatable under the current candidate rules and 1,639 are not: 1,142 are marked abstractive in the MIDAS source, 474 extractive references are blocked by POS filtering, 11 by punctuation boundaries, and 12 have no contiguous token match. Of the 3,254 generatable references, TextRank selects 741 at window 2 and 930 at window 10; it leaves 2,513 and 2,324, respectively, unselected. This points to ranking/selection quality as the larger opportunity after candidate generation, while the POS and phrase rules also exclude some extractive labels.

This is a diagnostic on the test set, not a tuning result. The older 150-document subset was sampled from this same test set and is not suitable for tuning. A separate upstream validation split was obtained and used for the distinct development-tuned variant below.

### Development-tuned variant (separate from Table 1 reproduction)

The upstream MIDAS validation split and AKE `dev.uncontr.json` labels were downloaded in memory, matched on all 500 IDs, and checked for normalized label-set equality (500/500). No validation text or labels were added to the workspace. `inspec_500_uncontrolled_paper_aligned_updated_v4/dev_fraction_sweep.json` records source revisions and hashes, the sweep, and its selection rule.

A single shared `top_word_fraction` was selected by maximizing the unweighted mean all-prediction micro F1 over windows 2/3/5/10. The search compared 0.20, 0.25, 1/3, 0.40, 0.50, 0.60, 2/3, 0.75, 0.90, and 1.00. The selected value was 0.90: validation mean F1 across windows was 0.2769, versus 0.1403 at the paper's 1/3 setting. This is development-set selection and is not the paper-aligned setting.

The 0.90 result is not meaningfully better than selecting all eligible words: at fraction 1.00 the mean F1 is 0.2762, only 0.0007 lower (paired 95% bootstrap CI for the difference, 0.90 minus 1.00: [-0.0015, 0.0028]). Fraction 0.90 assigns about 25.5 phrases per abstract versus 28.9 at 1.00, and unlike 1.00 it still lets TextRank's graph ordering affect selection. Thus 0.90 is a reasonable lower-output operating point for studying the windows, not a proven F1 optimum over 1.00.

#### Paired window comparison on validation

The four windows were rerun on the same 500 validation documents at both the fixed 1/3 baseline and the shared validation-selected 0.90 fraction. The rerun reproduces the saved aggregate sweep metrics exactly. Paired document bootstrap intervals (10,000 resamples) for corpus micro F1 show that the window effect depends on the fraction:

| Shared top-word fraction | Window 2 | Window 3 | Window 5 | Window 10 | Mean F1 |
|---|---:|---:|---:|---:|---:|
| 1/3 baseline | 0.1186 | 0.1214 | 0.1561 | 0.1653 | 0.1403 |
| 0.90 selected on validation | 0.2962 | 0.2842 | 0.2657 | 0.2616 | 0.2769 |

At 1/3, window 2 is not clearly different from window 3 (window-2 minus window-3 F1 = -0.0028; 95% paired bootstrap CI [-0.0121, 0.0064]), while windows 5 and 10 outperform it (differences -0.0375 [-0.0473, -0.0279] and -0.0467 [-0.0569, -0.0367]). At 0.90, window 2 outperforms windows 3, 5, and 10 by 0.0120 [0.0082, 0.0159], 0.0304 [0.0256, 0.0353], and 0.0346 [0.0299, 0.0393], respectively. Document-macro F1 has the same ordering at both fractions.

The mean F1 gain at 0.90 over 1/3 is 0.1366 (paired 95% bootstrap CI [0.1260, 0.1473]); the increase is positive at each window. This is useful evidence of an interaction between the graph window and post-ranking selection: at 1/3, larger windows perform better in aggregate, while at 0.90, smaller windows perform better. It does not support a universal claim that one window is best regardless of selection. All 0.90 intervals are conditional/descriptive because that fraction was selected on this same validation split; they are not independent confirmation. Keep one shared fraction across windows for the main comparison, report the fixed 1/3 baseline alongside tuned results, and treat test comparisons as exploratory because the test set was already inspected. Aggregate results and bootstrap details are in `updated_v4/dev_window_paired_bootstrap.json`.

That setting was then run once on the 500-document test set as a separate variant in `inspec_500_uncontrolled_dev_tuned_fraction_0_9/`. Its all-prediction micro results were:

| Window | P | R | F1 | Assigned mean |
|---:|---:|---:|---:|---:|
| 2 | 0.2119 | 0.5532 | 0.3065 | 25.546 |
| 3 | 0.2001 | 0.5322 | 0.2908 | 26.028 |
| 5 | 0.1861 | 0.5126 | 0.2731 | 26.952 |
| 10 | 0.1827 | 0.5079 | 0.2687 | 27.208 |

This improves F1 over the paper-aligned run at each window and restores the paper's qualitative ordering (window 2 highest). However, it assigns roughly twice as many phrases as Table 1's mean of 13.7, and its precision/recall balance differs from the paper. It is therefore a tuned experimental variant, **not** a numerical reproduction. The test set had already been examined in earlier work, so treat this as exploratory evidence rather than a fresh confirmatory test.

#### POS-tagger comparison

The next targeted experiment kept the same pretokenized input, POS-tag set, graph, phrase merging, evaluation, and both fractions (1/3 and 0.90), changing only the POS tagger. On the 500-document validation split, spaCy `en_core_web_sm` 3.8.0 with spaCy 3.8.18 improved the mean all-prediction micro F1 across the four windows from 0.1403 to 0.1436 at 1/3, and from 0.2769 to 0.2843 at 0.90. Candidate coverage increased from 0.6623 to 0.6682. The token sequence was identical for both taggers; their exact POS-tag agreement was about 90.1%.

Following that validation comparison, spaCy was evaluated on the test split at both fixed fractions. NLTK 3.9.1 was recomputed as a control and exactly reproduced the saved test metrics. At 0.90, spaCy F1 was 0.3182, 0.3030, 0.2849, and 0.2774 for windows 2/3/5/10, versus NLTK's 0.3065, 0.2908, 0.2731, and 0.2687. At 1/3, spaCy also improved all four windows, but by smaller amounts. This is a modest and consistent gain, not enough to close the gap to the paper. The test set was already examined in previous runs, so these results are exploratory rather than confirmatory.

The tagger comparison is in `updated_v4/dev_pos_tagger_comparison.json`; the paired test evaluation is in `inspec_500_uncontrolled_dev_tuned_fraction_0_9/spacy_pos_test_comparison.json`. The original paper-aligned run remains NLTK with a 1/3 selection fraction. The spaCy result is a separate implementation variant; its model was installed only in the local virtual environment and is not a project runtime dependency.

#### Joint POS-tagger and edge-weighting comparison

The next validation-only sweep tested a 2×2 combination: NLTK versus spaCy POS tags and binary versus inverse-distance-decayed edge weights. Token positions, phrase construction, graph windows, and document bootstrap resamples were held constant. The sweep checked all four windows at both 1/3 and the previously validation-selected 0.90 fraction; its inputs were the same pinned 500 validation abstracts and labels.

At 0.90, the mean micro F1 across windows was 0.2769 for the NLTK/binary baseline, 0.2843 for spaCy/binary, 0.2913 for NLTK/distance-decay, and 0.2987 for spaCy/distance-decay. The combined variant's mean paired gain over baseline was 0.0218 (95% document-bootstrap percentile interval [0.0174, 0.0260]); its gain was positive for every window: +0.0061, +0.0159, +0.0302, and +0.0347 for windows 2/3/5/10, respectively, with each window's interval above zero. At the project's one-third node budget, the combined variant's mean was 0.1572 versus 0.1403 for baseline, a paired gain of 0.0169 [0.0119, 0.0219], again positive at all four windows.

At a fixed one-third budget, spaCy with inverse-distance-decayed edges was the strongest of these four tested TextRank configurations. This remains a validation-screened result: the bootstrap intervals are descriptive after variant selection. Do not interpret it as independent confirmation or a recovered paper score. The runnable configuration is `configs/experiment_spacy_distance_decay.json`; it opts in to spaCy `en_core_web_sm`, inverse-distance-decayed edge weights, and the same one-third node budget as the paper-aligned baseline. The default configuration remains NLTK with binary edges. spaCy and its English model are optional local dependencies, not part of `code/requirements.txt`.

The complete 2×2 result and per-window paired intervals are in `updated_v4/dev_pos_edge_weighting_sweep.json`. No test-set run of this combined variant was used to choose these settings.

#### Phrase-construction and ranking checks

On validation, emitting every selected-run subphrase of length 1–3 increased predictions substantially and reduced mean F1 across windows from 0.1436 to 0.1197 at the 1/3 fraction, and from 0.2843 to 0.1692 at 0.90. Bridging a single `IN`/`TO` token between selected words increased candidate coverage only from 0.6682 to 0.6715 (15 more reference phrases out of 4,566), while lowering mean F1 to 0.1376 at 1/3 and 0.2541 at 0.90. Combining the two changes also reduced F1. These tested phrase-generation changes are not recommended.

A separate ranking comparison used the same spaCy tags, graph windows, phrase merging, and selection fractions. On validation at 1/3, unweighted graph degree reached mean F1 0.1734 across windows, versus 0.1436 for iterative TextRank; term frequency and TF-IDF reached 0.1951 and 0.1828, respectively. At 0.90, graph degree and TextRank were nearly tied (0.2856 and 0.2843), while the non-graph baselines scored lower. Since the primary comparison is how TextRank window size affects performance, degree/frequency/TF-IDF remain comparison baselines, not replacements for the TextRank method.

The degree variant was evaluated once on test after the validation comparison. At 1/3, mean F1 across windows was 0.1797 for degree versus 0.1472 for spaCy TextRank. At 0.90, it was 0.2988 versus 0.2959; for window 2, the scores were 0.3184 and 0.3182, effectively tied. These are exploratory test results because this test set has been inspected previously. Detailed tables are in `updated_v4/dev_phrase_construction_sweep.json`, `updated_v4/dev_ranking_method_sweep.json`, and `inspec_500_uncontrolled_dev_tuned_fraction_0_9/degree_ranking_test_comparison.json`.

No core algorithm or paper-aligned configuration was changed: the phrase variants did not help, and degree is a distinct graph-ranking baseline rather than TextRank. The validation evidence suggests the largest measured gain still comes from emitting more phrases at the 0.90 selection fraction, which produces about 25–27 phrases per abstract rather than the paper's reported mean of about 13.7.

#### Co-occurrence window convention audit

The paper describes the co-occurrence relation as controlled by the distance between word occurrences, connecting filtered lexical units within a window of maximum N words; it then says edges are added between eligible lexical units that co-occur within a window of N words. It does not explicitly say whether filtered-out tokens count toward N. The current implementation counts original token positions: a window of width N spans N raw token positions, so window 2 connects adjacent original tokens. This is the most direct reading of a window of N words, and no evidence justifies changing it.

To measure the ambiguity, validation data was evaluated with the same spaCy tags, ranking formula, phrase merging, and cutoffs under three graph interpretations. At the paper's 1/3 selection fraction, mean F1 across windows was 0.1436 for the current raw-token window, 0.1500 when maximum raw-token distance was N (an N+1-position span), and 0.1579 when ineligible tokens were removed before applying a width-N window. At 0.90 the corresponding values were 0.2843, 0.2778, and 0.2599. The filtered-token interpretation increases the mean number of window-2 edges from 21.3 to 48.5 and removes isolated nodes by construction, but does not reproduce the paper's trend of performance declining as the window grows. These are sensitivity results, not evidence to choose a convention by score.

The audit confirms the current graph implementation exactly matches its raw-token-width interpretation and the previous validation metrics. It is saved in `updated_v4/dev_window_semantics_sweep.json`. Keep the main implementation unchanged unless an author implementation or further protocol evidence resolves the remaining wording ambiguity.

#### Figure 2 worked-example check

The paper reports the illustrative high-scoring words `numbers` (1.46), `inequations` (1.45), `linear` (1.29), `diophantine` (1.28), `upper` (0.99), `bounds` (0.99), and `strict` (0.77). The corresponding Inspec example is present as document ID 1939. Under the current NLTK setup and window 2, the scores are 1.4595, 1.1791, 1.1791, 1.1191, 1.0000, 1.0000, and 1.1191, respectively. Five of the paper's seven words appear in the implementation's top seven; the listed scores have mean absolute difference 0.1303. The `numbers` score matches to two decimals, and `upper`/`bounds` are close, but the differences for `inequations`, `diophantine`, and especially `strict` are substantial.

This is only a conditional comparison: the paper's surrounding discussion does not identify which window produced its illustrative scores, and it does not specify the POS tagger/model details needed to recreate the candidate graph. NLTK and spaCy yield identical scores for these seven words at window 2 on this example, while differing in other eligible/high-ranked words; therefore, the tagger alone does not explain the score discrepancies. The evidence confirms partial agreement in the intermediate ranking but does not isolate a graph or ranking bug, so no paper-aligned code change is justified. This single test-example diagnostic was not used for parameter selection or aggregate score tuning. Details for both taggers and windows are in `updated_v4/paper_example_reproduction.json`.

## Superseded or incomplete runs

- `inspec_500_uncontrolled_paper_aligned/`, `inspec_500_uncontrolled_paper_aligned_final/`, `inspec_500_uncontrolled_paper_aligned_final_v2/`, and `_updated_v3/` are superseded runs; use `_updated_v4/` above.
- `inspec_150_seed2026_current/` is the optional 150-document run; other `inspec_150_seed2026*` directories are earlier experiments.

## Output files

- `run.log`: execution log for the directory.
- `run_metadata.json`: config/input/code hashes, ordered IDs, runtime versions, tagger checksums, timing, and case sample IDs.
- `metrics.csv`: macro P/R/F1@5 and @10, plus all-prediction corpus micro P/R/F1 and assignment counts.
- `predictions.jsonl`: one row per document with uncontrolled references, full phrase rankings, scores, and graph convergence diagnostics.
- `miss_analysis.json`: aggregate reference-generation and prediction-selection miss categories; no phrase text is duplicated into this diagnostic.
- `dev_fraction_sweep.json`: validation-only comparison of common top-node fractions; the test set was not used for selection.
- `dev_pos_tagger_comparison.json`: validation-only NLTK/spaCy comparison using identical tokens.
- `dev_pos_edge_weighting_sweep.json`: validation-only factorial comparison of POS tagger and binary/distance-decayed TextRank edges.
- `dev_phrase_construction_sweep.json`: validation-only phrase emission/connector variants.
- `dev_ranking_method_sweep.json`: validation-only TextRank/degree/frequency/TF-IDF ranking comparison.
- `dev_window_semantics_sweep.json`: validation-only sensitivity analysis of window-position interpretations.
- `dev_window_paired_bootstrap.json`: paired document-bootstrap comparison of the four windows at the fixed 1/3 and validation-selected 0.90 fractions.
- `paper_example_reproduction.json`: intermediate ranking check against the paper's Figure 2 example; test example only, not a tuning result.
- `cases_10.jsonl`: ten deterministic case records with IDs, references, predictions, and ranking diagnostics; the case seed only chooses IDs.
