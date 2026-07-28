# Food Safety Compass — Project Context

## What this is
TAU **Text Mining 2026** capstone project (instructor: Shay Palachy Affek). Goal:
solve a real problem with a **text-analysis model built/adapted by us**, demonstrating
*deep understanding of course text-mining techniques* — NOT standard tabular ML.

**Problem:** detect food-safety / health hazards (allergic reactions, food poisoning,
etc.) in Yelp restaurant reviews. Framed as **binary text classification** (`is_hazard`),
with room to extend to hazard-type discovery (topic modeling).

## Hard constraints from the assignment
- **Focus on text analysis.** The brief explicitly says NOT to invest depth in tabular
  processing / manual feature engineering ("אין צורך להיכנס לעומק של עיבוד טבלאות").
- Must use **course text-mining techniques**: Word Embeddings, Topic Models, Document
  Embeddings, fine-tuning pretrained models (BERT/T5/spaCy), generative LMs. Transfer
  learning is encouraged, especially with little data.
- Grading rewards **depth, comparison, and understanding** — not effort or raw accuracy.
- 2-person team. Submit via GitHub (add instructor + Tal as contributors).

## Deadlines
- **Submission: 2026-08-02, midnight** — code + presentation (optional written doc).
- **Presentations: 2026-08-03** (≤15 min: problem, text-based solution, results, conclusions).

## Grade structure
- Problem definition & approach — 20%
- Text-mining design & implementation — 40%
- Results analysis & evaluation — 25%
- Presentation & summary — 15%

Rubric bands (Appendix B): one technique + TF-IDF classifier, no comparison, shallow
analysis = 55–65. Deep understanding of **≥2 techniques** + comparison + multiple
baselines + error analysis + variant experiments = 80–89. + understanding *why*
techniques work/fail, tied to the business problem = 90–100.

## Current state of the repo
- [preprocessing/final_project_preprocessing.ipynb](preprocessing/final_project_preprocessing.ipynb)
  — filters Yelp businesses to food/restaurants, streams + keyword-filters reviews,
  creates the weak `is_hazard` label (severe keyword + low stars → 1), exports
  `data/processed_allergy_hazard_dataset.csv`.
- [postprocessing/final_project_postprocessing.ipynb](postprocessing/final_project_postprocessing.ipynb)
  — engineers tabular/lexicon features (medical-lexicon density, VADER negative
  intensity, negation-window flag), saves `enriched_allergy_hazard_dataset.csv`, plus 3 EDA figures.
- Final schema: `stars, useful, funny, cool, text, word_count, char_count,
  exclamation_count, is_hazard (label), medical_lexicon_density, vader_neg_intensity,
  negation_window_flag`.
- **Modeling code now exists under [src/](src/) and is wired together by [main.py](main.py):**
  - [src/data_pipeline.py](src/data_pipeline.py) — loads `enriched_allergy_hazard_dataset.csv`,
    stratified train/test split (80/20, `random_state=42`); `with_validation=True` further
    carves a validation split out of train (64/16/20 overall, test split unchanged).
    Checkpoint selection and the Optuna objective see only the validation split, so the
    test-split numbers are out-of-selection.
  - [src/baseline_model.py](src/baseline_model.py) — sklearn `Pipeline`: TF-IDF (2500 features,
    English stopwords) on `text` + `StandardScaler` on the tabular/lexicon features, feeding an
    `XGBClassifier`. Reports `classification_report` + confusion matrix. This is the "TF-IDF +
    classifier" baseline from the step-4 plan (item 1), but note it also feeds in the manual
    lexicon features alongside TF-IDF, not TF-IDF alone.
  - [src/sota_model.py](src/sota_model.py) — fine-tunes `microsoft/deberta-v3-base` (transfer
    learning, step-4 item 3) with a **custom `AsymmetricSafetyLoss`** (50x weight on false
    negatives on the hazard class) via a `CustomSafetyTrainer` subclass, and a **lowered
    decision threshold (0.20)** in `compute_metrics` — both choices tuned to bias the model
    toward recall, i.e. explicitly optimizing for not missing hazard reports over precision.
  - [main.py](main.py) — orchestrates: train/eval baseline → Optuna sweep (3 trials) over
    `learning_rate`/`batch_size` for the DeBERTa fine-tune, objective = `eval_recall`, logged to
    Weights & Biases. Device gate in [config/settings.py](config/settings.py) supports
    MPS/CUDA/CPU.
  - [src/losses.py](src/losses.py) — three loss formulations, comparable head to head:
    `pos_weight` (the original label-keyed weighting, which is exactly
    `BCEWithLogitsLoss(pos_weight=w)`), `focal_asymmetric` (weight scales with `(1-p)^gamma`),
    and `fn_gated` (penalty only where `p < tau`). The latter two are genuinely
    error-dependent, so "penalises false negatives" is now a true statement.
  - [src/topic_model.py](src/topic_model.py) — LDA + NMF over the 1,500 hazard reviews,
    swept over K ∈ {2,3,4,5,6,8,10}, scored with in-corpus NPMI coherence and validated
    against the LLM-assigned `hazard_type`. This is the **second course technique**.
  - [src/features.py](src/features.py) — feature engineering lifted out of the postprocessing
    notebook so the fresh holdout set can be enriched identically. Verified to reproduce the
    training CSV exactly on all six derived columns.
  - [analysis/error_analysis.py](analysis/error_analysis.py) — rule-based failure-mode
    taxonomy applied to both model errors and the labelling rule itself.
  - [tests/test_losses.py](tests/test_losses.py) — 9 tests; notably a regression guard
    asserting an all-positive model is caught by F2/PR-AUC/flag-rate.

## Known risks / gaps (most important)
- **Label leakage: addressed, with one caveat.** `config/settings.py` excludes `stars`,
  `medical_lexicon_density`, and `negation_window_flag` from `TABULAR_FEATURES` (all three
  derive from the labelling rule). A gold evaluation set now exists — see below. The residual
  caveat: TF-IDF over raw text still trivially recovers a keyword-based label, which is why the
  baseline scores far higher against the heuristic label than against gold.
- **Two gold sets, only one of them valid for model metrics:**
  - `labeling/gold_dataset.csv` (1,500 rows) was sampled from the *enriched* dataset, so
    **1,334 of its rows are in the training split**. Unusable for model evaluation. Still valid
    for measuring the labelling rule itself (85.8% agreement; heuristic precision 73.2%,
    recall 97.9%) because that comparison never consults a model.
  - `labeling/gold_dataset_holdout.csv` — sampled from `build_holdout_pool.py` output, which is
    verified to have **zero overlap** with the enriched dataset. This is the set to report on.
- Selection metrics are now F2 (checkpoint) and PR-AUC (hyperparameter search); bare recall was
  removed because an all-positive model maximises it by definition (regression-guarded in
  `tests/test_losses.py`) — that reason is sound a priori and stands on its own. The all-positive
  collapse itself was observed historically under the label-keyed `pos_weight` loss but was never
  written to an artifact, and the 8-cell grid has now **positively refuted** it under the current
  protocol: nothing collapses at w=50 in either formulation (`pos_weight@50` flag rate 0.1967,
  `focal_asymmetric@50` 0.1858, `collapsed=False` in all 8 rows). Do not quote the collapse as a
  measurement anywhere; cite the grid and state that w=50 does not degenerate.
- The heuristic's 97.9% recall is measured *inside* an already keyword-filtered dataset, so it
  is circular and optimistic. The holdout set gives the honest estimate.

## Step-4 modeling direction (the main deliverable)
Build a comparison of text-mining approaches, not a single classifier:
1. **TF-IDF + classifier** — ✅ `src/baseline_model.py`. Now class-balanced via
   `scale_pos_weight` so the comparison against the weighted transformer is fair.
   Counted as a *baseline*, not as one of the five course techniques.
2. **Word/Document embeddings** — ❌ not implemented; deliberately skipped in favour of
   topic modeling, which connects to the hazard-type research question.
3. **Fine-tuned transformer** — ✅ DeBERTa-v3-base (`src/sota_model.py`), asymmetric loss
   + lowered threshold, Optuna-tuned on PR-AUC. Course technique #1.
4. **Topic modeling (LDA/NMF)** — ✅ `src/topic_model.py`. Course technique #2.

### Loss-variant grid findings (results/grid_search_loss_variants.csv)
The signature contribution is the `AsymmetricSafetyLoss`, so it needs a head-to-head. As of
**2026-07-28** the committed grid is complete: **8 rows** — `{pos_weight, focal_asymmetric}` x
`{w=1, 5, 15, 50}` — every one re-trained from scratch under `--force`, **all at 3 epochs**
(900 optimiser steps, verified from `results/grid_search.log`), and **all scored on both ground
truths**. The earlier mixed 2-/3-epoch table is superseded and its numbers are void.
- **The heuristic metric is saturated and, worse, uninformative.** Validation PR-AUC spans
  0.9864–0.9894 across all eight — a spread of **0.0030**. On the gold holdout the same eight
  span 0.7219–0.8211 — a spread of **0.0992, 33× larger**. The decisive statistic is the rank
  correlation between the two ground truths: **Spearman ρ = −0.24** (Pearson −0.36). The
  validation ranking does not merely have low resolution; it carries no information about
  out-of-sample ordering. This is the strongest methodological result in the project.
- **Run-to-run noise is large and must be quoted.** Four configurations have now been trained
  twice under nominally identical settings. Three reproduced to within 0.006 gold PR-AUC; one
  (`pos_weight@5`) moved **0.054** — over half the entire between-configuration spread. Both of
  its runs selected the epoch-2 checkpoint, so this is not a selection flip; the trajectories
  diverged by epoch 1 and the cause is unattributed (most likely non-deterministic MPS kernels).
  **Treat gold gaps below ~0.05 as unresolved.** Note also that no `seed=` is passed to
  `TrainingArguments`, so every run uses the HuggingFace default of 42 — the 0.054 is
  non-determinism at a *fixed* seed and is a lower bound on true seed variance.
- **What the variants actually do — report the shape, not a ranking.** Across w = 1, 5, 15, 50
  the gold PR-AUC of `pos_weight` is 0.8096, 0.7956, 0.7905, 0.7219: a **monotone decline**,
  range 0.0877. `focal_asymmetric` gives 0.7913, 0.7781, 0.8211, 0.8021: **non-monotone**, range
  0.0430. A uniform penalty on every positive degrades out-of-sample ranking as it grows; the
  focal penalty decays as `(1-p)^gamma` and stops responding to the weight. The defensible claim
  is **the custom loss buys insensitivity to a hyperparameter, not peak performance** — and lead
  with monotone-vs-non-monotone, which does not depend on the noise estimate. Do not quote the
  old "7x less sensitive"; do not quote the endpoint-only "8x" either (it ignores the two middle
  points where the focal curve actually moves).
- **The deployed configuration is now in the grid and reproduces.** `focal_asymmetric @ w=50`
  has been trained three times across two independent code paths, scoring gold PR-AUC 0.7961
  (grid, 27 Jul), 0.8021 (grid, 28 Jul) and 0.8045 (`analysis.py`, the reported system) — a
  range of 0.0084. It ranks 3rd of 8 on gold PR-AUC, 0.019 behind `focal_asymmetric@15`, i.e.
  well inside the noise floor. **Do not switch the deployed config**; disclose the ordering and
  the margin instead.
- **Fixed-threshold metrics disagree with PR-AUC, for a knowable reason.** Gold F2@0.20 and risk
  cost@0.20 both crown `focal_asymmetric@5`, which flags **75.0%** of the holdout; under a 100:1
  cost ratio and a recall-weighted F2, over-flagging is nearly free. Select on PR-AUC, then
  re-tune the threshold per model.
- **The 0.20 threshold does not transfer across ground truths.** Gold flag rates run 0.659–0.750
  against a 46.0% funnel hazard rate, while the same models flag 0.186–0.208 on validation. Gold
  precision is 0.57–0.63 for every configuration. The threshold was chosen against the heuristic
  label and is miscalibrated on the real distribution — another instance of the central thesis.
- **No collapse anywhere.** `collapsed` is False in all 8 rows on both ground truths; max
  validation flag rate 0.2075. `pos_weight @ w=50` — the exact configuration the old docs claimed
  degenerated to all-positive — sits at flag rate 0.1967, recall 0.921, precision 0.936. The
  collapse claim is now positively refuted, not merely unevidenced.
- The `fn_gated` variant has still never been run.

### Topic-modeling findings so far (results/topic_model_*.csv)
All numbers below are from the committed artifacts (regenerated 2026-07-27 after the
NPMI bool-matmul fix — the earlier all-negative, monotonically-rising coherence values
were an artifact of that bug and are void).
- **Coherence cannot be trusted alone.** NMF's highest coherence is at K=2, a degenerate
  fit where one topic holds 95% of documents (NMF K∈{2,3,4,5} all have a topic holding
  >60%). The code excludes degenerate fits first and reports both the coherence-selected
  and validation-selected K when they disagree (NMF: coherence picks K=8 among healthy
  fits, validation picks K=6; LDA: coherence peaks at K=8, validation at K=4).
- **External validation is the stable selection criterion**: `nmi_above_null` picks
  **LDA K=4** and **NMF K=6** — the reported models.
- **Reproducibility caveat (LDA only):** with identical code, data, and seed, LDA fits
  differ across machines/library versions — a second environment produced different
  topics at K=4 (allergen lift 4.18 vs 2.52, purity 0.716 vs 0.690). NMF (deterministic
  `nndsvda` init) reproduces exactly. Quote NMF numbers as headline results; quote LDA
  numbers only from the committed artifacts, and say they are environment-sensitive.
- Recovery of the known hazard types is **weak**: LDA K=4 purity 0.690 (exactly the
  majority-class baseline), NMF K=6 purity 0.697; NMI 0.09–0.12 against a shuffle null
  of 0.006–0.014.
- But the **lift analysis is the real result**: NMF topic 1 (`gluten, celiac, cross
  contamination, gf` — per-topic NPMI +0.43, by far the most coherent topic found) has
  **lift 5.28** for `allergic_reaction`. What the models cannot do is subdivide the
  dominant `food_poisoning` mass (69% of the validated set). Report this as "the
  technique finds the rare, lexically-distinct hazard type and fails on the common,
  lexically-diffuse one" — that is the *why* the rubric rewards.

### Error-analysis findings (results/error_analysis_*, results/gold_fn_handread.*)
Regenerated 2026-07-28 after widening `NEGATED_HAZARD` to cover the contracted negations
(`haven't/hasn't/hadn't/don't/doesn't/isn't/aren't/nobody/none`). Re-bucketing runs without
retraining via `analysis/rebucket_errors.py`, which reloads full review text — the detail CSVs
store only a 400-char excerpt — and reconstructs predictions from the recorded FP/FN indices.
- **The model inherits its teacher's blind spots.** **65%** of DeBERTa's 197 gold false positives
  sit in the labelling rule's own top two failure modes: 32.5% `illness_mentioned_not_caused_here`
  + 32.0% `neutral_allergen_mention`. Tracing model errors back to label pathology is the
  strongest analytical move in the project. *(Was 66% / 34% + 32% before the regex fix, which
  moved 9 FPs — 4 on gold into `negated_hazard`.)*
- **`unexplained_fn` was a taxonomy defect, not a data mystery.** 17 of the 23 residual gold FNs
  *do* contain an `EXPLICIT_HAZARD`/`ILLNESS_WORD` term, but every FN rule is conditioned on
  `not has_explicit`, so a short low-starred review with a clear hazard word matches nothing and
  falls through. Frame it as auditing our own error taxonomy.
- **All 23 are now hand-named** (`results/gold_fn_handread.md`), so "59% unexplained" → 0%: five
  each of `implicit_hazard`, `second_hand_report`, `contamination_novel_phrasing` (real
  contamination phrased outside the rule's vocabulary) and `label_questionable`, plus 2
  `explicit_hazard_missed` and 1 `mild_or_hedged`.
- ⚠️ **The 256-token truncation hypothesis is refuted — do not quote it.** Measured with the real
  `DebertaV2TokenizerFast` at `max_length=256`: only **1 of 23** has its hazard cue past the
  window, only 2 of 23 exceed 256 tokens at all, and the median cue position is token **39**.
- **5 of 23 gold labels are arguable** at LLM confidence *high* (slip-and-fall, steak-doneness
  mix-up, flavour revulsion read as illness). Ground truth #2 is independent, not infallible —
  state this yourself.
- In 5 cases the only hazard keyword is a **disgust idiom** ("made me want to VOMIT") while the
  real hazard has no keyword; the idiom then suppresses `contamination_no_illness` via its
  `ILLNESS_WORD` guard.

## Working notes
- Large data files are gitignored; raw Yelp JSON is not committed.
- Subsample for fast iteration (preserve class distribution); train/evaluate on full data
  only at decision points or for final reported numbers.
