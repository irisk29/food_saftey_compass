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
  removed after the all-positive collapse was confirmed at w=50.
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

### Topic-modeling findings so far (results/topic_model_*.csv)
- **Intrinsic coherence and extrinsic validation independently agree on K=4 for LDA:**
  NPMI coherence peaks at K=4 (+0.106) — the same K that maximises NMI-above-null
  against the LLM hazard types. (An earlier version reported NPMI rising monotonically
  with K; that was an artifact of a bool-matmul bug in the co-occurrence counts, fixed
  2026-07-27 — all persisted coherence values are now positive and non-monotonic.)
- Coherence still cannot be trusted *alone*: NMF's highest coherence is at K=2, a
  degenerate fit where one topic holds 95% of documents (NMF K∈{2,3,4,5} all have a
  topic holding >60%). The code excludes degenerate fits first and reports both the
  coherence-selected and validation-selected K when they disagree (for NMF: coherence
  picks K=8 among healthy fits, validation picks K=6).
- Selected: **LDA K=4** and **NMF K=6**.
- Recovery of the known hazard types is **weak**: purity 0.716 vs a 0.690 majority-class
  baseline; NMI 0.10–0.12 against a shuffle null of 0.006–0.014.
- But the **lift analysis is the real result**: both models isolate a coherent
  allergen/gluten topic — NMF topic 1 (`gluten, celiac, cross contamination, gf`) has
  **lift 5.28** for `allergic_reaction`, LDA topic 0 has lift 4.18. What the models cannot
  do is subdivide the dominant `food_poisoning` mass (69% of the validated set).
  Report this as "the technique finds the rare, lexically-distinct hazard type and fails
  on the common, lexically-diffuse one" — that is the *why* the rubric rewards.

## Working notes
- Large data files are gitignored; raw Yelp JSON is not committed.
- Subsample for fast iteration (preserve class distribution); train/evaluate on full data
  only at decision points or for final reported numbers.
