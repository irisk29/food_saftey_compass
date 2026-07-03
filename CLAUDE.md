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
    stratified train/test split (80/20, `random_state=42`).
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
  - Still **missing** from the step-4 plan: no word/document-embedding model (item 2:
    Word2Vec/Doc2Vec/Sentence-BERT) and no topic modeling (item 4: LDA/NMF for hazard-type
    discovery) — currently only 2 of the 4 planned approaches (TF-IDF+XGBoost, DeBERTa) exist.
  - No results/metrics have been captured back into the repo yet (no saved run outputs, error
    analysis, or comparison writeup) — `main.py` prints reports/logs to stdout/W&B but nothing
    is persisted to the repo for the writeup.

## Known risks / gaps (most important)
- **Label leakage risk is still unaddressed.** Label is a keyword+stars heuristic; the baseline
  model is fed the same lexicon-density/negation features used to construct that heuristic, so
  both the baseline and (via text) the DeBERTa model risk relearning the labeling rule rather
  than a generalizable hazard signal. No keyword-free validation set or hand-labeled gold set
  exists yet — still needed before results can be trusted for the writeup.
- The asymmetric loss + 0.20 threshold in `sota_model.py` optimize purely for recall; this is a
  defensible business framing (missing a hazard is worse than a false alarm) but should be
  stated explicitly and contrasted with precision in the error analysis, not just reported as
  "better recall."
- Word/document embeddings and topic modeling (step-4 items 2 & 4) are not implemented — needed
  to hit the "≥2 techniques + comparison" rubric band with real variety, and topic modeling in
  particular is the way to move past binary hazard detection into hazard-type discovery.

## Step-4 modeling direction (the main deliverable)
Build a comparison of text-mining approaches, not a single classifier:
1. **TF-IDF + linear classifier** — ✅ implemented (`src/baseline_model.py`, XGBoost not linear —
   consider also reporting a plain linear model e.g. logistic regression for a cleaner "what TF-IDF
   alone sees" comparison).
2. **Word/Document embeddings** (Word2Vec/Doc2Vec or Sentence-BERT) — ❌ not yet implemented.
3. **Fine-tuned transformer** (BERT/DistilBERT) — ✅ implemented as DeBERTa-v3-base
   (`src/sota_model.py`), with recall-biased custom loss + threshold, Optuna-tuned.
4. (Optional) **Topic modeling (LDA/NMF)** to characterize hazard *types* (allergy vs.
   food poisoning vs. choking) — ❌ not yet implemented.

For each: proper metrics (not just accuracy), multiple baselines, variant experiments,
and qualitative **error analysis** tied back to the food-safety business problem — none of
this synthesis/writeup exists yet.

## Working notes
- Large data files are gitignored; raw Yelp JSON is not committed.
- Subsample for fast iteration (preserve class distribution); train/evaluate on full data
  only at decision points or for final reported numbers.
