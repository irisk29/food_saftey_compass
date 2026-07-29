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
  - [src/embedding_model.py](src/embedding_model.py) — document-embedding classifiers over four
    text-only representations (TF-IDF control, LSA, Doc2Vec PV-DBOW, frozen MiniLM) behind one
    class-balanced logistic head, scored on both ground truths through the *existing*
    `score_variant` / `analyze_errors` code path. This is the **third course technique**.
  - [analysis/error_analysis.py](analysis/error_analysis.py) — rule-based failure-mode
    taxonomy applied to both model errors and the labelling rule itself.
  - [tests/test_losses.py](tests/test_losses.py) — 9 tests; notably a regression guard
    asserting an all-positive model is caught by F2/PR-AUC/flag-rate.
  - [tests/test_embedding_model.py](tests/test_embedding_model.py) — 21 tests; notably a
    cross-*process* determinism guard (an in-process one passed while a real gensim defect was
    live), class-balance parity with the baseline, and a text-only guarantee.

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
  `tests/test_losses.py`) — that reason is sound a priori and stands on its own. **As of
  2026-07-28 it is also measured.** See the Optuna section below: `results/optuna_trials.csv`
  trial 1 collapsed to all-positive, and bare recall would have ranked it *first of three*.
  The two collapse facts are separate and must not be conflated:
  - **The loss weight does not cause collapse.** The 8-cell grid positively refutes it —
    nothing degenerates at w=50 in either formulation (`pos_weight@50` flag rate 0.1967,
    `focal_asymmetric@50` 0.1858, `collapsed=False` in all 8 rows).
  - **The optimiser can.** At lr=3.80e-05 with batch_size=4 (loss held at
    `focal_asymmetric@50`) the model went to flag-rate 1.000. Collapse is a high-learning-rate /
    tiny-batch instability, not a property of the asymmetric loss.
  The old instruction "do not quote the collapse as a measurement" is now **retired** — quote
  `results/optuna_trials.csv` trial 1, and attribute it to the learning rate, not the weight.
- The heuristic's 97.9% recall is measured *inside* an already keyword-filtered dataset, so it
  is circular and optimistic. The holdout set gives the honest estimate.

## Step-4 modeling direction (the main deliverable)
Build a comparison of text-mining approaches, not a single classifier:
1. **TF-IDF + classifier** — ✅ `src/baseline_model.py`. Now class-balanced via
   `scale_pos_weight` so the comparison against the weighted transformer is fair.
   Counted as a *baseline*, not as one of the five course techniques.
2. **Document embeddings** — ✅ `src/embedding_model.py` (added 2026-07-29; this item previously
   read "not implemented; deliberately skipped"). Four text-only representations behind one
   class-balanced logistic head: TF-IDF (sparse control), LSA 300d, Doc2Vec PV-DBOW 300d trained
   on our own corpus, and frozen `all-MiniLM-L6-v2` 384d. Course technique **#3**. Completes a
   lexical → frozen-embedding → fine-tuned-contextual progression. See the findings section above
   — the result is a **negative** one and is the better for it.
3. **Fine-tuned transformer** — ✅ DeBERTa-v3-base (`src/sota_model.py`), asymmetric loss
   + lowered threshold, Optuna-tuned on PR-AUC. Course technique #1.
4. **Topic modeling (LDA/NMF)** — ✅ `src/topic_model.py`. Course technique #2.

That is **3 of the 5** named course techniques. Still unimplemented: word embeddings as a
standalone representation (partly subsumed — PV-DBOW and MiniLM are both word-level models
aggregated to documents) and generative LMs (an LLM judge *is* used, but for labelling, not as
the classifier).

### Hyperparameter sweep findings (results/optuna_trials.csv, results/best_hyperparameters.json)
Committed **2026-07-28** (`441c436`). This closes the "no hyperparameter provenance" gap that
every review from V3 onward had flagged, and it delivered one result worth more than the
provenance itself.

3 Optuna trials, objective = **validation PR-AUC**, 4 epochs, loss held fixed at
`focal_asymmetric @ w=50`, search space lr ∈ [1e-5, 5e-5] (log), batch ∈ {4, 8, 16}:

| Trial | lr | batch | val PR-AUC | F2 | recall | precision | flag rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.835e-05 | 16 | **0.9877** | 0.9163 | 0.9125 | 0.9319 | 0.196 |
| 2 | 1.291e-05 | 8 | 0.9876 | **0.9352** | 0.9375 | 0.9259 | 0.203 |
| 1 | 3.798e-05 | 4 | **0.1960** | 0.5556 | **1.0000** | 0.2000 | **1.000** |

- 🔴 **The headline: trial 1 collapsed, and bare recall would have selected it.** Flag rate
  1.000, recall 1.000, precision 0.200 (exactly the validation base rate), PR-AUC 0.196 — a
  textbook all-positive degenerate model. Be precise about what did not recover: the *hard
  predictions* stayed all-positive at every epoch (F2 pinned at 0.5556, the analytic all-positive
  value at a 20% base rate), while the *ranking* partially recovered — best-epoch PR-AUC 0.643
  against 0.988 for the two stable trials. Last either way. Ranking the three trials by each candidate
  objective: **`recall` picks the collapsed trial 1**; `pr_auc` picks trial 0 and puts trial 1
  last (0.196 vs 0.988); `f2` and `f1` pick trial 2 and also reject trial 1. The decision to
  drop bare recall as a selection metric — argued a priori since V3 and regression-guarded in
  `tests/test_losses.py` — now has a **committed artifact showing the failure it prevents**.
  `main.py` prints this comparison (including recall) and a collapse alarm on every sweep.
- **Attribution — and the exact limit of it.** Only the trial at the highest lr (3.8e-05) with the
  smallest batch (4) degenerated, and the grid runs w=50 in both formulations at the tuned lr with
  `collapsed=False` in all 8 rows. So **the weight is not *sufficient* to cause collapse** — that
  much is established, and it means you must never say "w=50 collapses". But do **not** over-claim
  the converse: every sweep trial used `focal_asymmetric@50`, so there is no within-sweep contrast
  on the loss, and no low-weight/high-lr cell was ever run — the grid is not a clean control
  because it differs in lr (1.814e-05), effective batch (16) and epochs (3) simultaneously. A
  **learning-rate × weight interaction is not excluded**, and is mechanistically plausible (a ×50
  positive-class weight amplifies gradients precisely where a high lr with batch 4 is already
  unstable). Correct phrasing: *"collapse required the high learning rate; whether the weight
  contributed we did not isolate."* One extra cell (`pos_weight@1` at lr 3.8e-05, bs 4) would
  settle it in ~20 min on the MPS box.
- **The two selection metrics disagree, within and across runs.** `metrics_agree = False` in all
  three trials (F2 and PR-AUC pick different epochs), and across trials PR-AUC picks trial 0
  while F2 picks trial 2. Both reject the degenerate model; they differ only among the healthy
  ones, and the gap between trial 0 and trial 2 on PR-AUC is 0.00005 — noise. Report the
  disagreement as evidence the choice of metric was deliberate, not as a defect.
- ⚠️ **Provenance caveat — the sweep's best lr is NOT the lr that produced the reported system.**
  `best_hyperparameters.json` names lr **1.8346e-05**; every committed artifact in
  `results/performance_*`, `results/error_analysis_*` and the figures was trained at
  **1.8140e-05** (from an earlier sweep whose trials were not persisted). A 1.1% relative
  difference, far inside the measured 0.054 gold-PR-AUC noise floor — but `analysis.py` prefers
  the JSON, so a re-run would *not* reproduce the committed CSVs. `analysis.py` now prints a
  PROVENANCE WARNING on mismatch, and `FSC_USE_REPORTED_HPARAMS=1` pins the as-reported values.
  Batch size 16 matches. Disclose this in one sentence; do not silently re-run.
- Only **3 trials** — small, and the sweep varied lr and batch size only (not epochs, weight
  decay or warmup). Say "3-trial sweep", never "tuned".

### Loss-variant grid findings (results/grid_search_loss_variants.csv)
The signature contribution is the `AsymmetricSafetyLoss`, so it needs a head-to-head. As of
**2026-07-28** the committed grid is complete: **8 rows** — `{pos_weight, focal_asymmetric}` x
`{w=1, 5, 15, 50}` — every one re-trained from scratch under `--force`, **all at 3 epochs**
(900 optimiser steps, verified from `results/grid_search.log`), and **all scored on both ground
truths**. The earlier mixed 2-/3-epoch table is superseded and its numbers are void.
- **The heuristic metric is saturated and, worse, uninformative.** Validation PR-AUC spans
  0.9864–0.9894 across all eight — a spread of **0.0030**. On the gold holdout the same eight
  span 0.7219–0.8211 — a spread of **0.0992, 33× larger**. The decisive statistic is the rank
  correlation between the two ground truths: **Spearman ρ = −0.24** (p = 0.57, n = 8;
  Pearson −0.36). ⚠️ State this carefully — with n=8 the correlation is **not significant**, so the
  defensible claim is "**no detectable relationship** between the two orderings", not "it carries
  no information" (absence of evidence is not evidence of absence, and the 95% CI on ρ is roughly
  −0.85 to +0.55). The claim that carries the weight is the **spread ratio**: 0.0030 of resolution
  is not enough to select on regardless of the correlation. Lead with 33×, use ρ as support. This
  is still the strongest methodological result in the project.
- **Run-to-run noise is large and must be quoted.** Four configurations have now been trained
  twice under nominally identical settings. Three reproduced to within 0.006 gold PR-AUC; one
  (`pos_weight@5`) moved **0.054** — over half the entire between-configuration spread. Both of
  its runs selected the epoch-2 checkpoint, so this is not a selection flip; the trajectories
  diverged by epoch 1 and the cause is unattributed (most likely non-deterministic MPS kernels).
  **Treat gold gaps below ~0.05 as unresolved.** `seed=cfg.RANDOM_STATE` is now passed to
  `TrainingArguments` explicitly (added 2026-07-28); it equals HuggingFace's own default of 42, so
  this changed no committed number and every run in the project is at seed 42. The 0.054 is
  therefore non-determinism at a *fixed* seed, and a lower bound on true seed variance.
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

### Document-embedding findings (results/performance_embedding_*, results/embedding_*)
Added **2026-07-29** by `src/embedding_model.py`, on the CPU-only Windows box (no CUDA, no MPS —
verified, not assumed). This closes the largest remaining gap against the brief: the project
claimed 2 of the 5 named techniques and had **explicitly declined the embedding family** (see the
struck-through item 2 in the step-4 plan below). It is **strictly additive** — it writes only
`results/*embedding*` files, reads the committed `performance_*.csv` for the baseline and DeBERTa
rows rather than recomputing them, and `analysis.py` / the loss grid / the topic model were not
re-run.

**Design.** Four *text-only* representations behind one identical class-balanced
`LogisticRegression` (C=1.0, `class_weight="balanced"`, which reproduces the baseline's
`scale_pos_weight = n_neg/n_pos` exactly — pinned by a test), fitted on the same 64% train split
the reported baseline and DeBERTa saw, scored with the **same `score_variant`, thresholds and
cost model** imported from `analysis/evaluation_pipeline.py`, and error-bucketed with the same
`analyze_errors` taxonomy. Text-only is deliberate: it is what makes the comparison against
text-only DeBERTa legitimate, and the handicap runs *against* the embedding model, since the
baseline additionally gets all seven `TABULAR_FEATURES`.

| Model | heuristic test PR-AUC | gold PR-AUC | gold 95% CI | delta | gold F2@0.20 | gold flag rate | risk cost@0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeBERTa-v3 (committed) | 0.9874 | **0.8045** | — | −0.183 | 0.817 | 0.665 | $204,850 |
| LSA 300d + LogReg | 0.9262 | 0.7574 | 0.724–0.793 | −0.169 | 0.751 | 0.723 | $303,050 |
| XGBoost baseline (committed) | 0.9785 | 0.7279 | — | −0.251 | 0.677 | 0.447 | $585,300 |
| TF-IDF + LogReg (text-only control) | 0.9238 | 0.7112 | 0.667–0.754 | −0.213 | 0.768 | 0.582 | $348,100 |
| MiniLM-L6-v2 frozen 384d + LogReg | 0.8577 | 0.7092 | 0.666–0.750 | −0.148 | 0.787 | **0.817** | **$175,400** |
| Doc2Vec PV-DBOW 300d + LogReg | 0.7806 | 0.6831 | 0.639–0.732 | −0.098 | 0.671 | 0.534 | $553,300 |

- 🔴 **The headline is a negative result, and it is worth more than the win we expected.** The
  prior was "a frozen embedding beats or matches TF-IDF on gold while losing to DeBERTa". Half of
  that is wrong. **No frozen embedding is distinguishable from TF-IDF on the gold holdout**, and
  the only variant above the XGBoost baseline is **LSA — a linear rotation of the very same
  TF-IDF matrix** — by 0.029, comfortably inside the bootstrap CI. A *frozen transformer encoder*
  lands at 0.709, level with a bag of words at 0.711, while the *fine-tuned* one reaches 0.804.
  The defensible claim: **dense distributed representation is not the source of the transformer's
  advantage.**
  ⚠️ Do **not** upgrade this to "the advantage is fine-tuning". MiniLM-L6-v2 is 8× smaller than
  DeBERTa-v3-base (6 layers / 22M vs 12 / 184M) and was pretrained on a sentence-similarity
  objective, so freezing, capacity and pretraining objective vary together. This is **not** a
  clean ablation, and one honest sentence says so. Isolating it needs a frozen DeBERTa-v3-base
  encoder + linear head, which is one cheap CPU run (feature extraction only, no backprop) and is
  the single most valuable follow-up available.
- 🔴 **The ground-truth delta orders the six models by how little each can memorise the labelling
  rule** — Doc2Vec −0.098, MiniLM −0.148, LSA −0.169, DeBERTa −0.183, TF-IDF+LogReg −0.213,
  XGBoost −0.251. The mechanism is direct: the heuristic label *is* a keyword rule, so a
  representation that preserves individual token identity can reproduce it, and one that
  compresses 2,500 features into 300 dimensions or was never fitted on our vocabulary at all
  cannot. ⚠️ **The delta is partly mechanical** — a lower heuristic score has less room to fall —
  so do not quote the ordering alone. Quote the pair that has no such artifact: **TF-IDF and
  frozen MiniLM reach statistically indistinguishable gold PR-AUC (0.7112 vs 0.7092, each inside
  the other's CI) from heuristic scores 0.066 apart.** The heuristic metric pays TF-IDF 0.066 of
  PR-AUC for skill worth *exactly nothing* out of sample. That is the project's central thesis
  reproduced on a new axis, with a new model family, and it is the strongest sentence this
  section produces.
- **Doc2Vec is the worst representation in the project, and the reason is data volume, not
  domain.** 4,800 training documents is one to three orders of magnitude below what paragraph
  vectors need. The controlled comparison rules out the alternative explanation: **LSA is fitted
  on the identical 4,800 documents** and beats it by 0.074 on gold. So a *learned* embedding
  loses to a *linear projection* of the same data at this scale — the learning had too little
  data to pay for itself. Report this as a "when does the technique fail" finding, which is the
  same move the topic-model section makes.
- 🔴 **Every model inherits the labelling rule's blind spots, and the fine-tuned one inherits
  them least** (`results/embedding_vs_deberta_fp_modes.csv`). Share of gold false positives
  sitting in the rule's own top two failure modes (`illness_mentioned_not_caused_here` +
  `neutral_allergen_mention`): **DeBERTa 64.5%** (the figure the error-analysis section rounds to
  65%), TF-IDF 69.1%, frozen MiniLM 71.8%, LSA 75.9%, Doc2Vec 76.5%. This **upgrades the existing
  "the model inherits its teacher's blind spots" finding from an observation about one model to a
  dose–response across five**, in which the only model that partially escapes is the only one
  allowed to update its representation on the task.
- **And the dense representations make one of those two modes distinctly worse, for a knowable
  reason.** `neutral_allergen_mention` is 45.1% of TF-IDF's gold FPs but 62.1% of LSA's, 57.2% of
  Doc2Vec's and 52.6% of MiniLM's. In a dense space "they have a great gluten-free menu" and "the
  gluten made me ill" are near neighbours — the geometry actively destroys a distinction the
  sparse representation at least retains as separate coordinates. This is the mechanistic *why*
  behind the embedding models' low precision (0.51–0.64 on gold), and it is the same phenomenon
  `analysis/error_analysis.py` already names as "the labelling rule leaking into the model",
  amplified by the representation rather than caused by it.
- ⚠️ **Frozen MiniLM posts the lowest total risk cost in the entire project ($175,400 vs
  DeBERTa's $204,850) and this must NOT be reported as a win.** It flags **81.7%** of a holdout
  whose funnel hazard rate is 46.0%. This is the identical pathology the loss grid already
  documented for `focal_asymmetric@5` (75.0% flag rate crowning the cost and F2 columns): under a
  100:1 cost ratio over-flagging is nearly free. Its high recall (0.910, *above* DeBERTa's 0.890,
  with 32 missed hazards against 39) is bought with precision 0.512. PR-AUC ranks it 0.095 below
  DeBERTa and PR-AUC is the metric to believe. That the new model family reproduces this
  pitfall independently is corroboration of the project's own methodological finding.
- **The 0.20 threshold does not transfer here either.** Gold flag rates run 0.534–0.817 against a
  46.0% funnel rate, while the same models flag 0.214–0.335 on the test split. Third independent
  instance of the same thesis.
- **Reproducibility: exact, and it took three fixes to gensim to get there.** Every variant is now
  bit-identical **across processes** at a fixed seed (guarded by
  `test_doc2vec_is_identical_across_PROCESSES`). Three sources of non-determinism had to be
  pinned, and the third was found only by measurement after the first two looked sufficient:
  `workers=1`; a deterministic `hashfxn` on the model; and — the real defect —
  `Doc2Vec.infer_vector` initialises the new document vector via
  `pseudorandom_weak_vector(size, seed_string=...)`, whose signature is
  `(size, seed_string=None, hashfxn=hash)`. **It uses the module-default salted `hash` and ignores
  the model's `hashfxn` entirely**, so the constructor argument that makes *training*
  reproducible has no effect on *inference*. Symptom: the same fit in two interpreters produced
  bit-identical word vectors (md5 `1c0bd6c2…`) and different inferred vectors, with gold PR-AUC at
  0.6825 / 0.6833 / 0.6836. `Doc2VecRepresentation._deterministic_gensim_init()` scopes a patch
  over it. The wobble was 0.0011 — immaterial against the 0.0165 seed spread — so **no reported
  conclusion ever depended on it**; the point is that "reproducible" was false as stated until it
  was measured.
- **Doc2Vec seed spread, since exactness at one seed is not authority**
  (`results/embedding_doc2vec_dbow_seed_spread.csv`, seeds 42/43/44): gold PR-AUC 0.6831, 0.6684,
  0.6674 — mean 0.6730, sd 0.0088, **range 0.0157**. On the heuristic test split the range is
  0.0032. Note the shape: **the seed-to-seed range is 4.9× larger on gold than on the heuristic
  label**, the same direction as the loss grid's 33× spread ratio. Doc2Vec's deficit against LSA
  (0.074) is 4.7× its own seed range, so that gap is real; its deficit against TF-IDF (0.028) is
  under 2× it and is not resolved.
- **Sampling noise, which the project had never quantified**
  (`results/embedding_gold_pr_auc_bootstrap.csv`): a 1,000-replicate class-stratified bootstrap
  of the 772-row holdout gives 95% CIs **0.069–0.093 wide** (sd 0.018–0.023). This is a different
  quantity from the 0.054 training-noise floor and it applies to *every* gold number in the
  project, including the committed ones. It is the honest reason the LSA-over-baseline margin
  (0.029) and the LSA-over-TF-IDF margin (0.046) must both be called unresolved. ⚠️ It cannot be
  turned into a difference test against the baseline or DeBERTa: their probability arrays were
  never persisted to `results/` (only summary metrics were), so the paired resample is unavailable
  without re-running `analysis.py` and overwriting committed artifacts. **Persisting per-row
  probabilities is the cheapest rigour upgrade left in the repo.**
- **Two smaller true things.** (a) `src/topic_model.py`'s docstring says gensim "breaks against
  numpy 2", which is why its NPMI is hand-rolled — that was true of gensim 4.3.x and is **not**
  true of 4.4.0, which installs and runs here under Python 3.13 / numpy 2 / scipy 1.17. The topic
  model was left alone regardless, because its artifacts are committed and cited. (b) The
  enriched CSV holds **3 duplicate review texts** (7,497 unique of 7,500) and one duplicate pair
  **straddles the train/test boundary**, so 1 of 1,500 test-split rows shares text with a
  training row. Found by the embedding leakage guard. It affects the baseline and DeBERTa
  identically, bounds any test-split metric error at 1/1500 = 0.07%, and **the gold holdout
  remains at exactly zero overlap**, which is the number that matters and which
  `verify_no_text_overlap` re-checks on every run.

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
  majority-class baseline), NMF K=6 purity 0.697; NMI **0.084 (LDA K=4) and 0.119 (NMF K=6)**
  against shuffle nulls of **0.0054 and 0.0137** respectively. Quote the range as 0.08–0.12
  vs a null of 0.005–0.014 — not "0.09–0.12 / 0.006–0.014", which excluded LDA K=4.
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
