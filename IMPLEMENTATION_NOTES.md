# Implementation Notes — Review Fixes (2026-07-26)

Maps each item in [PROFESSOR_REVIEW_V2.md](PROFESSOR_REVIEW_V2.md) to what changed.
Items 2.6, 2.7, 3.2 and 3.4 were explicitly deferred.

---

## 2.1 — Gold set was 89% training data → fresh out-of-sample gold set

Took the "better" route: labelled genuinely unseen data rather than re-splitting.

**Ran `labeling/build_holdout_pool.py --target 3000`** against the 5.3 GB raw review
dump. Output: `labeling/holdout_candidate_pool.csv`, **3,144 candidates**, verified
**zero text overlap** with `enriched_allergy_hazard_dataset.csv` after applying the
same normalisation the preprocessing notebook uses.

**Running `labeling/create_gold_dataset.py --source labeling/holdout_candidate_pool.csv
--n 800`** → `labeling/gold_dataset_holdout.csv`. This is rate-limited by the Groq free
tier and resumes cleanly if interrupted; re-run the same command to continue.

New in `config/settings.py`: `GOLD_INSIDE_PATH`, `GOLD_HOLDOUT_PATH`, `HOLDOUT_POOL_PATH`,
each documented with what it may and may not be used for.

New in `src/data_pipeline.py`:
- `load_gold_holdout()` — loads the fresh set, enriches it, sets `llm_is_hazard` as the target.
- `load_gold_inside(restrict_to_test_split=True)` — returns only the ~166 uncontaminated rows.
- `heuristic_vs_llm_agreement()` — label-quality metrics, valid on the full in-sample set
  because it compares two labels and never consults a model.

**A first result this already unlocks:** the fresh set has a **46.0% hazard rate within the
keyword-screened funnel** (355 of 772 rows, final labelled set — an early partial read on the
first 182 rows gave 42.3%). The 50/50-by-construction in-sample set could not measure this at
all. Caveat for the write-up: this is the rate *among keyword-flagged candidates*, not among all
Yelp reviews (plausibly 2–5%), since the pool is pre-filtered.

`src/features.py` is new and exists for this: the enrichment features were defined only
inside the postprocessing notebook, so fresh data could not be scored by the baseline.
`verify_against_training_data()` confirms all six derived columns reproduce the training
CSV exactly.

---

## 2.2 — `AsymmetricSafetyLoss` didn't do what it claimed → made it real

New `src/losses.py` with three selectable formulations sharing one interface:

| variant | positive-class weight | keyed on |
|---|---|---|
| `pos_weight` | `w` | the label (the original behaviour) |
| `focal_asymmetric` | `1 + (w-1)(1-p)^γ` | the error, smoothly |
| `fn_gated` | `w` if `p < τ` else `1` | the error, hard gate |

The original is kept as an explicit, honestly-named baseline rather than deleted — it is
the control the other two are measured against.

Verified in `tests/test_losses.py` (9 tests, all passing):
- `γ=0` reduces `focal_asymmetric` exactly to `pos_weight`
- `pos_weight` is numerically identical to `BCEWithLogitsLoss(pos_weight=w)`
- `w=1` is unweighted BCE for all three variants
- negatives are never upweighted by any variant
- a confident miss costs >100× a confident hit under the error-dependent variants, and
  that ratio is strictly larger than under `pos_weight`

Default is now `LOSS_VARIANT = "focal_asymmetric"` in `config/settings.py`.

**Still to reconcile in the write-up:** `ASYMMETRIC_WEIGHT = 50` versus the cost model's
$5,000:$50 = **100:1** ratio. These are different things — one is a training-time
regulariser, the other a deployment-time cost — and `analysis/evaluation_pipeline.py`
now says so in a comment, but the slides should state it explicitly.

---

## 2.3 — Recall-only selection → F2 and PR-AUC, both, for different jobs

They answer different questions, so both are used rather than one being chosen:

- **`CHECKPOINT_METRIC = "f2"`** — picking an epoch should reflect the operating point
  actually deployed, so it is evaluated at `DECISION_THRESHOLD`.
- **`HPO_METRIC = "pr_auc"`** — threshold-free, so hyperparameter search stays
  independent of the 0.20 choice.

`compute_metrics` now returns `precision, recall, f1, f2, pr_auc, pred_positive_rate`
plus `precision_at_50 / recall_at_50 / f2_at_50`.

`pred_positive_rate` is the collapse detector: an all-positive model scores perfect
recall, and the smoke test confirmed the reporting catches it (flag rate 100% shown
alongside recall 100%).

`selection_disagreement()` records which epoch F2 picks versus which PR-AUC picks;
`main.py` does the same across trials and writes `results/optuna_trials.csv`. If they
disagree, that is a reportable finding about metric choice, not a problem to hide.

---

## 2.4 — Invalid `problem_type` → labels popped before the forward pass

```python
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    loss = self.loss_fn(outputs.logits, labels)
    return (loss, outputs) if return_outputs else loss
```

`problem_type="binary_classification"` is gone; the model's internal loss branch is now
unreachable by construction rather than by accident. `**kwargs` absorbs
`num_items_in_batch`, added to this signature in transformers 4.46.

Also made version-tolerant, since the pinned 4.40 and the installed 4.57 differ:
`_build_training_args()` picks `eval_strategy` vs `evaluation_strategy` by inspecting the
signature, and `_trainer_tokenizer_kwarg()` picks `processing_class` vs `tokenizer`.
Verified working end to end on transformers 4.57.

---

## 2.5 — Unfair baseline comparison → class-balanced, and PR-AUC leads

`build_baseline_pipeline(scale_pos_weight=...)`; `train_and_evaluate_baseline` computes
the empirical negative/positive ratio and prints it. The dead comment is gone.

`analysis/evaluation_pipeline.py` now puts **PR-AUC first in every table** — it is
threshold-free and therefore not distorted by the two models' different weighting schemes.

Also fixed while in there: `predict_proba` was being passed the full dataframe instead of
the fitted column subset, and `cm.ravel()` would crash on a degenerate single-class
prediction (now `labels=[0,1]`).

---

## 3.1 — Second course technique → topic modeling (`src/topic_model.py`)

LDA **and** NMF, swept over K ∈ {2,3,4,5,6,8,10}, coherence by in-corpus NPMI
(implemented directly — gensim is heavy and brittle against numpy 2).

Fit on all 1,500 heuristic-flagged hazard reviews; validated on the 545 that carry an
LLM `hazard_type`. Fitting on the larger set is legitimate because fitting never sees labels.

**Results (`results/topic_model_*`):**

- Coherence rises monotonically with K, so selecting on it alone just picks the largest K.
  NMF at K∈{2,3,4} is degenerate (one topic holds 84–95% of docs). The code excludes
  degenerate fits and reports coherence-selected and validation-selected K separately.
- Selected **LDA K=4**, **NMF K=6**.
- Recovery of known types is **weak and honestly reported**: purity **0.697** (NMF K=6) against
  a **0.690 majority-class baseline** — i.e. essentially the baseline — with NMI 0.09–0.12
  against a shuffle null of 0.006–0.014. Both baselines are computed and printed so the numbers
  cannot be over-read. *(The earlier 0.716 came from a run whose artifacts no longer exist; it is
  void — quote 0.697 from the committed sweep.)*
- **The lift analysis is the actual finding.** Both models isolate a clean allergen topic —
  **lead with NMF**, which reproduces exactly (deterministic `nndsvda` init); the LDA row is
  environment-sensitive and must be quoted only from the committed artifact:

  | model | topic | top words | best-lift type | lift |
  |---|---|---|---|---|
  | NMF K=6 | 1 | gluten, gluten free, celiac, contamination, cross contamination, gf | `allergic_reaction` | **5.28** (per-topic NPMI **+0.43**) |
  | LDA K=4 | 0 | free, gluten, asked, allergy, gluten free, allergic | `allergic_reaction` | **2.52** *(committed value; a second environment gave 4.18 — do not quote it)* |

  What neither can do is subdivide the dominant `food_poisoning` mass (69% of validated docs).

  The defensible claim: *the technique recovers the rare, lexically-distinct hazard type
  and fails on the common, lexically-diffuse one.* That is a "why it works / why it fails"
  result, which is the top rubric band.

---

## 3.3 — Error analysis → rule-based failure-mode taxonomy

New `analysis/error_analysis.py`. Rule-based rather than hand-coded so it covers every
error, is reproducible across retrains, and exposes its criteria for disagreement. Each
bucket carries a written explanation of *why* the technique fails there.

Runs against models **and** against the labelling rule itself (`analyze_label_disagreement()`),
which needs no trained model and works today.

**Result on the 201 over-flagged reviews:**

| failure mode | count | share |
|---|---:|---:|
| `illness_mentioned_not_caused_here` | 96 | 48% |
| `unexplained_fp` | 43 | 21% |
| `negated_hazard` | 25 | 12% |
| `neutral_allergen_mention` | 23 | 11% |
| others | 14 | 7% |

Nearly half of the labelling rule's false hazards are reviews mentioning illness with no
causal link to the meal — "drove across town to get food for a sick friend". A keyword
rule cannot represent causation, only co-occurrence, so these are guaranteed
mislabellings. The 21% unexplained residual is left visible and flagged for manual review
rather than forced into a bucket.

**A hypothesis that was tested and rejected:** the keyword rule does *not* have a
structural blind spot for particular hazard types. Its per-type recall is 100%
(allergic_reaction), 99.7% (food_poisoning), 93.1% (unsafe_handling), 88.5%
(contamination). The rule's problem is **precision (73.2%)**, not coverage. Note the
recall figure is circular — measured inside an already keyword-filtered dataset — which
is a further argument for the holdout set.

---

## What still needs a training run

Everything above is implemented and smoke-tested, but the DeBERTa numbers are not final
because this machine has no GPU. On the training machine:

```bash
python main.py                          # sweep, objective = PR-AUC
python grid_search_analysis.py --quick  # loss-variant comparison (2x4 grid)
python analysis.py                      # final run, both ground truths, all artifacts
```

`analysis.py` reads `results/best_hyperparameters.json` if the sweep has run, and falls
back to the previous hardcoded values otherwise.

**Smoke-tested end to end** with a tiny transformer on a 400-row subsample: training,
custom loss, metrics, checkpoint selection, both evaluations, error analysis and all
plots run clean. In that run the baseline scored PR-AUC 0.986 against the heuristic label but
0.717 against the gold label — a −0.27 collapse.

⚠️ **Those are smoke-test figures — do not quote them.** The full run has since completed and
the real numbers are in `results/performance_gold_llm_label_fresh_holdout.csv`: the baseline
scores PR-AUC **0.979 vs the heuristic label and 0.728 vs gold (−0.251)**, DeBERTa **0.987 vs
0.804 (−0.183)**. The smoke test predicted the shape correctly, which is worth one sentence —
but the reported quantity is the committed one.
