# Food Safety Compass — Revalidated Review (2026-07-25)

*Second pass over the repo, re-checking every claim in [PROFESSOR_REVIEW.md](PROFESSOR_REVIEW.md) against the code as it stands today. 8 days to submission (2026-08-02 midnight); presentations 2026-08-03.*

## Bottom line

**Roughly 72–77/100 if submitted today** — a small improvement over the first review's 70–75, earned by the finished gold set and the leakage fix in `config/settings.py`. The trajectory to 85–90 is unchanged and still reachable, but one of the things the first review counted as *solved* is not, and it is the same failure mode as before wearing a different hat.

The headline: **the gold set you built to escape label leakage is 89% training data.** 1,334 of its 1,500 rows sit in the train split. Evaluating on it as-is would replace "measuring how well we learned the labeling rule" with "measuring how well we memorized the training set" — a worse problem, not a better one. The fix is cheap (a re-split, or running the holdout path you already wrote), but it must happen before any number goes in the slides.

---

## Part 1 — What the first review got wrong / what is now stale

### 1.1 ✅ The gold set is finished (review said "unfinished plumbing")

`labeling/gold_dataset.csv` has **1,500 fully labeled rows**, judged by an independent LLM with `llm_is_hazard`, `llm_hazard_type`, `llm_confidence`, and a free-text rationale. Confidence is 1,485 high / 12 medium / 3 low.

Agreement with the heuristic label is **85.8%**, and the disagreement is strongly one-sided:

| | LLM says benign | LLM says hazard |
|---|---|---|
| **heuristic = 0** | 738 | 12 |
| **heuristic = 1** | **201** | 549 |

The heuristic **over-flags**: it produces ~27% false hazards (201/750) and almost no false benigns (12/750). That asymmetry is a real, quantified finding about your own ground truth and it is the single most presentable thing in the repo. Lead with it.

### 1.2 ✅ Label-leakage features are already excluded (review and `CLAUDE.md` both stale)

`config/settings.py` now drops `stars`, `medical_lexicon_density`, and `negation_window_flag` from `TABULAR_FEATURES`, with a documented rationale — all three derive from the same keyword+stars rule that builds the label. Only `vader_neg_intensity` is kept, correctly justified as an independent sentiment signal.

This is done and done well. **`CLAUDE.md` lines 71–76 still describe this as unaddressed — update it**, or a grader reading your own notes will believe the stale version.

### 1.3 ✅ Variant experiments partially exist

`grid_search_analysis.py` sweeps the FN penalty weight across 7 values (1, 3, 5, 10, 15, 25, 50) with lr/batch/epochs held fixed. That is a legitimate variant experiment and the first review missed it. It is not *saved* anywhere, though — see 3.4.

---

## Part 2 — New findings the first review missed

### 2.1 🔴 CRITICAL — the gold set cannot be used for evaluation as built

`labeling/create_gold_dataset.py` defaults to `--source enriched`, which takes a 50/50 stratified sample **from the labeled dataset you train on** (750 heuristic-positive, 750 heuristic-negative). Cross-referencing `source_index` against the `random_state=42` split:

```
gold rows in TRAIN split: 1,334   (89%)
gold rows in TEST  split:   166   (11%)
```

So "re-run evaluation against the gold labels" — action item #1 in the first review — would report numbers on data the model was fit on. 166 clean rows is too few to carry a Results section (≈60 hazards; a single misclassification moves recall by ~1.7 points).

You already wrote the correct alternative and never ran it. `labeling/build_holdout_pool.py` pulls candidates from the raw Yelp dump, applies the *exact same* text normalization as preprocessing, and excludes anything already in the enriched CSV — a genuine never-seen pool. Neither `holdout_candidate_pool.csv` nor `gold_dataset_holdout.csv` exists on disk.

**Two ways out. Pick based on time:**

- **Fast (hours, no new API calls) — re-split around the gold set.** Force all 1,500 gold rows into the test split and train on the remaining 6,000. You get a 1,500-row gold-labeled test set for free and lose 1,334 training rows. Requires a `load_and_split_data` variant that splits on `source_index` membership rather than randomly.
- **Better (needs a `build_holdout_pool.py` run + ~500–800 LLM calls) — label fresh holdout data.** Keeps the current split intact and gives you a truly out-of-sample gold set. Also lets you report the hazard base rate on unseen data, which the current 50/50-by-construction gold set cannot tell you.

The fast option is the safe call at 8 days out. If you have API budget, do both and report them side by side.

### 2.2 🟠 `AsymmetricSafetyLoss` is not what its comment claims

```python
weight_mask = torch.ones_like(targets)
weight_mask[targets == 1.0] = self.weight_fn   # comment says "penalty ... specifically to False Negatives"
```

The mask keys on the **label**, not on the error. Every positive example is upweighted 50× whether the model got it right or wrong. This is mathematically identical to `nn.BCEWithLogitsLoss(pos_weight=50)` — textbook class weighting, not a custom FN-targeted loss.

It is a defensible design choice; it is just not the novel contribution the code comments, `CLAUDE.md`, and (presumably) your slides describe. Presenting standard class weighting as a bespoke safety loss is exactly the kind of claim a reviewer probes in Q&A, and being caught on it costs more than the technique earned.

**Two honest routes**, either is fine:
- **Reframe.** Call it cost-sensitive class weighting derived from the FN:FP cost ratio in your business model ($5,000 vs $50 = 100:1 — note your weight of 50 doesn't match your own stated ratio; reconciling those two numbers is a genuinely good slide).
- **Make it real.** Weight by the error, so hard positives dominate: multiply by `(1 - p)^γ` focal-style, or apply the penalty only where `sigmoid(logit) < threshold`. Then the name is accurate and you have a second variant to compare against plain `pos_weight` — which is itself a rubric-scoring experiment.

### 2.3 🟠 The degenerate all-positive model is confirmed, not hypothetical

The first review flagged recall-only selection as a theoretical risk. Your own code confirms it happened — `grid_search_analysis.py` lines 12–15:

> *"cfg.ASYMMETRIC_WEIGHT (50.0) ... appears to push every row toward 'hazard' (100% recall, 37.5% precision)"*

And `cfg.ASYMMETRIC_WEIGHT` is still `50.0`, `metric_for_best_model` is still `"recall"` (`sota_model.py:123`), and the Optuna objective is still `eval_recall` (`main.py:59`). So the model-selection loop is currently configured to *prefer* the degenerate model, and you have written evidence that it produces one. Both lines must change to F2 or PR-AUC. This is the cheapest high-value fix in the repo.

### 2.4 🟡 `problem_type="binary_classification"` is not a valid HuggingFace value

`sota_model.py:81`. The valid set is `regression`, `single_label_classification`, `multi_label_classification`. The invalid string means none of the model's internal loss branches match, so its built-in loss is silently skipped — which is what you wanted (FIX 1's comment), but by accident rather than by design, and it depends on the branch structure of the pinned `transformers==4.40.0`.

Clean fix: pop the labels before the forward pass so the model never tries to compute a loss.

```python
def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    loss = AsymmetricSafetyLoss(weight_fn=self.asymmetric_weight)(outputs.logits, labels)
    return (loss, outputs) if return_outputs else loss
```

### 2.5 🟡 Baseline vs SOTA at th=0.50 is not a fair comparison

`baseline_model.py:29–30` has a comment explaining `scale_pos_weight = neg/pos` — and then never sets it. So the baseline trains unweighted on a 4:1 imbalanced set while the SOTA trains with a 50× positive weight, and `evaluation_pipeline.py` compares them at the same 0.50 threshold. The baseline is handicapped by construction.

PR-AUC (already computed) is threshold-free and mostly immune to this, so **lead the comparison with PR-AUC**, and either set `scale_pos_weight` or drop the dead comment.

### 2.6 🟡 `requirements.txt` will not reproduce the project

Missing: `sentencepiece` (**hard requirement** for `AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)` — a fresh install crashes at tokenizer load), plus `matplotlib` and `seaborn`, both imported by `analysis/evaluation_pipeline.py`. A grader who clones and runs hits an error before seeing any output. One-line fix, disproportionate cost if missed.

### 2.7 🟢 Minor

- **Two entrypoints.** `main.py` (Optuna sweep) and `analysis.py` (full eval) duplicate setup, and `analysis.py` hardcodes `lr=1.814...e-05` from a past sweep with no record of where it came from. Add a one-line comment naming the trial, or a `README` note.
- **`llm_hazard_type` schema drift.** 2 rows say `cross-contamination` while 24 say `contamination`. Normalize before using these as topic-model ground truth.
- **Topic-modeling corpus is small.** Only 561 LLM-confirmed hazard docs, and the two smallest classes have 38 (`allergic_reaction`) and 24 (`contamination`). LDA will not cleanly separate categories that thin — expect to report "K=4 recovers food_poisoning and unsafe_handling but collapses the rare classes," and treat *that* as the finding rather than a failure.

---

## Part 3 — First review's claims that still stand, unchanged

### 3.1 🔴 Still only one course technique (the #1 grade risk)

Unchanged from the first review, and it is still worth more of the grade than everything else on this page. TF-IDF+XGBoost is a baseline, not one of the five course techniques; DeBERTa fine-tuning is technique #4, done well. **You need a second.** Topic modeling remains the right choice — it extends into hazard-*type* discovery, which is your stated second research question, and `llm_hazard_type` gives you ground truth to validate against.

### 3.2 🔴 Nothing persisted (`results/` does not exist)

Every number lives in stdout and a private W&B dashboard. The grader sees none of it.

### 3.3 🟠 Error analysis is 2 FPs + 2 FNs with no synthesis

`evaluation_pipeline.py:150–160`. Needs categorized failure modes tied back to *why* the technique failed.

### 3.4 🟠 No written problem framing; `README.md` is one line; no slides

---

## Part 4 — What to do next, in order

Ordered by grade-points-per-hour. Items 1–3 are half-day tasks that de-risk everything downstream; do them first even though they feel small.

| # | Task | Why now | Target |
|---|---|---|---|
| 1 | Change `metric_for_best_model` → `f2` and Optuna objective → F2 or PR-AUC (`sota_model.py:123`, `main.py:59`). Add an F2 to `compute_metrics`. | Two lines. Stops model selection from preferring the degenerate all-positive model you've already observed. Everything you train after this is trustworthy; everything before it isn't. | 07-26 |
| 2 | Add `sentencepiece`, `matplotlib`, `seaborn` to `requirements.txt`. | One line. Without it the grader can't run your code at all. | 07-26 |
| 3 | Fix the gold-set split (§2.1). Fast route: split on `source_index` so all 1,500 gold rows land in test. | Unblocks every number in your Results section. Do not train the final models before this. | 07-26 |
| 4 | Topic modeling (LDA + NMF) on the 561 confirmed hazard docs. ≥2 values of K, chosen by coherence. Cross-tab discovered topics against `llm_hazard_type`. | The second course technique — the largest single block of unearned points. Both the "≥2 techniques" and "variant experiments" rubric lines in one move. | 07-29 |
| 5 | Retrain baseline + DeBERTa on the new split; run `run_production_evaluation` twice — once with heuristic labels as `y_true`, once with gold labels — and report both. | The heuristic-vs-gold delta *is* your error analysis. It quantifies how much the 85.8% agreement rate was inflating your numbers. | 07-30 |
| 6 | Reframe or rebuild `AsymmetricSafetyLoss` (§2.2), and reconcile weight=50 against your own 100:1 cost ratio. | Removes the most likely Q&A trap and, if you build the focal variant, adds another comparison. | 07-30 |
| 7 | Categorized error analysis: bucket 20–30 FP/FN by cause (sarcasm, "gluten-free menu" false trigger, mishandled negation, mixed sentiment), with counts per bucket and a sentence on *why* each technique fails there. | This is the explicit 80–89 → 90–100 discriminator in Appendix B. | 07-31 |
| 8 | Commit `results/`: `model_performance_profile.csv`, `grid_search_asymmetric_weight.csv`, the 3 figures, plus `RESULTS.md`. Rewrite `README.md` as the one-page problem framing. Update the stale sections of `CLAUDE.md`. | Cheap points in two rubric categories. | 07-31 |
| 9 | Slides (≤15 min). Structure: problem + cost framing → two techniques → results (heuristic vs gold) → **the label-leakage story as the closing conclusion**. | 15% of the grade, currently at zero. | 08-01 |

**Cut line.** If you fall behind, items 1, 2, 3, 4, 5 and 9 are non-negotiable — they cover the two biggest rubric blocks. Items 6 and 7 are the difference between the low 80s and the high 80s. Item 8 is an hour and should never be what you drop.

**The story to tell.** Your strongest asset is not the model, it is the methodology arc: built a heuristic label → suspected it → built independent tooling to test it → measured 85.8% agreement and found a 27% over-flagging bias → discovered the validation set itself was contaminated → fixed that too → and only then reported numbers. Most course projects never question their ground truth once. Make that the spine of the presentation, and the two techniques the evidence.
