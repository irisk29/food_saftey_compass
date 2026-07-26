# Food Safety Compass — Third-Pass Review (2026-07-26)

*Deep re-check after the 2026-07-26 refactor wave. Every claim below was verified by four independent review passes — reading the code line-by-line, **running computations against the CSVs on disk** (contamination joins, loss-function numerics, NPMI recomputation, test suite execution), and grading against the assignment PDF. 7 days to submission (2026-08-02 midnight); presentations 2026-08-03.*

## Bottom line

**68–73/100 if submitted today — but this number is misleading in both directions.** It is *lower* than V2's 72–77 only because "today" now means submitting with zero slides and zero classifier results at 7 days out. The *underlying work* has improved dramatically: every code-level issue V2 flagged is genuinely fixed, the holdout gold set is real and verified contamination-free, and the analytical depth (label-quality analysis, error taxonomy, topic-model honesty) is already top-band material.

The project's situation in one sentence: **you have built a top-band evaluation apparatus and never turned it on.** Not a single model has been trained under the fixed configuration. `results/` contains no precision, no recall, no PR-AUC for either classifier, against either ground truth. Every "FIXED" verdict below is verified-by-code, not verified-by-execution — and two reproduction bugs (W&B hardcoded, requirements.txt broken) mean the first fresh run will crash before producing anything. Running `analysis.py` is the single blocker between here and 85–90.

One new genuine bug was found: **the NPMI coherence values in every persisted topic-model artifact are mathematically wrong** (§2.1). The fix is one line, and — unusually — the corrected numbers make your story *better*, not worse.

---

## Part 1 — V2 scorecard: what is now verified fixed

Each item below was checked against the code and, where possible, by execution.

### 1.1 ✅ V2 §2.1 (CRITICAL) — gold-set contamination: **FIXED, verified by computation**

`labeling/gold_dataset_holdout.csv` exists: **744 rows** (of the 800 targeted — the labeling run stopped early; it is resumable), 402 benign / 342 hazard (46.0% hazard rate), 739/744 high confidence.

The contamination check was **recomputed from scratch**, not taken on trust. Using the exact text normalization from `build_holdout_pool.py:78-81`:

| Check | Overlap |
|---|---|
| holdout (744) vs enriched dataset (7,500) — exact normalized text | **0** |
| same, case-insensitive | **0** |
| full candidate pool (3,144) vs enriched | **0** |
| holdout vs in-sample `gold_dataset.csv` (1,500) | **0** |

Since the enriched CSV is a strict superset of train+test, zero overlap with it implies zero overlap with train. Statistical adequacy: 342 positives give a 95% CI half-width on recall of ±3–4 points — usable for a Results section (vs. ±1.7 points *per single error* on the 166-row alternative V2 warned about).

Also verified: sampling is a plain seeded random draw from the pool (index sets match `pool.sample(n=744, random_state=42)` exactly — **not** 50/50 stratified), `src/features.py` reproduces all 6 training-time derived features to 6 decimals, `load_gold_holdout` runs end-to-end and produces every column both models need, and `verify_setup.py` re-runs the contamination check on every preflight. The in-sample gold set's rows-in-train situation is now coherently quarantined as "label-quality analysis only" (`data_pipeline.py:71-96`, `settings.py:17-25`).

**Residual caveat — see §2.3:** the 46% is a *keyword-funnel* base rate, not a Yelp base rate.

### 1.2 ✅ V2 §2.2 — AsymmetricSafetyLoss: **FIXED, math verified numerically**

`src/losses.py` has three variants, and the two new ones are genuinely error-dependent — confirmed by running them:

- `pos_weight` — honestly documented as label-keyed, ≡ `BCEWithLogitsLoss(pos_weight=w)`.
- `focal_asymmetric` — `w = 1 + (w_fn−1)(1−p)^γ`. Confident-correct positive gets weight ~1, confident miss ~w_fn. Gradients finite at the default γ=2.
- `fn_gated` — penalty only where `p < τ`, gate correctly `.detach()`ed, negatives never upweighted. Verified: confident correct positive loss equals plain BCE *exactly* (0.04858 == 0.04858); penalty 83.0 just below τ vs 1.56 just above.

Shape handling is safe (`view(-1)` on both logits and targets — `[N,1]` vs `[N]` verified identical), no double sigmoid, `τ` is tied to the deployed threshold. Default in use: `focal_asymmetric` (`settings.py:61`). The 9 tests in `tests/test_losses.py` **pass (9/9, executed)** and their assertions were independently re-derived — including the regression guard proving an all-positive model is caught by F2/PR-AUC/flag-rate.

**Gap:** the variant *comparison* is wired in `grid_search_analysis.py` but has never been run — `results/grid_search_loss_variants.csv` does not exist. The loss work is currently *claimed*, not *demonstrated*.

### 1.3 ✅ V2 §2.3 — degenerate-model selection: **FIXED**

`metric_for_best_model = "f2"` with `greater_is_better=True` (`sota_model.py:170-171`), F2 computed with `beta=2.0` correctly, Optuna objective is genuine threshold-free PR-AUC (`main.py:67`, `average_precision_score` on sigmoided probs). `pred_positive_rate` is logged as a collapse detector. Threshold use is consistent between training-time metrics and final eval (everything flows from `cfg.DECISION_THRESHOLD`; the old hardcoded-0.20 scatter is gone).

### 1.4 ✅ V2 §2.4 — invalid `problem_type` hack: **FIXED**

Labels are popped before the forward pass (`sota_model.py:44`), no `problem_type` set, `compute_loss(..., **kwargs)` absorbs the `num_items_in_batch` argument newer transformers pass. Works on both the pinned 4.40 and the actually-installed 4.57. (The stale pin itself is §2.2 below.)

### 1.5 ✅ V2 §2.5 — baseline fairness: **FIXED (one asymmetry left)**

`scale_pos_weight = n_neg/n_pos` is really computed from the train split and passed to XGBoost (`baseline_model.py:53-59`). Both models are scored on the same test/gold sets and PR-AUC leads every table. Residual nit: the baseline is tabulated only at th=0.50 while DeBERTa also gets its tuned 0.20, so the "Total Risk Cost" row hands DeBERTa a tuned threshold the baseline never gets. Defensible since PR-AUC leads; worth one footnote.

### 1.6 ✅ V2 §3.1 — second course technique: **DONE (topic modeling), with one bug — §2.1**

`src/topic_model.py` is genuinely course-grade: shared 3,000-term vocabulary across LDA/NMF (verified identical), domain stopwords + `max_df=0.5` to kill selection keywords, K∈{2,3,4,5,6,8,10} × 2 algorithms, degenerate-fit exclusion, purity vs majority baseline, NMI vs a correctly-seeded shuffle null, and lift with the correct denominator (spot-verified: 14/38 in an 88/545 topic → 4.183 ✓). **Every number in CLAUDE.md's findings section matches the persisted artifacts** — LDA K=4 / NMF K=6, purity 0.716 vs 0.690 baseline, NMI 0.10–0.12 vs null 0.006–0.014, allergen lift 5.28/4.18. Two caveats: the 0.716 purity holds for **LDA only** (NMF K=6 purity is 0.697, essentially at baseline), and the coherence column is invalid (§2.1).

### 1.7 ✅ V2 item 7 (half) — error analysis of the **labeling rule** is 90-band quality

`results/error_analysis_heuristic_label.md` buckets all 201 FPs + 12 FNs with counts, shares, examples, and per-bucket causal explanations (48% `illness_mentioned_not_caused_here`, 12% `negated_hazard`, …). The causation-vs-co-occurrence finding is genuinely top-band material. **The model half does not exist** — see §2.4.

### 1.8 ✅ Docs — CLAUDE.md is no longer stale

The "Known risks" section now accurately describes the leakage exclusions, the two-gold-sets distinction, the F2/PR-AUC switch, and the circular-recall caveat. RUNBOOK.md and IMPLEMENTATION_NOTES.md are accurate against the code (spot-checked). Minor number drift listed in §3.

---

## Part 2 — What is broken or missing now

### 2.1 🔴 HIGH (bug) — every persisted NPMI coherence value is wrong

`src/topic_model.py:84` computes document co-occurrence as `cols.T @ cols` on a **boolean** sparse matrix (`coherence_matrix = (X_count > 0)` at :180/:190). Boolean matrix multiplication saturates at `True`, so every co-occurring term pair gets count = 1 no matter how often it co-occurs. This was confirmed two ways: the buggy path exactly reproduces the persisted values, and the corrected path gives wildly different ones — every persisted coherence is negative (−0.24 to −0.47) while the correct values are mostly **positive** (e.g. NMF topic 1: persisted −0.045, correct +0.431).

**Fix is one line:** `coherence_matrix = (X_count > 0).astype(np.int32)` — then rerun the sweep.

**The corrected results improve your story.** With real counts, LDA coherence **peaks at K=4 — the same K external validation selects** — and does *not* rise monotonically. The current narrative in CLAUDE.md ("NPMI rises monotonically with K, so coherence-based selection is unreliable here") is an artifact of the bug and must be rewritten. The new narrative — "intrinsic coherence and extrinsic validation independently agree on K=4 for LDA; NMF's coherence-preferred K=2 is degenerate" — is stronger and more defensible in Q&A. `topic_model_sweep.csv`, `topic_model_topics.csv`, and the left panel of `topic_model_selection.png` all need regeneration.

### 2.2 🔴 CRITICAL (blocker) — no classifier has ever been trained under the fixed configuration

`results/` contains only topic-model and label-rule artifacts. There is **no persisted performance number for either model, against either ground truth**: no `performance_*.csv`, no `ground_truth_comparison.csv`, no PR/cost curves, no `optuna_trials.csv`, no `best_hyperparameters.json`, no loss-variant grid. The only post-fix training evidence is a 400-row tiny-transformer smoke test explicitly labeled non-reportable. Every real DeBERTa number ever produced predates the fixes and was selected by the metric that preferred the degenerate all-positive model.

**And the first fresh run will crash before producing anything**, for two reasons:

- **W&B is a hard dependency with no offline fallback.** `report_to="wandb"` is hardcoded (`sota_model.py:172`), and `analysis.py`/`grid_search_analysis.py` never init it — on a machine without `WANDB_API_KEY`, the run prompts interactively or errors. Fix: set `WANDB_MODE=offline` in the scripts' `__main__` blocks (they already set other env vars there) or `report_to=os.getenv("REPORT_TO", "none")`.
- **`requirements.txt` is still broken (V2 §2.6, unfixed — the one-line item was the only one skipped).** Missing: `sentencepiece` (hard crash at DeBERTa tokenizer load), `matplotlib`, `seaborn` (imported by the eval pipeline and topic model), and — new — `accelerate` (transformers' Trainer refuses to construct `TrainingArguments` without it; `verify_setup.py` doesn't check it either). Worse, the pins describe an environment nobody uses: `torch==2.2.2` has no wheels for the Python 3.13 this actually runs on, and `transformers` is pinned to 4.40.0 while the code was verified on 4.57. RUNBOOK step 1 literally instructs installing "the three packages requirements.txt is missing" instead of fixing the file.

Run `python analysis.py` **first thing** — if it surfaces a surprise (flag-rate collapse, gold-set score cliff), you need the week's slack to react.

### 2.3 🟠 The "46% hazard base rate on unseen data" claim needs a caveat

Computed: **100% of holdout rows pass the STRONG/WEAK keyword filter** used to build the candidate pool (`build_holdout_pool.py:63-75`), whose vocabulary substantially overlaps the heuristic label's lexicon, and whose WEAK tier reuses the identical stars≤3 gate. The hazard rate is 71.5% among STRONG-matched rows vs 21.4% among weak-only rows — the 46% figure is heavily conditional on the filter. The raw Yelp base rate is plausibly 2–5% (create_gold_dataset.py's own admission).

This does **not** invalidate the evaluation — it measures performance on the funnel a deployed system using the same pre-filter would see, which is the realistic deployment scenario. But `data_pipeline.py:66-67` and `verify_setup.py:104` currently print it as "the hazard base rate", and that phrasing will get picked apart in Q&A. Say "hazard rate within the keyword-screened funnel" everywhere. (One genuine improvement worth mentioning: STRONG terms are kept at *any* star rating — 73 holdout rows have 4–5 stars — so the holdout is not a pure replica of the heuristic's selection rule.)

### 2.4 🟠 No validation split — the test split does double duty as the selection set

`run_sota_training` uses the test split as `eval_dataset` with `load_best_model_at_end=True` (checkpoint picked by F2 *on test*), and the Optuna objective is PR-AUC *on the same test split*. The heuristic-label test-split numbers `evaluation_pipeline.py` will report are therefore selection-biased upward. **Mitigated** because your headline numbers come from the gold holdout, which selection never touches — that path is clean. But footnote the test-split table as "in-selection" and lead with the holdout, or a sharp grader lands this exact punch.

### 2.5 🟠 Model-error analysis exists as code only

`evaluate_one_set` calls `analyze_errors(label=f"deberta_{slug}")` (`evaluation_pipeline.py:128-131`), so running `analysis.py` produces it for free — but as persisted, the project has zero analysis of what the *models* get wrong, which is what V2 item 7 (and the 90-band rubric line) actually asked for. Also: the `NEGATED_HAZARD` regex omits `haven't/hasn't/hadn't` (`error_analysis.py:39`) — the first example shown under `illness_mentioned_not_caused_here` ("haven't gotten sick from") is really a negation case in the wrong bucket. Fix the regex before the model-error run.

### 2.6 🟠 README is still one line; slides are at zero

A grader opening the repo cold finds no document stating the problem, the cost model, or the results — the good framing lives in CLAUDE.md/IMPLEMENTATION_NOTES, which are internal process docs. And with presentations on 08-03, 15% of the grade currently has no artifact. The slide skeleton (problem → methodology arc → label-quality findings → topic model) needs **no pending numbers** and can be built today.

### 2.7 🟡 Topic-model write-up caveats (cheap to add, expensive to be caught omitting)

- **Selection circularity:** the winning K is chosen by `nmi_above_null` against `llm_hazard_type`, then the same NMI is reported as the validation result. Disclosed only in stdout — put one sentence in the write-up.
- **Fit-corpus contamination:** ~27% of the 1,500 heuristic-positive fit documents are LLM-benign (per your own gold set), which is visible as the service-complaint topics and partly explains why the food-poisoning mass won't subdivide. Own it in one sentence; it connects the two halves of your project.
- NMF K=5 was excluded at share 0.6053 vs the 0.60 cutoff — borderline; conclusion robust (its NMI is below K=6's) but worth a sensitivity sentence.
- Validation-set size drift: docstring says ~549, V2 said 561, the artifact says **545** (561 LLM hazards → 557 typed → 545 in the fit corpus). Standardize on 545.

### 2.8 🟡 Data hygiene (an hour, total)

- **Holdout run incomplete: 744/800 rows.** Resumable (`create_gold_dataset.py` skips done rows); statistically adequate as-is, but either finish or update docs to say 744.
- One self-contradictory holdout row: `review_id MlLvFjiF4Agsv6tUB5wh6w` has `llm_is_hazard=1` but `hazard_type="none"`, low confidence, and a rationale arguing benign. Drop or re-judge.
- One exact duplicate-text pair in the holdout (two review_ids, identical text) — `build_holdout_pool.py:192` dedupes on id only; dedupe on normalized text too.
- `cross-contamination` vs `contamination` drift persists in `gold_dataset.csv` (2 vs 24 rows) — normalize before topic validation. The holdout is clean.
- **`labeling/gold_dataset_holdout.csv` is modified-uncommitted** — the project's most important data file is one disk failure from gone. Commit it now.

### 2.9 🟢 Minor code findings (fix if touched, else note)

- `focal_asymmetric` produces NaN gradients for γ<1 (verified: `(1−p)^(γ−1)` → inf at p→1). Never hit by current configs; a one-line `clamp_min(1e-6)` closes it.
- Optuna failure masking: `main.py:93-95` catches every exception and returns 0.0, so a crashed trial looks like a legitimate score; if all trials crash, `analysis.py` silently trains with garbage params.
- Optuna sampler unseeded despite `RANDOM_STATE=42` discipline everywhere else; and trials train 4 epochs while `analysis.py`/grid-search retrain the winner for 3 — hyperparameters selected under one budget, deployed under another.
- `compute_loss` ignores `num_items_in_batch` → mean-of-means under gradient accumulation; standard small bias, one sentence if asked.
- `verify_setup.py` is genuinely good preflight (it would catch the missing packages, contamination, feature drift, gameable metrics) but doesn't check `accelerate`.

---

## Part 3 — What to do next, in order

The strategy is unchanged from V2 but the shape is different: almost everything left is **execution and packaging**, not construction.

| # | Task | Time | Why |
|---|---|---|---|
| 1 | **Unblock and run `analysis.py`**: fix `requirements.txt` (add sentencepiece, matplotlib, seaborn, accelerate; re-pin to the tested versions), set `WANDB_MODE=offline`, commit the holdout CSV, then run the full evaluation on the GPU machine. | ~1h attention + 2–3h GPU | **The hard blocker.** Produces every classifier number (both ground truths), the model error analysis, and all figures in one run. Do it first — if something collapses you need the slack. |
| 2 | Fix the NPMI bool-matmul bug (`.astype(np.int32)`), fix the `NEGATED_HAZARD` regex, rerun the topic sweep, regenerate artifacts, rewrite the coherence narrative in CLAUDE.md. | 1–2h | One-line bug invalidating every persisted coherence value — and the corrected story (coherence and validation agree on K=4) is *better*. |
| 3 | Slide skeleton — problem + cost framing, methodology arc, label-quality findings, topic model. Only the results slides wait on #1. | start today, 4–6h total | 15% of grade at zero; presentation is the day after the deadline. |
| 4 | Loss-variant grid overnight (`grid_search_analysis.py`, e.g. pos_weight vs focal at 2 weights). | unattended GPU | Turns the loss work from claimed to demonstrated — the rubric's "variant experiments" line for the classifier. Skippable only if #1 slips. |
| 5 | Rewrite `README.md` as the one-page problem framing + results index (content already exists in CLAUDE.md/IMPLEMENTATION_NOTES). | 1–2h | Cheap points in two rubric categories; currently the grader's first impression is a one-line README. |
| 6 | Wording/number pass: base-rate → "keyword-screened funnel" everywhere (§2.3); footnote test-split numbers as in-selection (§2.4); 744 not 800, 46.0% not 42.3%, 545 not 561; add the circularity + fit-contamination sentences (§2.7). | 1h | Every item here is a Q&A punch you can take away for free. |
| 7 | Data hygiene from §2.8 + hand-verify a sample per error-analysis bucket (the MD file itself says to). | 1–2h | Converts the rule-based taxonomy into a validated one — an 80s→90s discriminator. |
| 8 | Housekeeping: add instructor + Tal as GitHub collaborators (submission requirement); archive or delete the PROFESSOR_REVIEW*.md files before submission (self-reviews quoting grade estimates invite anchoring). | 15 min | — |

**Cut line:** items 1, 2, 3 are non-negotiable; 5 and 6 are cheap and high-leverage; 4 and 7 are the difference between low-80s and high-80s.

**Honesty line:** any performance number quoted anywhere before item 1 completes is either from the pre-fix recall-selected regime or the 400-row smoke test — both would be dishonest to present as results.

**The story to tell** (updated from V2 — the arc now has real evidence at every step except the last): built a heuristic label → independently judged 1,500 rows → measured 85.8% agreement and a one-sided 27% over-flagging bias → explained *why* the rule errs, bucket by bucket → discovered the first gold set was 89% training data → built a verified zero-overlap holdout and labeled 744 fresh rows → and only then reported numbers. The missing climax is the heuristic-vs-gold performance delta on real models — which is exactly what item 1 produces. Everything else is packaging.
