# Food Safety Compass — Fourth-Pass Review (2026-07-27)

*Post-execution review. The V3 blocker is gone: `analysis.py` has been run end-to-end
("results until stage 4", commit `557df34`) and every claim below was re-verified against
the committed artifacts — confusion counts recomputed from the persisted metrics, cost
arithmetic re-derived, the topic sweep re-run independently in a second environment.
6 days to submission (2026-08-02 midnight); presentations 2026-08-03.*

## Bottom line

**~80–85/100 if submitted today — up from V3's 68–73, and for the right reason.** The
evaluation apparatus V3 called "built but never turned on" has now produced every number
the write-up needs, and the numbers deliver the story: the classifier climax (the
heuristic-vs-gold performance delta) is real, large, and points in the direction the
whole project predicted. The remaining gap to 90 is almost entirely packaging — a slide
deck that exists only as a skeleton with stale numbers, a one-line README, and one
demonstration gap (the loss-variant grid).

One genuinely new technical finding surfaced during verification: **LDA topic-model
results are not reproducible across machines even with a fixed seed** (§2.1). NMF is.
This invalidated several numbers quoted in CLAUDE.md and the slide skeleton; CLAUDE.md
was corrected today, the skeleton was not.

---

## Part 1 — The results, verified

### 1.1 ✅ The climax exists and is quotable

`results/ground_truth_comparison.csv`, verified arithmetically consistent with the
per-set performance tables (TP/FN/FP counts and the cost model re-derived exactly:
39×5,000 + 197×50 = 204,850 ✓).

| | vs heuristic label (test split) | vs gold LLM label (fresh holdout) | delta |
|---|---:|---:|---:|
| Baseline XGBoost — PR-AUC | 0.979 | 0.728 | **−0.251** |
| Baseline XGBoost — recall | 0.993 | 0.673 | −0.320 |
| DeBERTa-v3 — PR-AUC | 0.987 | 0.804 | **−0.183** |
| DeBERTa-v3 @0.20 — recall | 0.947 | 0.890 | −0.057 |

Three findings fall out of this table, each a slide:

1. **The baseline memorised the labelling rule.** 2 FN + 5 FP out of 1,500 against the
   heuristic label — near-perfect fidelity to a keyword rule TF-IDF can trivially
   recover — then a 0.25 PR-AUC collapse and 32-point recall drop on real ground truth.
   This is the "TF-IDF recovers the label" hypothesis, now demonstrated instead of asserted.
2. **The transformer generalises past its own training label.** Its heuristic→gold drop
   is smaller on every metric, and on the holdout it beats the baseline by +0.077 PR-AUC,
   +22 recall points, and a **2.9× lower total risk cost** (204,850 vs 585,300).
3. **The tuned threshold is vindicated out-of-sample.** Moving 0.50→0.20 on the holdout
   buys +3.4 recall points for −0.3 precision points (0.619→0.616) — recall almost free,
   on data the threshold was never tuned on. FNs drop 51→39.

### 1.2 ✅ V3 §2.4 closed — the validation split ran and worked

`load_and_split_data(with_validation=True)` produced 4,800/1,200/1,500 with the test
split verified byte-identical to the historical one. Checkpoint selection happened on
the validation split (`checkpoint_selection.json`: F2 and PR-AUC both pick epoch 2 of 3,
val F2 0.928) — so the heuristic test-split table is now **out-of-selection** and needs
no footnote. The gold holdout remains out-of-everything.

### 1.3 ✅ V3 §2.1 closed — NPMI fixed, artifacts regenerated (see §2.1 for the twist)

The `astype(np.int32)` fix is in, the sweep re-ran, and all persisted coherence values
are now positive and non-monotonic. Committed artifact: LDA coherence peaks at K=8,
external validation still selects K=4; NMF coherence still prefers the degenerate K=2
(one topic = 95% of docs), validation selects K=6. The lift result that matters —
**NMF's gluten/celiac topic, lift 5.28 for `allergic_reaction`, per-topic NPMI +0.43** —
survives the fix and is reproducible (§2.1).

### 1.4 ✅ V3 §2.5 half-closed — the model error analysis now exists

`error_analysis_deberta_gold_llm_label_fresh_holdout.md` buckets all 197 FPs + 39 FNs.
The FP taxonomy is the analytical payoff of the whole project:

- **34% `illness_mentioned_not_caused_here` + 32% `neutral_allergen_mention`** — two
  thirds of the model's false alarms are *exactly the labelling rule's own top failure
  modes*. The model inherited its teacher's blind spots. Best single example in the
  file: a review *praising* a shop's cross-contamination prevention, flagged at p=0.995.
  The model has learned "allergen vocabulary ⇒ hazard" with no polarity or causation.
- FN side: 7/39 are `buried_in_long_review` (past the 256-token truncation — directly
  actionable), but **23/39 (59%) are `unexplained_fn`** — see §2.5.

### 1.5 ✅ Data movement since V3

Holdout grew 744 → **772 rows** (417 benign / 355 hazard, 46.0% funnel rate, 767 high
confidence) — labeling was resumed as recommended. The holdout CSV is committed and the
working tree is clean. (The two hygiene defects V3 flagged are still in it — §2.6.)

---

## Part 2 — What is broken or missing now

### 2.1 🔴 NEW — LDA results are environment-dependent; two docs quote numbers that no committed artifact contains

Re-running the fixed topic sweep in a second environment (same code, same data, same
`random_state=42`) produced **different LDA fits**: allergen-topic lift 4.18 vs the
committed 2.52, purity 0.716 vs the committed 0.690, coherence peak K=4 vs K=8. NMF
reproduced exactly in both environments (its `nndsvda` init is deterministic; LDA's
variational updates are sensitive to library version / BLAS).

Consequences, in order of embarrassment potential:

- **CLAUDE.md quoted purity 0.716 and "LDA lift 4.18" — numbers from a run whose
  artifacts no longer exist.** Fixed today: the findings section now matches the
  committed artifacts, leads with NMF, and carries an explicit reproducibility caveat.
- **`slides/SLIDES_SKELETON.md` still carries the stale numbers** (0.716, 4.18 — plus
  744/342 for the holdout, now 772/355). Fix before building the deck (§3, item 1).
- In the committed artifact, **LDA K=4 purity is 0.690 — exactly the majority baseline**.
  The "recovers structure" claim now rests entirely on NMI-above-null and on NMF's lift.
  Lead with NMF everywhere; it is both the stronger and the reproducible result.
- Defensively, this is a *good* Q&A story if told first: "we caught our own topic model
  being irreproducible across environments, and moved the headline claims to the
  algorithm that reproduces" — one sentence in the write-up converts a liability into
  evidence of rigor.

### 2.2 🟠 The loss-variant grid is still the one rubric line that is claimed, not demonstrated

`results/grid_search_loss_variants.csv` does not exist. Everything is wired
(`grid_search_analysis.py`, now correctly scoring the validation split) and the run is
unattended GPU time. `--quick` (2 variants × 4 weights) is enough to demonstrate the
comparison and the w=50 collapse. This was V3 item 4; it is now the only piece of the
"variant experiments" rubric line missing for the classifier.

### 2.3 🟠 The reported hyperparameters have pre-fix provenance

`optuna_trials.csv` / `best_hyperparameters.json` are absent, so `analysis.py` fell back
to its `DEFAULT_LR` / `DEFAULT_BATCH_SIZE` literals — which came from the *old* sweep,
run before the metric fixes and without the validation split. The results themselves are
clean (evaluation never touched selection data); only the *provenance* of lr=1.81e-5,
bs=16 is impure. Two honest options: re-run `main.py` (3 trials, one overnight) or add
one sentence — "hyperparameters were carried over from an earlier sweep; the final
configuration was re-trained and evaluated under the fixed protocol." Do not present the
defaults as the product of the current pipeline.

### 2.4 🟠 Reproduction is still broken for a fresh machine (V3 §2.2, half-resolved)

The run succeeded because it ran on a machine with W&B configured. Unchanged:
`report_to="wandb"` hardcoded (`src/sota_model.py:183`), and `requirements.txt` still
missing `sentencepiece`/`matplotlib`/`seaborn`/`accelerate` while pinning
`transformers==4.40.0` / `torch==2.2.2`, which nobody runs. A grader cloning the repo
cannot reproduce anything. Both fixes remain one-liners.

### 2.5 🟡 59% of gold false negatives are `unexplained_fn` — and the negation regex fix was never applied

The model-error taxonomy explains FPs well but 23/39 FNs land in the no-rule-matched
bucket. A one-hour hand pass over those 23 (they are in
`error_analysis_deberta_gold_llm_label_fresh_holdout_detail.csv`, and several visible
examples are hazards stated in the review's final sentence — consistent with truncation)
would either grow the truncation bucket or find a new mode. Also: `NEGATED_HAZARD`
(`analysis/error_analysis.py:39`) still omits `haven't/hasn't/hadn't` (V3 §2.5), so the
committed error-analysis MDs were generated with the known-flawed regex. Fix and re-run
`analyze_errors` — no retraining needed, the detail CSVs carry the probabilities.

### 2.6 🟡 Holdout hygiene — same two rows as V3, plus the wording pass

- Still present: 1 exact duplicate-text pair, 1 self-contradictory row
  (`llm_is_hazard=1`, `hazard_type="none"`, low confidence). Dropping 3 rows out of 772
  changes nothing statistically; do it and re-run the (cheap) holdout evaluation cells,
  or one footnote.
- `data_pipeline.py:86` still prints "hazard base rate" — say "hazard rate within the
  keyword-screened funnel" (V3 §2.3). The slide skeleton already does this correctly.
- Number drift to sweep: 744→772 and 342→355 in the skeleton and any doc that quotes
  them; topic numbers per §2.1.

### 2.7 🟡 Packaging (unchanged from V3)

- README is still one line — the grader's first impression. The content exists in
  CLAUDE.md; this is a 1–2h rewrite.
- `slides/SLIDES_SKELETON.md` is a genuinely good skeleton (spine, per-slide evidence
  lines, timing) — but it is not a deck, presentations are the day after the deadline,
  and its numbers need the §2.1/§2.6 corrections first.
- PROFESSOR_REVIEW*.md (now four of them) still in the repo; archive before submission.
  Instructor + Tal as collaborators — unverified, 15 minutes.

---

## Part 3 — What to do next, in order

| # | Task | Time | Why |
|---|---|---|---|
| 1 | **Build the deck** from the skeleton, after correcting its numbers (772/355; NMF-first topic story; committed-artifact LDA numbers only). The results slides can now be filled — nothing is pending. | 4–6h | 15% of grade; the only remaining zero. |
| 2 | **Loss grid, `--quick`, overnight.** | unattended | Last missing rubric line for the classifier (§2.2). |
| 3 | README rewrite: problem framing + headline table from §1.1 + results index. | 1–2h | First-impression fix; content already written. |
| 4 | Wording/number pass (§2.1, §2.6): skeleton numbers, funnel phrasing, NMF-first, one provenance sentence for the hyperparameters (§2.3), one reproducibility sentence for LDA. | 1h | Every item is a free Q&A punch removed. |
| 5 | Fix `NEGATED_HAZARD` regex + hand-review the 23 unexplained FNs, re-run `analyze_errors` from the detail CSVs. | 1–2h | Converts the FN story from "59% unexplained" to a finding; no GPU needed. |
| 6 | Drop the 3 defective holdout rows, re-run holdout eval; fix requirements.txt + `WANDB_MODE` one-liners. | 1h | Hygiene + a grader who clones the repo can actually run it. |
| 7 | Housekeeping: archive PROFESSOR_REVIEW*.md, add instructor + Tal as collaborators. | 15 min | Submission requirement. |

**Cut line:** items 1–3 are the difference between low-80s and high-80s; items 4–5 are
the difference between high-80s and 90+.

**The story to tell — now complete at every step:** built a heuristic label → audited it
(85.8% agreement, one-sided 27% over-flagging) → explained *why* it errs, bucket by
bucket → caught the first gold set contaminated (89% in train) → built a verified
zero-overlap holdout (772 rows) → trained under a clean selection protocol
(val split, F2/PR-AUC) → and the numbers landed the predicted punch: **the baseline
memorised the rule and collapsed on real ground truth (−0.25 PR-AUC); the transformer
generalised (−0.18) and cut deployment risk cost 2.9×; and its residual false alarms are
precisely the label's own blind spots inherited.** That last clause — model errors
traced back to label pathology, with the topic model finding the one lexically-distinct
hazard type — is a 90-band arc. What stands between here and there is a deck, a README,
and one overnight GPU run.
