# Overview & Gap Analysis — `final_work.docx` vs the repo (2026-07-26)

*Assessment of the written report ("Allergen & Food Safety Hazard Compass Engine") against the actual state of the repo as verified in [PROFESSOR_REVIEW_V3.md](PROFESSOR_REVIEW_V3.md). 7 days to submission.*

## Bottom line

**The document is well-written and its narrative instincts are exactly right — but it describes a different project than the one in the repo.** Roughly half of its Results and Iteration sections report experiments, datasets, loss functions, and numbers that do not exist in the committed code or `results/` directory, and several of its technical claims repeat things the repo has since proven wrong. Meanwhile the repo's two strongest, fully-evidenced assets — the label-quality analysis (85.8% agreement, 27% over-flagging, error taxonomy) and the **entire topic-modeling technique** — are absent from the document.

This must be reconciled before submission, in one of two directions: either the docx's Phase III/IV experiments exist somewhere (a teammate's machine, an old branch) and get committed with their artifacts, or the document is rewritten around what the repo can actually prove. A grader who cross-reads the report against the code will find the mismatches in minutes — and the assignment grades reproducibility and honesty of analysis heavily.

---

## Part 1 — What the document gets right (keep all of this)

- **Background & problem framing (§Background) is the best-written part.** "Aggregate rating masking" is a crisp, memorable frame; the 4.5-star restaurant with a systemic allergy failure is a strong opening example. Reuse verbatim in the README and slide 2.
- **The research question is well-formed** and matches the repo's design (text-only transformer vs tabular+TF-IDF baseline).
- **Data section is accurate and matches the code.** The streaming pipeline description (domain filter → length filter → hazard/benign buffers → 1,500/6,000 merge, seed 42) matches the preprocessing notebook, including the honest 0.12% natural-sparsity motivation. The feature table matches the postprocessing notebook.
- **The leakage narrative (Phase II) is genuinely strong and true.** The "100% baseline = red flag → the model reverse-engineered our own labeling rule" story is real, evidenced, and is exactly the methodology arc V2/V3 recommend building the presentation around.
- **Future-work section is sensible** (XAI attribution, ordinal risk tiers, cross-platform transfer) and costs nothing.

## Part 2 — Claims that conflict with the repo or are now known to be wrong

### 2.1 🔴 The Phase III/IV experiments and final results are not in the repo

| Docx claim | Repo reality |
|---|---|
| Gold Dataset of **1,132 records**, stratified **905 train / 227 validation** split, *trained on* | Two gold sets exist: 1,500 in-sample (agreement analysis only) and **744-row holdout** (evaluation only). Nothing with 1,132/905/227 exists anywhere. No code trains on gold labels. |
| `AttenuatedAsymmetricBCEWithLogitsLoss`, ω dialed 50→**5.0**, **linear epoch annealing** ω: 1→max | `src/losses.py` has `pos_weight`, `focal_asymmetric`, `fn_gated`. **No attenuation/annealing class exists in any committed file.** |
| Final results: **PR-AUC 0.9279, Recall 90.59%, Precision 72.64%** (Figure 3) | `results/` contains **zero model-performance artifacts**. No run under the fixed configuration has ever completed. These numbers are unreproducible from the repo. |
| Baseline collapse to **74.10% recall / 22 missed** on gold (Figure 1) | No artifact backs this; plausibly from an uncommitted earlier run. |
| SOTA collapse to 37.4% precision at 100% recall (Figure 2) | The *phenomenon* is confirmed (grid-search notes observed the w=50 collapse) — but no persisted figure/CSV exists. |
| Figures 1–3 | No figure files in the repo. |

**Decision needed now:** if these runs happened, commit the code, data, and figures that produced them (and reconcile the 1,132-row dataset with the two gold sets the repo documents). If they cannot be recovered, the Results/Phase III/Phase IV sections must be rewritten around the output of `analysis.py` — which is V3's #1 blocker anyway. Do not submit a report whose headline numbers the repo cannot generate.

### 2.2 🔴 The performance matrix reports the pre-fix, discredited numbers

The table (Baseline 100%/100%; DeBERTa 97.7%/88.3% at th=0.50, etc.) is measured **against the heuristic label** with models selected by the old recall-only regime the repo has since abandoned (selection is now F2/PR-AUC precisely because recall-selection produced a degenerate model). The document does frame the baseline's 100% as a leakage artifact — good — but then presents the DeBERTa rows as valid results without noting they share the same inflated ground truth. V3's honesty line applies: **no number predating the fixed configuration should appear as a result.** Present the 100% row only as the leakage exhibit it is.

### 2.3 🟠 The asymmetric-loss description repeats the refuted "false negative penalty" claim

§"Asymmetric Loss Core" says the loss forces "an explicit 50x gradient penalty multiplier on any True Positive... misclassified as benign (a False Negative)." **This is the exact mislabeling V2 §2.2 caught**: the ω=50 loss keys on the *label*, not the *error* — it upweights every positive example whether classified correctly or not, and is mathematically identical to `BCEWithLogitsLoss(pos_weight=50)`. The repo now has honestly-documented error-dependent variants (`focal_asymmetric`, `fn_gated`, verified numerically). Rewrite this section around the loss *family* and the w=50 collapse as the motivating failure — that is both true and a better story.

### 2.4 🟠 Terminology that will draw fire in Q&A

- **"Stratified Balanced Streaming Pipeline"** — the hazard buffer is *keyword-filtered*, not stratified in the statistical sense; and the resulting 20% "hazard distribution" is 20% *heuristic-flagged*, of which ~27% are benign per your own gold set. One honest sentence defuses this.
- **"does combining a SOTA model with tabular metadata improve detection"** (research question) — the actual design *withholds* metadata from DeBERTa and gives it only to the baseline; nothing "combines" them. Reword to what was tested: *contextual sequence modeling vs lexical+tabular statistical baseline*.
- **"maximize validation Recall"** (§Optuna) — stale; the repo now optimizes PR-AUC after the recall-selection failure. Update, and tell that failure as a finding.
- 50× multiplier vs the (implied) FN:FP cost framing — if the report introduces a cost rationale, reconcile the numbers explicitly (V2 §2.2's point).

### 2.5 🟠 What the document is missing entirely

1. **Topic modeling — the second course technique — is completely absent.** The rubric's 80–89 band requires deep understanding of ≥2 techniques; the document currently presents one. The repo has the full LDA/NMF study with persisted artifacts (allergen-topic lift 5.28, the "finds lexically-distinct, fails on lexically-diffuse" finding). This is a mandatory addition.
2. **The label-quality findings** — 85.8% agreement, the one-sided 27% over-flagging, and the bucketed error taxonomy (48% "illness mentioned, not caused here", 12% negated hazards) — the repo's single most presentable result, and stronger evidence for the document's own Phase II thesis than anything currently in it.
3. **The gold-set contamination discovery** (first gold set was 89% training data → verified zero-overlap holdout built). This is the best chapter of the methodology arc and it's not in the report.
4. **The base-rate caveat**: any hazard-rate or gold-set metric must say it's measured within the keyword-screened funnel (100% of holdout rows pass the STRONG/WEAK filter; 46% funnel rate ≠ population rate).
5. **Limitations & error analysis of the final model** — the assignment's results-analysis category (25%) wants why the model fails, not only how well it scores. (Pending `analysis.py`.)

### 2.6 🟡 Small factual/consistency nits

- "24 hazards out of 20,000 sequential reviews" and window "5–800 words" — fine, but keep consistent with the notebook if quoted.
- The docx's negation-window feature and medical-lexicon density are described as model inputs; per `config/settings.py` they are now **excluded** from the tabular features (leakage) — the feature table's "Role in the Modeling Ecosystem" column is stale for `stars`, `medical_lexicon_density`, `negation_window_flag`. Mark them "engineered, later excluded after leakage audit" — which is a *good* look, not a bad one.
- σ(·) is described as "the model's predicted probability logit" — it's the sigmoid of the logit, i.e. the probability. Fix before a formula-literate grader reads it.
- "Character densities between 5 and 800 words" (Phase I) — words, not characters; and Phase I says the same filter twice with different units.

---

## Part 3 — Recommended rewrite plan (mirrors the V3 action order)

1. **Run `analysis.py` first** (V3 item 1). Every empty results slot in the document fills from that one run: baseline-vs-DeBERTa on heuristic *and* gold-holdout labels, the heuristic-vs-gold delta, model error analysis, PR/cost curves.
2. **Restructure the middle of the document around the true four-phase arc** (the docx's phase structure is right; its contents need swapping):
   - Phase I: heuristic label + baseline (keep as-is).
   - Phase II: leakage discovery — keep, and add the *feature exclusion* fix and the LLM audit numbers (85.8% / 27% over-flagging / taxonomy).
   - Phase III: the validation set was itself contaminated (89% in train) → holdout construction, zero-overlap verification, 744 rows, funnel caveat.
   - Phase IV: selection-metric failure (recall → all-positive collapse at ω=50) → F2/PR-AUC selection + error-dependent loss variants → final numbers from `analysis.py`.
3. **Add a Topic Modeling chapter** (LDA/NMF, K sweep, corrected NPMI after the V3 §2.1 bug fix, lift table, the honest negative result).
4. **Replace the performance matrix** with the new dual-ground-truth table; keep the old 100% row only as the leakage exhibit.
5. **Fix the loss section** per §2.3 above; fix the research-question wording per §2.4.
6. Keep Background, Data, and Future Work nearly untouched.

**Reuse note:** the docx's prose for Background/Phase II and the repo's verified numbers are complementary — the rewrite is mostly *transplanting evidence into an already-good skeleton*, not writing from scratch. The same structure drives the slides: see [slides/SLIDES_SKELETON.md](slides/SLIDES_SKELETON.md).
