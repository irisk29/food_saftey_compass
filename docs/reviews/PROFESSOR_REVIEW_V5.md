# Food Safety Compass — Fifth-Pass Review (2026-07-27, late)

*Post-grid review, commit `ca6e686` ("rerun"). The gold-holdout scoring added earlier today
was exercised for the first time, and it worked: it turned a table that could not tell four
configurations apart into one with a 17× larger signal. Every number below was read from the
committed CSVs and cross-checked against the run logs — including the epoch budgets, which
is where the one new problem lives. ~5.5 days to submission (2026-08-02 midnight);
presentations 2026-08-03.*

## Bottom line

**~82–86/100 if submitted today, and the analytical ceiling just moved to the 90 band.**
The loss-variant grid is no longer the "claimed but not demonstrated" item — it ran, and it
produced the single best methodological result in the project: **direct empirical proof that
your heuristic label is too saturated to evaluate anything on.** Validation PR-AUC spread
across eight configurations: 0.0075. Gold-holdout spread across four: 0.1294. Same models,
same day, 17× the signal. That is not a footnote; it is a slide, and it retroactively
justifies the entire two-ground-truths design.

The score has not jumped more because the gap is now *entirely* packaging. Slides are still
a skeleton. README is still one line. Those two items are worth more than everything else
left combined.

One new defect: **the eight grid rows were not all trained under the same epoch budget**
(§2.1), and the CSV has no column recording it. That one is my fault — the runbook command I
gave you omitted `--epochs 2`, so your second batch ran at 3.

---

## Part 1 — What the re-run delivered

### 1.1 ✅ The saturation hypothesis is now proven, not argued

This is the headline finding of the re-run.

| Ground truth | Configurations | PR-AUC range | Spread |
|---|---:|---|---:|
| Heuristic keyword label (validation) | 8 | 0.9818 – 0.9892 | **0.0075** |
| Gold LLM label (fresh holdout) | 4 | 0.7203 – 0.8497 | **0.1294** |

The grid printed the caveat itself, unprompted: *"validation PR-AUC spread across
configurations is only 0.0075 … this metric cannot discriminate between loss variants and
any ranking on it is within single-seed noise."*

Say this out loud in the talk. "We built an evaluation set because we could not tell our own
models apart without one" is a far stronger claim than "we built a holdout because
contamination is bad."

### 1.2 ✅ The best-on-validation model is the worst on real data

The cleanest single illustration in the whole project:

| Config | Validation PR-AUC | Gold PR-AUC |
|---|---:|---:|
| `pos_weight @ w=50` | **0.9892 (best of 8)** | **0.7203 (worst of 4)** |
| `pos_weight @ w=5` | 0.9871 | **0.8497 (best of 4)** |

The configuration that looks best against the keyword label is the one that generalises
worst. If you put one table on a methodology slide, make it this one.

### 1.3 ✅ The collapse myth is dead, with positive evidence

`pos_weight @ w=50` — the exact formulation and weight the old docs claimed collapsed to an
all-positive model — **does not collapse**: flag rate 0.197, recall 0.921, precision 0.936,
`collapsed=False`. Neither does `focal_asymmetric @ w=50` (0.186). No cell in the grid
collapsed on either ground truth.

The doc corrections made earlier today hedged this correctly ("un-persisted historical
observation"). You can now go further and state it positively: *under the current training
protocol, w=50 does not degenerate in either formulation* — with a citation to the artifact.
Delete the hedge, keep the reason bare recall was abandoned (it remains sound a priori, and
`tests/test_losses.py` guards it).

### 1.4 ✅ What the loss variants actually do — a real mechanistic finding

Do not read this table as "which variant wins." Read it as "how does each variant respond to
the penalty magnitude":

| Variant | Gold PR-AUC @ w=5 | Gold PR-AUC @ w=50 | Swing |
|---|---:|---:|---:|
| `pos_weight` | 0.8497 | 0.7203 | **0.1294** |
| `focal_asymmetric` | 0.7782 | 0.7961 | **0.0179** |

You have an accidental replicate that makes this interpretable. `focal_asymmetric @ w=50` was
trained twice under byte-identical settings — once by `analysis.py`, once by the grid (both
3 epochs, lr 1.814e-05, bs 16, verified from both logs). The two runs differ by **0.0084
gold PR-AUC** (0.8045 vs 0.7961; 39 vs 43 false negatives). That is your run-to-run noise
floor on MPS.

Against that floor: `pos_weight`'s swing is **15× noise** — real. `focal_asymmetric`'s is
**2× noise** — barely distinguishable from nothing.

**That is exactly what the focal formulation was designed to do.** Its penalty decays as
`(1−p)^γ`, so once the model is confident the weight stops mattering; a 10× change in `w`
moves it almost not at all. Plain `pos_weight` applies the penalty uniformly to every
positive, so cranking it to 50 over-penalises and measurably damages out-of-sample ranking.

So the honest, defensible claim is: **the custom loss buys insensitivity to a
hyperparameter, not peak performance.** That is a better result than "our loss wins," it is
supported by 15× and 2× effect sizes against a measured noise floor, and it is precisely the
"understand *why* the technique behaves as it does" the 90-band rubric line asks for. Lead
with it.

### 1.5 ✅ The machinery worked exactly as built

Resume skipped the 4 completed cells rather than re-paying ~2 GPU-hours for them; the
saturation caveat fired with the correct spread; the "NOT MEASURED, not zero" warning
correctly named all four blank-gold rows; the summary re-sorted on `gold_pr_auc`. Nothing to
fix here.

---

## Part 2 — What is broken or missing now

### 2.1 🔴 NEW — the grid mixes two epoch budgets, and nothing in the CSV says so

Verified from progress-bar step counts in `results/grid_search.log`: the first invocation ran
**600 optimiser steps (2 epochs)**, the second ran **900 (3 epochs)**.

| Rows | Weights | Epochs | Gold scored? |
|---|---|---:|---|
| `pos_weight@1`, `pos_weight@15`, `focal@1`, `focal@15` | 1, 15 | **2** | ❌ |
| `pos_weight@5`, `pos_weight@50`, `focal@5`, `focal@50` | 5, 50 | **3** | ✅ |

This is my error — the runbook command I gave you dropped the `--epochs 2` the original run
used.

**The damage is contained**, and this matters: *all four gold-scored rows are 3-epoch*, so
every comparison in §1.1–§1.4 is internally consistent and stands. What is not safe is the
8-row validation table — comparing `pos_weight@1` against `pos_weight@5` there confounds
weight with training budget.

Two things to do:
- **Add an `epochs` column to the CSV.** As committed, the artifact silently presents eight
  rows as comparable. A reader cannot recover the budget from it. This is a data-integrity
  fix, not cosmetics.
- **Re-run the four 2-epoch cells at 3 epochs** (`--force --weights 1 15`, ~4 GPU-hours
  unattended). This makes all eight rows comparable *and* fills their gold columns, which
  turns §1.4's two-point comparison into a four-point weight curve (w=1, 5, 15, 50) per
  variant — a genuine figure, and the strongest version of the mechanistic story.
  Worth one night if you have one; see §3 for where it sits against the slides.

### 2.2 🟠 The deployed configuration is dominated on the gold holdout

Your reported system is `focal_asymmetric @ w=50`. Among the four gold-scored cells it ranks
**3rd of 4 on PR-AUC** (0.7961 vs `pos_weight@5`'s 0.8497 — a 0.054 gap, ~6× noise) and
**worst of 4 on your own cost model** ($225,700 vs `focal@5`'s $112,250).

| Config | Gold PR-AUC | FN | FP | Risk cost |
|---|---:|---:|---:|---:|
| `pos_weight @ 5` | **0.8497** | 32 | 204 | $170,200 |
| `focal @ 50` *(deployed)* | 0.7961 | 43 | 214 | $225,700 |
| `focal @ 5` | 0.7782 | **20** | 245 | **$112,250** |
| `pos_weight @ 50` | 0.7203 | 37 | 236 | $196,800 |

Note PR-AUC and cost disagree, and the reason is instructive: cost is computed at a fixed
th=0.20 while PR-AUC is threshold-free, so a model that simply flags more (focal@5 flags
75.1% of the holdout) looks cheap under a 100:1 cost ratio. **Do not select on fixed-threshold
cost** — pick on PR-AUC, then re-tune the threshold per model. Worth one sentence in the
write-up; a sharp grader will ask why the two columns disagree.

**My recommendation: do not switch the deployed config.** Switching means re-running
`analysis.py` (2–3h), regenerating every performance CSV, figure and error-analysis file, and
re-checking every number you have already put in the docs — 5 days out, with no slides built.
The committed headline (gold PR-AUC 0.804 vs the baseline's 0.728) is strong and stands on
its own. Disclose the finding instead: *"the deployment weight was fixed before the variant
grid was run; the grid subsequently showed a lower penalty ranks better out-of-sample, which
we report rather than retrofit."* That reads as discipline. Silently swapping to the
better-scoring config and hoping nobody asks reads as the opposite.

### 2.3 🟠 "Our custom loss beats plain class weighting" is now refuted — write it up that way

Previously this claim was merely undemonstrated. It is now contradicted: the best gold
PR-AUC belongs to `pos_weight`, which is exactly `BCEWithLogitsLoss(pos_weight=w)` — the
plain baseline the custom loss was meant to improve on.

This is fine, but only if you say it first. The §1.4 framing (robustness, not peak
performance) is true, evidenced, and rubric-friendly. What would be damaging is a slide
claiming superiority that the committed CSV refutes. Check the skeleton for any such wording
before you build the deck.

### 2.4 🟠 Carried over from V4, unchanged and still the highest-value items

- **Slides: skeleton only.** 15% of the grade at zero, presentation the day after the
  deadline. Still the largest single risk in the project.
- **README: one line.** Grader's first impression.
- **LDA is environment-dependent** (V4 §2.1): quote NMF (K=6, allergen lift 5.28, per-topic
  NPMI +0.43) as the headline topic result; LDA numbers only from committed artifacts. The
  skeleton still carries the stale 0.716 / 4.18 and 744/342.
- **Hyperparameter provenance** (V4 §2.3): no `optuna_trials.csv`; `analysis.py` used the
  pre-fix hardcoded defaults. One disclosure sentence is the right trade.
- **`requirements.txt` and hardcoded W&B** (V4 §2.4): a clean clone still cannot run.
- **`NEGATED_HAZARD` regex** still omits `haven't/hasn't/hadn't`; 23 of 39 gold FNs remain
  `unexplained_fn` (V4 §2.5).
- **Three defective holdout rows**; "hazard base rate" wording at `data_pipeline.py:86`.

---

## Part 3 — What to do next, in order

| # | Task | Time | Why |
|---|---|---|---|
| 1 | **Build the deck.** Correct as you go: 772/355, NMF-first topics, and add the two new slides this grid earned — the saturation table (§1.1) and the best-on-validation-worst-on-gold inversion (§1.2). | 4–6h | 15% of the grade at zero. Unchanged as top priority. |
| 2 | **Add an `epochs` column** to `grid_search_analysis.py` and backfill the committed CSV (2 for w∈{1,15}, 3 for w∈{5,50} — verified from the log). | 20 min | The artifact currently misrepresents eight rows as comparable. |
| 3 | README rewrite. | 1–2h | First impression; content already exists. |
| 4 | Disclosure pass: the §2.2 deployment-weight sentence, the §2.3 honest framing of the loss result, the PR-AUC-vs-cost explanation, plus the V4 carry-overs (LDA reproducibility, hyperparameter provenance, funnel wording). | 1–1.5h | Every line here removes a Q&A punch for free. |
| 5 | **Optional overnight:** `--force --weights 1 15` to put all 8 rows on 3 epochs with gold scores. | unattended | Turns §1.4's 2-point comparison into a 4-point curve per variant. Best remaining use of a spare night — but only after 1 and 3. |
| 6 | `NEGATED_HAZARD` regex + hand-read the 23 unexplained FNs; re-run `analyze_errors` from the detail CSVs. | 1–2h | No GPU. Converts "59% unexplained" into a named failure mode. |
| 7 | Hygiene: 3 holdout rows, `requirements.txt`, `WANDB_MODE`, archive the five review/overview docs, add instructor + Tal as collaborators. | 1h | Submission requirements and clean-clone reproducibility. |

**Cut line:** items 1 and 3 are the difference between low-80s and high-80s. Item 4 is cheap
and buys disproportionate credibility. Item 5 is the only thing left that could add new
*evidence*, and it is still worth less than a finished deck.

**Do not** re-run `analysis.py` to chase the better config (§2.2), and **do not** re-run the
topic model hoping the old LDA numbers return (V4 §2.1). Both cost hours and buy nothing.

**The story to tell — now with a methodological spine, not just a result:** built a heuristic
label → audited it against an independent judge (85.8% agreement, one-sided 27% over-flagging)
→ explained *why* it errs, bucket by bucket → caught the first gold set 89% contaminated →
built a verified zero-overlap holdout (772 rows) → trained under a clean protocol → **and then
discovered we could not tell our own models apart on the heuristic label at all (spread 0.0075)
until we scored them on the holdout (spread 0.1294)** → which revealed that the
best-on-validation configuration is the worst on real data, that heavy uniform FN penalties
hurt generalisation while the focal formulation is 7× less sensitive to that choice, and that
the transformer's residual false alarms are precisely the label's own inherited blind spots.
Every step of that arc is now backed by a committed artifact. What is missing is a deck to
say it on.
