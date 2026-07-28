# Food Safety Compass — Sixth-Pass Review (2026-07-28)

*Post full-grid re-run, commit `5fc0d88` ("rerun stage 1 with full results (golden dataset)"),
merged as `38ad1d8`. All eight loss-variant cells were re-trained from scratch under `--force`
and all eight now carry gold-holdout scores. Every number below was read from
`results/grid_search_loss_variants.csv` and cross-checked against `results/grid_search.log`
(step counts, per-epoch eval dicts, checkpoint selection) and against the independently
committed `results/performance_gold_llm_label_fresh_holdout.csv`. ~4.5 days to submission
(2026-08-02 midnight); presentations 2026-08-03.*

## Bottom line

**~83–87/100 if submitted today.** The re-run fixed the one data-integrity defect V6's
predecessor raised — all eight rows are now 3 epochs, so the eight-row table is finally a
legitimate comparison — and it doubled the gold-scored evidence from four cells to eight. The
saturation finding, which is the best methodological result in the project, came back
*stronger*: validation spread 0.0030 against gold spread 0.0992, a **33×** ratio, up from 17×.

It also did something more valuable and less comfortable. Four configurations have now been
trained twice under nominally identical settings, and those replicate pairs give you a
**measured noise floor for the first time** — and it is much larger than V5 assumed. One pair
moved **0.054 gold PR-AUC** between runs. V5 quoted a noise floor of 0.0084 and built
"15× noise" and "7× less sensitive" claims on it. Those specific claims do not survive. The
underlying story does, and §1.4 below shows it survives on better evidence than before.

Read that as good news, not bad. You now have an honest error bar, and a project that reports
one is in a different band from a project that reports point estimates it cannot defend.

**The gap remains packaging, and it has not moved since V5.** Slides are still a 113-line
skeleton. That is 15% of the grade sitting at zero with four days left, and it is worth more
than everything else on this list combined.

---

## Part 1 — What the full re-run delivered

### 1.1 ✅ The epoch defect is gone — the eight-row table is now legitimate

V5 §2.1 flagged that four rows had been trained at 2 epochs and four at 3, making the
validation table incomparable. The re-run resolves this by brute force: `--force` discarded all
eight prior cells and retrained every one of them. Verified from the log — each of the eight
cells ran **900 optimiser steps** (4,800 train rows ÷ effective batch 16 = 300 steps/epoch ×
3 epochs), against 600 for the old 2-epoch cells. Identical learning-rate schedule across all
eight (linear decay from 1.814e-05), identical batch size, identical 256-token truncation.

All eight rows are comparable. Every conclusion in this section rests on that.

### 1.2 ✅ Saturation, proven harder: 33× and a *negative* rank correlation

This is still the headline, and it is now measured two ways instead of one.

| Ground truth | Configurations | PR-AUC range | Spread |
|---|---:|---|---:|
| Heuristic keyword label (validation) | 8 | 0.9864 – 0.9894 | **0.0030** |
| Gold LLM label (fresh holdout) | 8 | 0.7219 – 0.8211 | **0.0992** |

The grid printed the caveat itself, unprompted: *"validation PR-AUC spread across
configurations is only 0.0030 … this metric cannot discriminate between loss variants and any
ranking on it is within single-seed noise."*

The stronger version of the claim — and this is new, V5 could not compute it with four points —
is the **rank correlation between the two ground truths across all eight configurations:**

> **Spearman ρ = −0.24** (p = 0.57). Pearson r = −0.36.

That is a better slide than the spread ratio. The spread ratio says the heuristic metric has
*low resolution*. The rank correlation says it has **no information** — knowing a
configuration's validation rank tells you nothing about its gold rank, and if anything points
mildly the wrong way. "We could not tell our own models apart on our own label, and the
ordering it did give us was uncorrelated with the ordering on real data" is the single
sharpest methodological sentence available to you.

Retire V5 §1.2's framing ("the best-on-validation model is the worst on real data"). With
eight points that anecdote no longer holds cleanly — best-on-validation is now `pos_weight@5`,
which ranks 4th of 8 on gold, not last. The rank correlation is both true and stronger.

### 1.3 ✅ The collapse myth stays dead, now across eight cells

`collapsed = False` in **all eight rows**, on both ground truths. Maximum validation flag rate
0.2075 (`focal_asymmetric@5`); `pos_weight@50` — the exact formulation and weight the old docs
claimed degenerated to an all-positive model — sits at flag rate 0.1967, recall 0.921,
precision 0.936. The evidence base for "under the current training protocol, w=50 does not
degenerate in either formulation" has doubled. State it positively and cite the artifact.

### 1.4 ✅ The mechanistic finding survives, on better evidence — a monotone curve vs a flat one

V5 had two points per variant and read a swing ratio off them. You now have four, and the shape
is more informative than the range:

| Weight | `pos_weight` gold PR-AUC | `focal_asymmetric` gold PR-AUC |
|---:|---:|---:|
| 1 | 0.8096 | 0.7913 |
| 5 | 0.7956 | 0.7781 |
| 15 | 0.7905 | **0.8211** |
| 50 | **0.7219** | 0.8021 |
| **Range** | **0.0877** | **0.0430** |

`pos_weight` declines **monotonically** across all four weights. Under pure noise the chance of
drawing an exactly monotone ordering of four points is 1/24 ≈ 4%. That is a real pattern, and
it is the pattern theory predicts: a uniform penalty on every positive, cranked to 50,
over-penalises and measurably damages out-of-sample ranking.

`focal_asymmetric` is **non-monotone** — it wanders within a band half as wide and does not
respond to the penalty magnitude in any consistent direction. That is exactly what the
formulation was designed to do: its penalty decays as `(1−p)^γ`, so once the model is confident
the weight stops mattering.

So the defensible claim is unchanged from V5 in substance and better supported in form:
**the custom loss buys insensitivity to a hyperparameter, not peak performance.** What must
change is the arithmetic around it:

- ❌ Do not say "7× less sensitive." Say **2× narrower range (0.043 vs 0.088)**, and lead with
  *monotone vs non-monotone*, which is the qualitative fact and does not depend on a noise
  estimate.
- ⚠️ There is a tempting mis-statement here to avoid. Comparing only the **endpoints** (w=1 vs
  w=50) gives `pos_weight` 0.088 against `focal` 0.011 — an "8× less sensitive" headline. Do not
  use it. `focal` is non-monotone, so its endpoints happen to sit close together while the curve
  wanders 0.043 in between; the endpoint ratio flatters the result by ignoring the two middle
  points. Quote the full range across all four weights.
- ❌ Do not say "15× noise." See §2.1 — the noise floor is 0.054, so `pos_weight`'s 0.088 range
  is roughly **1.6× the largest replicate discrepancy observed**, not 15×. The monotonicity
  argument is what carries this result now, not the effect size.

### 1.5 ✅ The deployed configuration reproduces across three independent trainings

`focal_asymmetric @ w=50` has now been trained three times by two different scripts, and scored
against the same 772-row holdout each time:

| Run | Source | Gold PR-AUC | Gold FN |
|---|---|---:|---:|
| A | grid, 2026-07-27 | 0.7961 | 43 |
| B | grid, 2026-07-28 (this re-run) | 0.8021 | 43 |
| C | `analysis.py` (the reported system) | **0.8045** | 39 |

Range 0.0084. **Your headline number — gold PR-AUC 0.804 — is independently reproduced to
within 0.008 by a completely separate code path.** That is a genuinely good reproducibility
result and nobody has written it down yet. Put it in a footnote on the results slide; it costs
one line and pre-empts "is that number stable?"

---

## Part 2 — What is broken, newly known, or now stale

### 2.1 🔴 NEW — the run-to-run noise floor is ~6× larger than V5 assumed, and it invalidates V5's effect sizes

Four configurations have been trained twice under nominally identical settings. The replicate
discrepancies:

| Config | Run A gold PR-AUC | Run B gold PR-AUC | Δ |
|---|---:|---:|---:|
| `focal_asymmetric @ 5` | 0.77816 | 0.77811 | 0.00005 |
| `pos_weight @ 50` | 0.7203 | 0.7219 | 0.0016 |
| `focal_asymmetric @ 50` | 0.7961 | 0.8021 | 0.0060 |
| **`pos_weight @ 5`** | **0.8497** | **0.7956** | **0.0541** |

Three of four reproduce to within 0.006. One moved **0.054** — larger than the gap between any
adjacent pair in the eight-row gold ranking, and more than half the entire 0.0992 spread across
all eight configurations.

I checked the obvious explanation and it is not that. Both `pos_weight@5` runs selected the
**epoch-2 checkpoint** on F2, so this is not a checkpoint-selection flip. The divergence is
visible at epoch 1 (validation PR-AUC 0.9841 vs 0.9864), so the two training trajectories
separated early. The other three cells reproduced to ~1e-5 on validation, which rules out a
systematic difference in data, ordering, or configuration. **I cannot attribute the cause** —
the most likely candidate is non-deterministic MPS kernels producing a divergence that a
near-tie somewhere in training then amplified. Say "unattributed", not "seed noise".

Two consequences, and the second is the uncomfortable one:

**(a) Every ranking claim needs an error bar.** Gold differences below ~0.01 are certainly
noise. Differences up to ~0.054 have been *observed* to arise from re-running the same
configuration. The only gold gap in the table that clearly exceeds this is
`focal@15` (0.8211) vs `pos_weight@50` (0.7219) — 0.099, about 1.8× the largest replicate
discrepancy. That one gap is defensible. **The rest of the ranking is not**, and the deck must
not present the eight-row gold ordering as a ranking.

**(b) You have never measured seed variance at all, and the true figure is probably larger.**
No `seed=` is passed to `TrainingArguments` in `src/sota_model.py:171`, so every run in this
project has used the HuggingFace default of **42**. The 0.054 above is *non-determinism at a
fixed seed* — a lower bound on how much the number would move under a genuinely different seed.

This is not a disaster; it is an honest limitation that costs you two sentences and buys real
credibility:

> *"All runs use the HuggingFace default seed (42). Re-training four configurations under
> identical settings produced gold PR-AUC discrepancies of 0.00005–0.054, so we treat gold
> differences below ~0.05 as unresolved and report the weight-response shape rather than a
> ranking. A multi-seed study was out of budget at ~1 GPU-hour per cell."*

Say that, and a grader reads statistical maturity. Present the eight-row ranking without it and
the first question in Q&A takes the result away from you.

### 2.2 🟠 V5's numbers are now stale — five specific things must not reach a slide

The re-run overwrote the CSV, so anything quoted from the previous version is dead. Before you
build the deck, purge:

| Stale claim (V5 / RUNBOOK / CLAUDE.md) | Replacement |
|---|---|
| Gold spread 0.1294, "17×" | **0.0992, 33×** (validation spread also shrank, to 0.0030) |
| `pos_weight@5` gold PR-AUC **0.8497**, best of 4 | **0.7956**, 4th of 8 — this row moved the most |
| "best on validation is worst on gold" | **Spearman ρ = −0.24** across 8 (§1.2) |
| `focal_asymmetric` is "7× less sensitive" | **2× narrower range**; lead with monotone vs non-monotone (§1.4) |
| Noise floor 0.0084, "15× noise" | **Up to 0.054** observed; no effect-size multiplier is safe (§2.1) |

Also note the gold-holdout *winner* changed: it is now `focal_asymmetric @ 15` (0.8211), which
was not gold-scored at all before. Per §2.1 do not call it a winner — its lead over the
deployed config is 0.019, well inside the noise floor.

### 2.3 🟠 The `epochs` column regressed out of the CSV — fixed, but understand why

V5 §2.1 asked for an `epochs` column; commit `e75cda0` added it to
`grid_search_analysis.py:245` and to the committed CSV. The re-run's CSV **does not have it**.

The cause is worth knowing because it will happen again: the grid process started at 23:25:46
on 2026-07-27, *before* the column was committed at ~23:27, and ran for ten hours holding the
old code in memory. When it wrote out at 09:12 it overwrote the fixed CSV with the old schema.
The script on disk is correct and unchanged — `git diff e75cda0 HEAD -- grid_search_analysis.py`
is empty. No code fix is needed.

**I have backfilled the column** (`epochs = 3` for all eight rows, verified from the 900-step
counts in the log). The artifact is now self-describing again. Lesson for the remaining days:
do not edit a script that a long-running job is going to write through.

### 2.4 🟠 Do not switch the deployed configuration — and the argument is now much stronger

V5 recommended against switching on cost-of-rework grounds. The re-run gives you a *principled*
reason instead, which is a better answer in Q&A.

`focal_asymmetric @ 50` ranks **3rd of 8** on gold PR-AUC (0.8021). The leader, `focal@15`,
is ahead by **0.019** — roughly a third of the observed replicate discrepancy for a single
configuration (§2.1). There is no evidence that the leader is actually better. Switching would
cost a full `analysis.py` re-run (2–3h), regeneration of every performance CSV, figure and
error-analysis file, and re-verification of every number already written into the docs — to
chase a difference the data cannot resolve.

The disclosure sentence, updated:

> *"The deployment weight was fixed before the variant grid was run. The grid subsequently
> ranked two other configurations ahead of it on the gold holdout, by margins (0.011 and 0.019
> PR-AUC) smaller than the run-to-run discrepancy we measured for a single configuration
> (up to 0.054), so we report the grid rather than retrofit the deployment to it."*

That reads as discipline *and* as understanding your own error bars.

### 2.5 🟠 Three gold metrics, three different winners — this needs one explanatory sentence

| Metric | Winner | Deployed config's rank |
|---|---|---:|
| Gold PR-AUC (threshold-free) | `focal@15` (0.8211) | 3 / 8 |
| Gold F2 @ th=0.20 | `focal@5` (0.8379) | **8 / 8** |
| Risk cost @ th=0.20 | `focal@5` ($112,200) | **8 / 8** ($225,600) |

The two fixed-threshold metrics agree with each other and disagree with PR-AUC, and the reason
is the same one V5 identified: `focal@5` flags **75.0%** of the holdout at th=0.20, and under a
100:1 cost ratio and an F2 that weights recall 2×, flagging more is nearly free. **Select on
PR-AUC, then re-tune the threshold per model.** A sharp grader will notice that your deployed
configuration is last of eight on two of the three columns; the honest answer — "those two
columns are threshold-dependent and reward over-flagging, which is why we select on the
threshold-free one" — is completely satisfying, but only if you give it first.

### 2.6 🟠 NEW — every configuration over-flags the gold holdout, and th=0.20 does not transfer

Gold flag rates across the eight cells run **0.659 to 0.750**, against a holdout hazard rate of
**46.0%**. Gold precision is 0.57–0.63 for every single configuration. On the validation split
the same models flag 0.186–0.208.

That is not a bug, but it is a fact nobody has stated: **the 0.20 threshold was chosen against
the heuristic label and does not transfer to the gold distribution** — the same models flag
~3.5× more of the gold set proportionally than they do the validation set. It is one more
instance of your central thesis (the heuristic label misrepresents the real problem), it costs
one sentence, and it is the natural lead-in to "and this is why we report PR-AUC."

### 2.7 🟠 Carried over, unchanged

- **Slides: still a 113-line skeleton.** 15% of the grade at zero. Largest risk in the project,
  for the third review running. *(Its stale numbers have since been corrected in place and the
  two new slides specified — but a corrected skeleton is still not a deck.)*
- ✅ **README** — rewritten to a full page on 2026-07-28.
- **LDA is environment-dependent**: quote NMF (K=6, allergen lift 5.28, per-topic NPMI +0.43)
  as the headline topic result. *(The stale 0.716 / 4.18 / 744/342 have since been purged from
  the skeleton and `IMPLEMENTATION_NOTES.md`.)*
- **Hyperparameter provenance**: no `optuna_trials.csv`; `analysis.py` used the pre-fix
  hardcoded defaults. One disclosure sentence remains the right trade.
- **`requirements.txt` and hardcoded W&B** (`src/sota_model.py:187`): a clean clone still
  cannot run. **This is now the single biggest remaining hygiene risk** — a grader who clones
  and runs hits a hard crash at tokenizer load.
- ✅ **`NEGATED_HAZARD` regex and the 23 `unexplained_fn` rows** — both resolved; see §2.8, which
  changed two things you were about to say on a slide.
- ✅ **"hazard base rate" wording** — fixed in `src/data_pipeline.py` and `verify_setup.py`.
- **Three defective holdout rows** (the duplicate pair is confirmed: exactly one duplicate-text
  group in the 772); five review/overview docs still in the repo root.

### 2.8 🟠 NEW — the error-analysis pass refuted the truncation hypothesis and found a taxonomy defect

Two results from the Step 5 work that change what belongs on a slide:

**The 256-token truncation story is wrong. Drop it.** V5 and the runbook both suggested the
unexplained false negatives were reviews that state the hazard in a final sentence beyond the
window. Measured properly — real `DebertaV2TokenizerFast`, real `max_length=256`, offset
mapping, not a word-count proxy — **only 1 of the 23 has its hazard cue past the window**, and
only 2 of 23 exceed 256 tokens at all. The median hazard-cue position is **token 39**. Raising
`max_length` would recover at most one false negative, and that one has an arguable gold label
anyway. A grader who asks "did you check?" and hears the measured version gets a much better
answer than the plausible-sounding guess.

**The residual bucket was our taxonomy's blind spot, not the model's.** 17 of the 23
`unexplained_fn` rows *do* contain an explicit hazard or illness term — but every FN rule in the
taxonomy is conditioned on `not has_explicit`, so a short, low-starred review with a clear hazard
word matches no rule and falls through. That is a *better* slide than the one it replaces:
*"we audited our own error taxonomy and found the unexplained residual was a gap in our rules,
not a mystery in the data."* All 23 are now hand-named in `results/gold_fn_handread.md`, in four
roughly equal modes — `implicit_hazard`, `second_hand_report`, `contamination_novel_phrasing`,
`label_questionable` (5 each) — so "59% unexplained" becomes **0% unexplained**.

Two by-products worth one line each: in 5 cases the only hazard keyword is a **disgust idiom**
("made me want to VOMIT") while the real hazard has no keyword, and the idiom then *suppresses*
the contamination bucket via its own `ILLNESS_WORD` guard. And **5 of 23 gold labels are
arguable** at LLM confidence *high* — honest, self-found evidence that ground truth #2 is
independent of the heuristic but not infallible. Say that before someone else does.

One number moved: the headline FP claim is now **65%** (32.5% `illness_mentioned_not_caused_here`
+ 32.0% `neutral_allergen_mention`), not 66% / 34% + 32%. The re-bucketing moved 9 false
positives. Already corrected in the README and runbook.

---

## Part 3 — What to do next, in order

| # | Task | Time | Why |
|---|---|---|---|
| 1 | **Build the deck.** Correct as you go: 772/355, NMF-first topics, and add the two slides this grid earned — the 33× saturation table and the ρ = −0.24 rank correlation (§1.2). | 4–6h | 15% of the grade at zero. Unchanged as top priority for three reviews. |
| 2 | **Purge the five stale numbers** from `CLAUDE.md`, `RUNBOOK.md` and the slide skeleton (§2.2 table). | 30 min | The committed CSV now contradicts the committed docs. This is the only *incorrectness* on the list. |
| 3 | **Add the noise-floor disclosure** (§2.1) and the four other one-sentence disclosures: threshold non-transfer (§2.6), PR-AUC vs cost (§2.5), deployment weight (§2.4), LDA reproducibility and hyperparameter provenance. | 1h | Converts five Q&A punches into five demonstrations of rigour. Best value-per-minute left. |
| 4 | ✅ README and error-analysis polish — both landed 2026-07-28. Fold §2.8 into the deck: drop the truncation claim, add the taxonomy-blind-spot slide. | done | Turns "59% unexplained" into 0% unexplained, and replaces a wrong hypothesis with a measured one. |
| 5 | Hygiene: `requirements.txt` (**the biggest one — a clean clone crashes at tokenizer load**), `WANDB_MODE`, 3 holdout rows, archive the review docs, add instructor + Tal as collaborators. | 1h | Submission requirements and clean-clone reproducibility. |
| 6 | **Optional, only with a genuinely free night:** re-run 2–3 configurations at a second seed (`seed=1337`) to replace "we observed 0.054" with a measured standard deviation. | ~3h unattended | Turns §2.1's honest caveat into an actual error bar. Nice-to-have; the caveat alone is worth most of the credit. |

**Cut line:** item 1 is still the difference between low-80s and high-80s. Items 2 and 3 are
cheap and are now *required* rather than optional — the docs currently disagree with the data.

**Do not** re-run `analysis.py` to chase `focal@15` (§2.4), and **do not** re-run the topic
model hoping the old LDA numbers return. Both cost hours and buy nothing.

**The story to tell, with the V6 spine:** built a heuristic label → audited it against an
independent judge (85.8% agreement, one-sided 27% over-flagging) → explained *why* it errs,
bucket by bucket → caught the first gold set 89% contaminated → built a verified zero-overlap
holdout (772 rows) → trained eight loss configurations under one clean protocol → **and found
we could not tell them apart on the heuristic label at all (spread 0.0030), while the holdout
separated them by 33× — with a rank correlation between the two of −0.24, meaning our own
label's ordering carried no information** → then measured our own run-to-run noise and found it
large enough to forbid ranking the middle of the table, so we report the *shape* of the weight
response instead: a uniform penalty degrades monotonically with weight, the focal formulation
does not respond to it at all → robustness, not peak performance → and the transformer's
residual false alarms are precisely the label's own inherited blind spots.

Every step of that arc is backed by a committed artifact, and the last two steps are the ones
that separate an 85 from a 92. What is still missing is a deck to say it on.
