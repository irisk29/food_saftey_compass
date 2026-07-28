---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Allergen & Food Safety Hazard Compass · Text Mining 2026'
style: |
  section { font-size: 22px; }
  section.lead { text-align: center; }
  h1 { color: #8a1c1c; font-size: 40px; }
  h2 { color: #8a1c1c; font-size: 30px; }
  table { font-size: 18px; margin: 0 auto; }
  th { background: #f3e9e9; }
  .small { font-size: 17px; color: #555; }
  .ev { font-size: 15px; color: #777; font-style: italic; }
  .big { font-size: 34px; color: #8a1c1c; font-weight: bold; }
  .win { color: #0b6b3a; font-weight: bold; }
  .bad { color: #b00; font-weight: bold; }
---

<!-- ===================================================================== -->
<!-- BUILD NOTE (delete before presenting):                                -->
<!--   Render:  npx @marp-team/marp-cli@latest slides/DECK.md -o deck.pdf   -->
<!--           npx @marp-team/marp-cli@latest slides/DECK.md -o deck.pptx  -->
<!--   Present: npx @marp-team/marp-cli@latest -s slides/                  -->
<!--                                                                       -->
<!-- STRUCTURE: 20 rendered sections =                                      -->
<!--   slides 1-18  PRESENTED  (title, hook, 15 content slides, thank-you)  -->
<!--   slides 19-20 APPENDIX   (Q&A back-pocket, figure + rehearsal notes)  -->
<!--                           -> NOT presented; delete before submitting   -->
<!--                              the deck if you want a clean 18.          -->
<!--                                                                       -->
<!-- TIMING: 15 content slides at ~55 s = 13.8 min, +30 s for title/hook/   -->
<!-- close = ~14.3 min. Slides 5-7 and 9-12 are the differentiators — do    -->
<!-- NOT rush them. If running long: drop slide 16 (Limitations; every item -->
<!-- also lives on its home slide) and compress 13 to 30 s.                 -->
<!--                                                                       -->
<!-- EVERY number on every slide traces to a file in results/ or labeling/. -->
<!-- The "Evidence:" line names it. Verified against artifacts 2026-07-28.  -->
<!-- ===================================================================== -->

<!-- _class: lead -->

# Allergen & Food Safety Hazard Compass

### Detecting health hazards hidden in Yelp restaurant reviews

**Iris Kronfeld · Tal Edrehy**

Text Mining 2026 — Prof. Shay Palachy Affek · TA: Tal Cordova

---

<!-- _class: lead -->

## The one-sentence version

# We built a hazard detector — then spent the project proving our own ground truth wrong, **twice**, before trusting a single number.

<!--
SPEAKER: Say this verbatim and pause. It frames every slide that follows.
The project's contribution is not the classifier — it's the audit trail.
-->

---

## The problem: aggregate rating masking

- A **4.5★ restaurant can hide a systemic kitchen failure.** 98% of reviewers love the
  ambiance; 2% end up in the ER. The average hides the tail — and the tail is the safety signal.
- For allergy and celiac diners, a star average makes every meal an invisible gamble.

### The cost asymmetry drives every design choice downstream

| | Cost | Why |
|---|---:|---|
| **False negative** — missed hazard | **$5,000** | ER visit, litigation, an unwarned diner |
| **False positive** — false alarm | **$50** | ~10 minutes of human triage |

<span class="big">100 : 1</span> — so we optimise **recall-weighted** metrics (F2), deploy a
**lowered threshold** (0.20), and train a **cost-sensitive loss**.

**Two research questions:** ① Can we *detect* hazard reviews from raw text?
② Can we *discover hazard types* without labels?

<span class="ev">Evidence: COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE in analysis/evaluation_pipeline.py — the 100:1 ratio reproduces every cost figure in results/</span>

---

## Data & the weak label we started with

- Yelp Academic Dataset → food/restaurant businesses only, reviews 5–800 words
- Streamed and keyword-filtered until **1,500 flagged + 6,000 benign = 7,500 rows**
- Natural hazard sparsity is brutal: **~0.12%** (24 hazards in the first 20,000 reviews) —
  this is *why* a keyword funnel exists at all

### The heuristic label

> `is_hazard = 1` ⟺ safety keyword (allergy, celiac, contamination, food poisoning, …)
> **AND** stars ≤ 3

- Split 64 / 16 / 20 → train 4,800 · **validation 1,200** · test 1,500
- Checkpoint selection and the hyperparameter objective see **only validation**, so test-split
  numbers are out-of-selection
- Metadata features engineered, then **deliberately withheld** from the transformer

**Closing beat:** *a rule-based label is a hypothesis, not ground truth — so we tested it.*

<span class="ev">Evidence: preprocessing/ + postprocessing/ notebooks · src/data_pipeline.py · config/settings.py</span>

---

## Act I — We audited our own label

An **independent LLM judge** re-labelled 1,500 rows blind, with no access to the rule.

|  | LLM: hazard | LLM: benign | |
|---|---:|---:|---|
| **Rule: hazard** | 549 | <span class="bad">201</span> | 750 |
| **Rule: benign** | 12 | 738 | 750 |
|  | 561 | 939 | **1,500** |

- Agreement **85.8%** — but the disagreement is **almost entirely one-sided**
- <span class="bad">**201 / 750 = 27% of the rule's positives are benign**</span>; only 12 hazards missed
- Heuristic precision **73.2%**, recall **97.9%** — and that recall is **circular**: it is
  measured *inside the rule's own keyword filter*. Nothing outside the funnel can be counted.

<!--
SPEAKER: The asymmetry is the point. The rule is not noisy — it is BIASED, and in one
direction. That is diagnosable, which is what the next slide does.
-->

<span class="ev">Evidence: results/label_quality.json · labeling/gold_dataset.csv (1,500 rows) · labeling/create_gold_dataset.py</span>

---

## *Why* the rule over-flags — a failure taxonomy

We bucketed all **201 false positives** by cause. This is a rule-based taxonomy, not hand-waving.

| Failure mode | Share | Real example |
|---|---:|---|
| **Illness mentioned, not caused here** | **45.8%** | *"haven't gotten sick from them in 10 years"* |
| Residual / unexplained | 21.4% | — |
| **Negated hazard** | **14.9%** | *"nothing here to trigger my allergy"* |
| **Neutral allergen mention** | **11.4%** | *"great gluten-free options"* |
| Secondhand, hypothetical, unpleasant-not-unsafe | 6.5% | *"my friend said she felt off"* |

### The mechanism, in one line

<span class="big">Co-occurrence is not causation.</span>

The rule matches a keyword and a low star rating. It cannot represent **negation**, **attribution**,
or **agency** — the exact three things bag-of-words methods cannot represent either. Remember this
when we get to the model's own errors.

<span class="ev">Evidence: results/error_analysis_heuristic_label.md + _summary.csv · analysis/error_analysis.py</span>

---

## Act II — Our *evaluation* set was contaminated too

We built a gold set. Then we checked it against the training split.

- <span class="bad">**1,334 / 1,500 rows (89%) of the first gold set sat in the training split.**</span>
  Evaluating on it would have measured **memorisation**, and it would have looked *excellent*.
- We caught this ourselves, before reporting a single model number.

### The fix — a genuinely fresh holdout

- Rebuilt a candidate pool **straight from raw Yelp**, identical normalisation
  (`labeling/build_holdout_pool.py`)
- **Verified 0 / 772 text overlap** with all training data — the check re-runs in the
  `verify_setup.py` preflight, so it cannot silently rot
- LLM-labelled: **772 rows · 355 hazards · 46.0%**

⚠️ **Say the caveat before anyone asks:** 46.0% is a **funnel** rate — 100% of these rows passed
the keyword screen. Population prevalence is ~2–5%. Every gold number in this deck is
"performance *within* the screened funnel".

<span class="ev">Evidence: labeling/gold_dataset_holdout.csv · results/holdout_integrity.md · verify_setup.py</span>

---

## Technique 1 — Fine-tuned DeBERTa-v3 + a cost-sensitive loss *family*

**`microsoft/deberta-v3-base`**, text only. Metadata withheld on purpose: `stars`,
`medical_lexicon_density` and `negation_window_flag` all derive from the labelling rule, so
feeding them back would be **direct label leakage**.

### We did not build one loss — we built three, so they could be compared

| Variant | Formulation | Honest description |
|---|---|---|
| `pos_weight` | `BCEWithLogits(pos_weight=w)` | **This is standard class weighting.** We say so. |
| `focal_asymmetric` | `1 + (w−1)(1−p)^γ`, γ=2 | Penalty scales with the **error**, decays as confidence rises |
| `fn_gated` | penalty only where `p < τ` | Implemented + unit-tested, **never run** — we say that too |

- Deployed: `focal_asymmetric @ w=50`, threshold **0.20** (from the 100:1 cost model)
- **9 unit tests**, including a regression guard asserting an all-positive model is caught by
  F2 / PR-AUC / flag-rate

<span class="ev">Evidence: src/losses.py · src/sota_model.py · tests/test_losses.py</span>

---

## Act III — The metric that selects a broken model

Our hyperparameter sweep produced a **degenerate model**, and it is on disk.

**Optuna trial 1** — lr = 3.80e-05, batch = 4, loss held at `focal_asymmetric@50`:

| | flag rate | recall | precision | PR-AUC |
|---|---:|---:|---:|---:|
| Trial 0 (lr 1.83e-05, bs 16) | 0.196 | 0.913 | 0.932 | <span class="win">0.9877</span> |
| Trial 2 (lr 1.29e-05, bs 8) | 0.203 | 0.938 | 0.926 | 0.9876 |
| **Trial 1** (lr 3.80e-05, bs 4) | <span class="bad">1.000</span> | <span class="bad">1.000</span> | <span class="bad">0.200</span> | <span class="bad">0.196</span> |

Precision 0.200 **is** the base rate. It flagged every single review.

### Which objective would have picked it?

| `recall` → | `pr_auc` → | `f2` / `f1` → |
|---|---|---|
| <span class="bad">**trial 1. The broken one.**</span> | <span class="win">trial 0; ranks trial 1 **last, by 5×**</span> | <span class="win">trial 2; also rejects trial 1</span> |

- ⚠️ **Attribution matters:** the loss weight was w=50 in *all three* trials. The 8-cell grid shows
  `collapsed = False` at w=50 in **both** formulations. So this is **optimiser instability at high
  learning rate / tiny batch — not a pathology of our loss.**
- <span class="big">Any metric you can maximise with a constant answer is not a safety metric.</span>

<span class="ev">Evidence: results/optuna_trials.csv · results/best_hyperparameters.json · results/grid_search_loss_variants.csv</span>

---

## THE payoff — same models, two ground truths

Gold holdout = 772 rows that never touched training, selection, or tuning.

| Model | PR-AUC (heuristic) | PR-AUC (**gold**) | Δ |
|---|---:|---:|---:|
| Baseline: TF-IDF + XGBoost | 0.9785 | 0.7279 | <span class="bad">**−0.2506**</span> |
| **DeBERTa-v3** (deployed, th=0.20) | 0.9874 | **0.8045** | −0.1830 |

### Read the two columns against each other — this is the whole slide

- On **our own label**, the bag-of-words baseline is within **0.009** of a transformer. It looks
  like the transformer was a waste of a GPU.
- On **honest ground truth**, DeBERTa leads by **0.077** — an 8× wider gap.
- The baseline **falls further** (−0.251 vs −0.183, **1.37×**) because **TF-IDF partially
  re-learns the keyword rule.** It was never doing the task; it was reverse-engineering our
  regex.

**The flawed label did not merely inflate scores — it inverted the comparison between techniques.**

Deployed operating point on gold: recall **0.890**, precision 0.616, F2 **0.817**, 39 FN / 197 FP,
risk cost **$204,850** vs the baseline's **$585,300** (**2.9× cheaper**).

<span class="ev">Evidence: results/performance_gold_llm_label_fresh_holdout.csv · results/performance_heuristic_label_test_split.csv · results/ground_truth_comparison.csv</span>

---

## Why we needed a gold set at all: the heuristic metric is *saturated*

We trained **8 configurations** — `{pos_weight, focal_asymmetric}` × `{w = 1, 5, 15, 50}` —
under one identical protocol (3 epochs, 900 optimiser steps each, verified from the log).

| Ground truth | PR-AUC range across 8 configs | Spread |
|---|---|---:|
| Heuristic label (validation) | 0.9864 – 0.9894 | **0.0030** |
| Gold label (fresh holdout) | 0.7219 – 0.8211 | **0.0992** |

<span class="big">33× wider on real data.</span>

### The sharper statistic — rank correlation between the two ground truths

<span class="big">Spearman ρ = −0.24</span> &nbsp;&nbsp;(p = 0.57; Pearson r = −0.36)

The spread ratio says our label had **low resolution**. The rank correlation says it had
**no information** — knowing a configuration's validation rank tells you *nothing* about its
real-world rank, and if anything points mildly the wrong way.

> *We could not tell our own models apart on our own label — and the ordering it did give us was
> uncorrelated with the ordering on real data.*

<span class="ev">Evidence: results/grid_search_loss_variants.csv (8 rows) · results/grid_search.log</span>

---

## What the custom loss actually buys — shape, not a ranking

Gold PR-AUC as the false-negative penalty grows:

| Weight | `pos_weight` | `focal_asymmetric` |
|---:|---:|---:|
| 1 | 0.8096 | 0.7913 |
| 5 | 0.7956 | 0.7781 |
| 15 | 0.7905 | **0.8211** |
| 50 | <span class="bad">0.7219</span> | 0.8021 |
| **Range** | **0.0877** | **0.0430** |

- `pos_weight` declines **monotonically** across all four weights. Under pure noise, drawing an
  exactly monotone ordering of 4 points has probability 1/24 ≈ **4%**. A uniform penalty on
  *every* positive, cranked to 50, over-penalises and measurably damages out-of-sample ranking.
- `focal_asymmetric` is **non-monotone** in a band **half as wide** — because its penalty decays
  as `(1−p)^γ`, so once the model is confident, the weight stops mattering. **That is what it was
  designed to do.**

<span class="big">The custom loss buys insensitivity to a hyperparameter, not peak performance.</span>

### 🔴 And here is our own error bar

Four configurations were trained **twice** under nominally identical settings. Three reproduced
to within 0.006 gold PR-AUC. **One moved 0.054** — more than half the entire between-configuration
spread. Both runs picked the same checkpoint, so it is not a selection flip; the trajectories
diverged by epoch 1 and the cause is **unattributed** (likely non-deterministic MPS kernels).
All runs use the HuggingFace default **seed 42**, so 0.054 is non-determinism at a *fixed* seed —
a **lower bound** on true seed variance.

**Therefore: we treat gold gaps below ~0.05 as unresolved and report the *shape*, not a ranking.**

<span class="ev">Evidence: results/grid_search_loss_variants.csv · replicate pairs documented in CLAUDE.md</span>

---

## Technique 2 — Topic modelling for hazard-type discovery

**LDA + NMF** over the 1,500 flagged reviews. Shared vocabulary, domain stopwords,
K ∈ {2,3,4,5,6,8,10}.

### Selecting K honestly required rejecting our first criterion

- **Coherence alone is untrustworthy here.** NMF's highest NPMI is at **K=2** — a degenerate fit
  where one topic holds **95% of documents**. NMF K ∈ {2,3,4,5} *all* have a topic holding >60%.
- So: **exclude degenerate fits first**, then report both the coherence-selected and the
  externally-validated K when they disagree (NMF: coherence K=8, validation K=6).
- **External validation is the stable criterion:** `nmi_above_null` picks **LDA K=4** and
  **NMF K=6** — the reported models.

⚠️ Disclose in one breath: K was selected using the LLM hazard types (**mild circularity** — we
report it), and ~27% of the fit corpus is benign **per our own Act I audit**.

<span class="ev">Evidence: src/topic_model.py · results/topic_model_sweep.csv · results/topic_model_selection.png</span>

---

## What topic modelling can and cannot find

### ✅ It finds the rare, lexically-distinct hazard

**NMF K=6, topic 1:** *gluten · gluten free · celiac · cross contamination · gf · celiac disease*

- **Lift 5.28** for `allergic_reaction` — the strongest association in the study
- **Per-topic NPMI +0.43** — by far the most coherent topic found (the worst scores **−0.22**)

### ❌ It cannot subdivide the common, lexically-diffuse one

- `food_poisoning` is **376 / 545 = 69%** of the validated set and stays one undifferentiated mass
- Overall purity **0.697** (NMF K=6) vs a majority-class baseline of **0.690** — essentially no gain
- NMI 0.09–0.12 against a shuffle null of 0.006–0.014: real, but weak

<span class="big">Topic models find the hazard with its own vocabulary, and fail on the hazard that shares everyone else's.</span>

That is a **property of the technique** — distributional co-occurrence needs distinctive words —
not a tuning failure. "Got sick" looks lexically identical to "bad service".

⚠️ **LDA is environment-sensitive**: identical code, data and seed produced different fits across
two machines. We lead with **NMF** (deterministic `nndsvda` init, reproduces exactly) and quote
LDA only from committed artifacts. We caught this ourselves and moved the headline accordingly.

<span class="ev">Evidence: results/topic_model_type_lift.csv · topic_model_topics.csv · topic_model_crosstabs.txt</span>

---

## Where the *model* still fails — and it inherits its teacher's blind spots

DeBERTa on the gold holdout: **197 false positives, 39 false negatives.** Same taxonomy as Slide 5.

| False positive mode | Share |
|---|---:|
| **Illness mentioned, not caused here** | **32.5%** |
| **Neutral allergen mention** | **32.0%** |
| Generic complaint (11.2%) + residual (11.2%) | 22.3% |
| Negated hazard | 8.6% |
| Secondhand · hyperbole · unpleasant-not-unsafe | 4.6% |

<span class="big">65% of the model's false alarms sit in the labelling rule's own top two failure modes.</span>

The transformer did not invent new errors. **It learned our rule's blind spots** — the strongest
analytical result in the project, because it traces a model failure back to a *label* pathology.

### We also audited our own error taxonomy — and found the defect was ours

- 23 of 39 FNs first landed in `unexplained_fn` (**59%**). Cause: every FN rule was conditioned on
  `not has_explicit`, so a short low-starred review with a clear hazard word matched **nothing**.
- All 23 **hand-read and named** → **0% unexplained**: 5 `implicit_hazard`, 5 `second_hand_report`,
  5 `contamination_novel_phrasing`, 5 `label_questionable`, 2 `explicit_hazard_missed`, 1 hedged.
- ❌ **We tested the obvious hypothesis and it was wrong.** "The hazard is past the 256-token
  window" — measured with the real tokeniser and offset mapping: **1 of 23**, only 2 of 23 exceed
  256 tokens, **median cue position token 39**. Raising `max_length` would recover ~one FN.
- 🔍 **5 of 23 gold labels are arguable** at LLM confidence *high*. Ground truth #2 is
  *independent*, not infallible — we say it before anyone else does.

<span class="ev">Evidence: results/error_analysis_deberta_gold_*.md · results/gold_fn_handread.md · analysis/rebucket_errors.py</span>

---

## Limitations we report rather than hide

| # | Limitation | Our disclosure |
|---:|---|---|
| 1 | **Run-to-run noise up to 0.054 gold PR-AUC**, single seed (42) | Report weight-response *shape*, not a ranking; multi-seed study out of budget (~1 GPU-hr/cell) |
| 2 | **Deployed config is not the grid leader** — 3rd of 8; 0.019 behind `focal@15`, 0.008 behind `pos_weight@1` | Weight was fixed *before* the grid ran; both margins are inside our own noise floor, so we report the grid rather than retrofit the deployment |
| 3 | **Three gold metrics, three winners.** Deployed config is 3/8 on PR-AUC but **8/8 on F2** and 7/8 on cost | F2 and cost are computed at a **fixed** th=0.20 and reward over-flagging (`focal@5` flags **75%** of the holdout). We select on the **threshold-free** metric. |
| 4 | **th=0.20 does not transfer.** Gold flag rate 0.659–0.750 vs a 46.0% base rate; gold precision 0.57–0.63 | Threshold was tuned against the heuristic label. Same models flag 19–21% on validation. **One more instance of the central thesis** — re-tune per distribution. |
| 5 | **Hyperparameters**: 3-trial sweep only; reported system trained at lr 1.814e-05, sweep best 1.835e-05 | A 1.1% difference, far inside the 0.054 noise floor. `analysis.py` now warns on the mismatch. |
| 6 | **LDA irreproducible** across machines; **46.0% is a funnel rate**; `fn_gated` never run | All three stated on their own slides, unprompted |

<!--
SPEAKER: This slide is optional if time is short — every item also lives on its home slide.
But presenting it as a block reads as maturity. Ten seconds per row, do not read verbatim.
-->

---

## Conclusions

### The methodology arc — this is what we are actually submitting

built a heuristic label → **audited it** (85.8% agreement, 27% one-sided over-flagging) →
explained *why* it errs, bucket by bucket → **caught our gold set 89% contaminated** → built a
verified zero-overlap holdout (772 rows) → trained 8 loss configurations under one protocol →
**found we could not tell them apart on our own label at all** (spread 0.0030 vs 0.0992, ρ = −0.24)
→ **measured our own run-to-run noise** and let it forbid ranking → reported *shape*, not
leaderboard → showed the model's residual false alarms **are the label's own inherited blind spots**

### Answers to the two research questions

- **① Detection: yes, and the honest number is lower than the flattering one.** Gold PR-AUC
  **0.804** vs 0.987 against our own label. Recall 0.890 at 100:1 cost, **$204,850** risk cost
  against the baseline's **$585,300**.
- **② Discovery: partially, and the failure is informative.** Allergen/celiac hazards separate
  cleanly and unsupervised (lift **5.28**, NPMI **+0.43**); the food-poisoning mass (69%) does not.

### Business recommendation

Deploy as a **screening funnel**, not a verdict — threshold set by the 100:1 cost model, human
triage downstream, with the over-flagging rate quoted honestly to whoever staffs that triage.

**Future work:** token-attribution XAI · ordinal risk tiers · multi-seed error bars · cross-platform transfer

---

<!-- _class: lead -->

# Thank you

### Questions?

<span class="small">Every number in this deck traces to a committed artifact in <code>results/</code>.<br>
Repo: code · notebooks · 9 unit tests · 8-cell loss grid · 3-trial sweep · two gold sets · full error taxonomies</span>

---

# APPENDIX (not presented) — Q&A back-pocket

**1. "Is your asymmetric loss just class weighting?"**
The `pos_weight` variant **is**, exactly, and we say so on the slide. `focal_asymmetric` is
error-dependent. The grid compares both across 4 weights (8 cells). `fn_gated` is implemented and
unit-tested but **never run** — we say that rather than imply coverage.

**2. "46% hazard rate — really?"**
Funnel rate; 100% of the holdout passed the keyword screen. The screen is two-tier, and the tiers
behave very differently: **strong-term tier (n=378) 71.2% hazard**, **weak-term tier + stars≤3
(n=394) 21.8%**. So the 46.0% is a mixture of a high-precision and a low-precision funnel, not a
uniform rate. Population prevalence ~2–5%.

**3. "Why w=50 when your cost ratio implies 100:1?"**
The weight was fixed before the grid ran. The grid then ranked two configs ahead of it on gold —
`focal@15` by **0.019** and `pos_weight@1` by **0.008** — both **far smaller than our measured
run-to-run discrepancy (up to 0.054)**.
So we report the grid rather than retrofit the deployment to noise. Re-running `analysis.py` to
chase 0.019 would cost 2–3 h and regenerate every artifact to buy a difference the data cannot resolve.

**4. "Did you use the test set for selection?"**
No. Checkpoint selection and the Optuna objective ran on the **validation** split, so heuristic
test-split numbers are out-of-selection; the gold holdout touched **nothing** — not training, not
selection, not tuning.

**5. "Only 772 holdout rows?"**
Recall CI ≈ ±3–4 points at n_pos = 355. Adequate for the claims made, and labelling is resumable.
One duplicate-text pair and 5 sub-`high`-confidence rows are documented in
`results/holdout_integrity.md` and deliberately **not** repaired — every committed number is
computed on these exact 772 rows.

**6. "Why no word/document embeddings?"**
Chose topic modelling as technique #2 because it answers research question ②. Static embeddings
answer neither question better than a fine-tuned contextual transformer already does.

**7. "How stable are these numbers?"**
All runs use HF default seed 42. Re-training four configurations under identical settings moved
gold PR-AUC by 0.00005–0.054, so we treat gaps below ~0.05 as unresolved. **The deployed config
was trained 3× across 2 independent code paths: 0.7961 / 0.8021 / 0.8045 — range 0.0084.**

**8. "Why do PR-AUC and your cost column disagree?"**
Cost and F2 are computed at a **fixed** th=0.20; PR-AUC is threshold-free. A model that simply
flags more (`focal@5`: 75% of the holdout) looks cheap under a 100:1 ratio. We select on PR-AUC
and would re-tune the threshold per model.

**9. "Flag rate 66–75% on gold but base rate 46% — isn't the model over-flagging?"**
Yes, and knowably. The 0.20 threshold was tuned against the heuristic label and does not transfer.
Same models flag 19–21% on validation. One more instance of the central thesis.

**10. "Did the collapse come from your w=50 loss?"**
No — and we checked. All three Optuna trials used `focal_asymmetric@50`; only the high-lr/small-batch
trial collapsed. The 8-cell grid shows `collapsed=False` at w=50 in **both** formulations
(`pos_weight@50`: flag rate 0.197, recall 0.921, precision 0.936). It is optimiser instability.

**11. "Your baseline nearly matches DeBERTa on your own label — was the transformer worth it?"**
That is exactly the trap the gold set exists to expose. On the heuristic label the gap is 0.009;
on gold it is 0.077. TF-IDF was re-learning our regex, not doing the task.

---

# APPENDIX (not presented) — Figure export checklist

- [ ] **S4** — 2×2 confusion table, heuristic vs LLM (549 / 201 / 12 / 738)
- [ ] **S5** — horizontal bar chart of the 201 FP buckets
- [ ] **S6** — contamination diagram: 1,334/1,500 overlap → 0/772 clean
- [ ] **S8** — 3-trial table with trial 1 highlighted red (flag rate 1.000)
- [ ] **S9** — grouped bars: heuristic vs gold PR-AUC, baseline vs DeBERTa (the payoff figure)
- [ ] **S10** — dot plot: 8 configs on validation (clustered) vs gold (spread) — shows 33× visually
- [ ] **S11** — weight-response curve, `pos_weight` monotone vs `focal` flat, with a ±0.054 noise band
- [ ] **S13** — lift table heat-strip (5.28 highlighted)
- [ ] **S15** — FP bucket comparison: rule vs model, side by side (the "inherited blind spots" figure)
- [ ] Reuse existing: `results/pr_curve_gold_*.png`, `results/cost_curve_gold_*.png`, `results/topic_model_selection.png`

# Rehearsal notes

- Target **14:00** to leave buffer. Slides 4–6 and 8–11 are the differentiators — do not rush them.
- If long: drop slide 15 (every item lives on its home slide) and compress 12 to 30 s.
- Open with the one-sentence version. Close on the arc, not on a metric.
- The three sentences that earn the top band, say them slowly:
  1. *"We could not tell our own models apart on our own label."*
  2. *"Bare recall would have selected the broken model — here it is."*
  3. *"65% of the model's false alarms are the label's own blind spots."*
