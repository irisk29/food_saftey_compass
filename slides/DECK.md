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
<!-- STRUCTURE: 22 rendered sections =                                      -->
<!--   slides 1-20  PRESENTED  (title, hook, 17 content slides, thank-you)  -->
<!--   slides 21-22 APPENDIX   (Q&A back-pocket, figure + rehearsal notes)  -->
<!--                           -> NOT presented; delete before submitting   -->
<!--                              the deck if you want a clean 20.          -->
<!--                                                                       -->
<!-- TIMING: 17 content slides at ~50 s = 14.2 min, +30 s for title/hook/   -->
<!-- close = ~14.7 min. That is tight against the 15-min cap, so slide 18   -->
<!-- (Limitations) is now the DEFAULT cut, not a contingency: every item on  -->
<!-- it also lives on its home slide. Cutting it lands at ~13.8 min.         -->
<!-- Slides 5-7 and 9-12 plus 15 and 17 are the differentiators — do NOT    -->
<!-- rush them.                                                             -->
<!--                                                                       -->
<!-- EVERY number on every slide traces to a file in results/ or labeling/. -->
<!-- The "Evidence:" line names it. Verified against artifacts 2026-07-29,   -->
<!-- including the three techniques, the bootstrap intervals and the         -->
<!-- topic x error integration added that day.                              -->
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
- Hazard reviews are rare enough that reaching 1,500 required **streaming and keyword-filtering
  the corpus** — this is *why* a funnel exists at all, and why every gold rate in this deck is a
  *funnel* rate rather than a population rate

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
| Secondhand · hypothetical · unpleasant-not-unsafe · other | 6.5% | *"my friend said she felt off"* |

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

- Deployed: `focal_asymmetric @ w=50`, threshold **0.20** — lowered from 0.50 *because of* the
  100:1 asymmetry, but **not read off it.** ⚠️ Taken literally, a 100:1 ratio puts the
  cost-minimising threshold at **0.01** — and that is exactly what our pipeline measured on both
  evaluation sets. A 0.01 threshold means *flag essentially everything*, which is useless.
  **We reject our own cost model's optimum** — and that is a second, independent reason cost is
  not our selection metric.
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

Precision 0.200 **is** the base rate. It flagged every single review, at **every epoch** — its F2
sat at exactly **0.5556**, the analytic value of an all-positive answer at a 20% base rate.

### Which objective would have picked it?

| `recall` → | `pr_auc` → | `f2` / `f1` → |
|---|---|---|
| <span class="bad">**trial 1. The broken one.**</span> | <span class="win">trial 0; trial 1 ranks **last** (0.196 vs 0.988)</span> | <span class="win">trial 2; also rejects trial 1</span> |

- ⚠️ **Be precise about what did and did not recover.** The *hard predictions* never recovered —
  all-positive at every epoch. The *ranking* partially did: trial 1's best-epoch PR-AUC reached
  0.643, against 0.988 for the two stable trials. Still last, either way.
- ⚠️ **Attribution — and the limit of what we can claim.** The loss weight was w=50 in *all three*
  trials, and the 8-cell grid shows `collapsed = False` at w=50 in **both** formulations, so the
  weight alone is **not sufficient** to cause collapse. But collapse appeared only at the high
  learning rate with batch 4, and we have **no low-weight / high-lr run**, so we *cannot separate
  the learning rate from a learning-rate × weight interaction* — and we don't claim to.
- <span class="big">Any metric you can maximise with a constant answer is not a safety metric.</span>

<span class="ev">Evidence: results/optuna_trials.csv · results/best_hyperparameters.json · results/grid_search_loss_variants.csv</span>

---

## THE payoff — same models, two ground truths

Gold holdout = 772 rows that never touched training, selection, or tuning.

| Model | PR-AUC (heuristic) | PR-AUC (**gold**) | Δ |
|---|---:|---:|---:|
| Baseline: TF-IDF + XGBoost **+ metadata** | 0.9785 | 0.7279 | <span class="bad">**−0.2506**</span> |
| **DeBERTa-v3** (deployed, th=0.20) — **text only** | 0.9874 | **0.8045** | −0.1830 |

<span class="small">Note the handicap runs *against* us: the baseline gets TF-IDF **plus all 7 tabular
features**; the transformer sees raw text alone. The advantaged model is the one that loses.</span>

### Read the two columns against each other — this is the whole slide

- On **our own label**, the bag-of-words baseline is within **0.009** of a transformer. It looks
  like the transformer was a waste of a GPU.
- On **honest ground truth**, DeBERTa leads by **0.077** — an 8× wider gap.
- The baseline **falls further** (−0.251 vs −0.183, **1.37×**) because **TF-IDF partially
  re-learns the keyword rule.** It was never doing the task; it was reverse-engineering our
  regex.

**The flawed label did not merely inflate scores — it inverted the comparison between techniques.**

### With error bars, because 772 rows is not infinite

Deployed operating point on gold, 20,000-resample bootstrap: recall **0.890 [0.857, 0.922]**,
precision 0.616 [0.574, 0.658], F2 **0.817 [0.787, 0.845]**, **39 missed [28, 51]**, risk cost
**$204,850 [$148,900, $264,950]** vs the baseline's $585,300. So "89% recall" is an *estimate*,
±3 points — not a specification.

- The comparison itself is **paired** on the same resampled rows: **−29 missed hazards
  [−44, −15]**, exact McNemar **p = 1.5e-4**. That is the right test — overlapping marginal
  intervals are compatible with a difference that is reliably one-signed.
- ⚠️ **Two limits we state ourselves.** The paired test runs against a **text-only TF-IDF control**,
  because the XGBoost baseline's per-row errors were never persisted — so the headline "116 vs 39
  missed" has a point estimate and **no interval**. And **PR-AUC has no interval at all**: no
  checkpoint is committed and `results/` stores probabilities only for error rows.
- ⚠️ These intervals hold the **trained model fixed**. They *compose* with the up-to-**0.054**
  training non-determinism on slide 12 — quoting either alone understates total uncertainty.

<span class="ev">Evidence: results/performance_gold_llm_label_fresh_holdout.csv · results/performance_heuristic_label_test_split.csv · results/bootstrap_ci_gold_llm_label_fresh_holdout.{csv,md} · results/bootstrap_ci_paired_*.csv · results/bootstrap_ci_mcnemar_*.csv</span>

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

The spread ratio is the claim that carries: our label had **almost no resolution** — 0.0030 of
spread to distinguish eight models.

The rank correlation adds that there is **no detectable relationship** between the two orderings,
and if anything it points mildly the wrong way. ⚠️ **We state the limit ourselves:** at n=8,
p=0.57, this shows we *cannot demonstrate* the validation ranking carries information — it is not
proof that it carries none. Either way it is nowhere near enough to select a model on.

> *We could not tell our own models apart on our own label — and the ordering it did give us showed
> no relationship to the ordering on real data.*

<span class="ev">Evidence: results/grid_search_loss_variants.csv (8 rows) · results/grid_ground_truth_agreement.csv (ρ, r, spreads) · results/grid_search.log</span>

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
  exactly monotone ordering of 4 points in the direction theory predicts has probability 1/24 ≈ **4%** (one-sided, pre-specified). A uniform penalty on
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
- ⚠️ Own the limit: that topic is 5.28× *enriched* for allergen hazards but **captures only 7 of
  the 38** such documents. High precision on a narrow slice, not broad coverage.

### ❌ It cannot subdivide the common, lexically-diffuse one

- `food_poisoning` is **376 / 545 = 69%** of the validated set and stays one undifferentiated mass
- Overall purity **0.697** (NMF K=6) vs a majority-class baseline of **0.690** — essentially no gain
- NMI 0.08–0.12 against a shuffle null of 0.005–0.014: real, but weak

<span class="big">Topic models find the hazard with its own vocabulary, and fail on the hazard that shares everyone else's.</span>

That is a **property of the technique** — distributional co-occurrence needs distinctive words —
not a tuning failure. "Got sick" looks lexically identical to "bad service".

⚠️ **LDA is environment-sensitive**: identical code, data and seed produced different fits across
two machines. We lead with **NMF** (deterministic `nndsvda` init, reproduces exactly) and quote
LDA only from committed artifacts. We caught this ourselves and moved the headline accordingly.

<span class="ev">Evidence: results/topic_model_type_lift.csv · topic_model_topics.csv · topic_model_crosstabs.txt</span>

---

## Technique 3 — Document embeddings: *is it the representation, or the fine-tuning?*

Slide 10 showed a transformer beating a bag of words on honest ground truth. **Why?** Four
**text-only** representations, one identical class-balanced logistic head, same splits, same
metrics, same cost model:

| Representation | Gold PR-AUC | 95% CI |
|---|---:|---:|
| **DeBERTa-v3 — *fine-tuned*** | **0.8045** | — |
| LSA 300d (TF-IDF → SVD) | 0.7574 | [0.724, 0.793] |
| <span class="small">Baseline: TF-IDF + XGBoost + metadata</span> | 0.7279 | — |
| TF-IDF + LogReg (sparse control) | 0.7112 | [0.667, 0.754] |
| **MiniLM-L6-v2 — *frozen*** | 0.7092 | [0.666, 0.750] |
| Doc2Vec PV-DBOW 300d | 0.6831 | [0.639, 0.732] |

- **No frozen embedding is distinguishable from a bag of words.** The only variant above the
  baseline is **LSA — a linear rotation of the very same TF-IDF matrix** — by 0.029, comfortably
  inside the CI.
- A **frozen transformer encoder sits at 0.709, level with TF-IDF at 0.711**, while the
  **fine-tuned** one reaches **0.804**.

<span class="big">Dense distributed representation is not the source of the transformer's advantage.</span>

- ⚠️ **We do not upgrade that to "the advantage is fine-tuning."** MiniLM differs from DeBERTa in
  freezing, **capacity** (6 layers / 22M vs 12 / 184M) *and* pretraining objective simultaneously —
  not a clean ablation. A **frozen DeBERTa encoder + linear head** would settle it in one CPU run.
- **Doc2Vec's failure is attributed by control, not guessed:** LSA sees the *identical* 4,800
  documents and beats it by 0.074. A learned embedding losing to a linear projection of the same
  data is a **data-volume** verdict, not a domain one.

<span class="ev">Evidence: src/embedding_model.py · results/embedding_technique_comparison.csv · results/performance_embedding_*.csv · results/embedding_gold_pr_auc_bootstrap.csv · 21 unit tests in tests/test_embedding_model.py</span>

---

## Where the *model* still fails — and it inherits its teacher's blind spots

DeBERTa on the gold holdout: **197 false positives, 39 false negatives.** Same taxonomy as the label-audit slide.

| False positive mode | Share |
|---|---:|
| **Illness mentioned, not caused here** | **32.5%** |
| **Neutral allergen mention** | **32.0%** |
| Generic complaint (11.2%) + residual (11.2%) | 22.3% |
| Negated hazard | 8.6% |
| Secondhand · hyperbole · unpleasant-not-unsafe · other | 4.6% |

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

### And it is a *dose–response*, across all five models we trained

| DeBERTa *(fine-tuned)* | TF-IDF | MiniLM *(frozen)* | LSA | Doc2Vec |
|---:|---:|---:|---:|---:|
| <span class="win">**64.5%**</span> | 69.1% | 71.8% | 75.9% | <span class="bad">**76.5%**</span> |

Share of each model's false positives landing in those same two label failure modes. **The only
model that partially escapes its teacher is the only one allowed to update its representation.**
With a mechanism: `neutral_allergen_mention` is 45.1% of TF-IDF's false positives but **62.1% of
LSA's** — in dense space *"great gluten-free menu"* and *"the gluten made me ill"* become
neighbours, so the geometry **destroys a distinction sparse features retain**.

<span class="ev">Evidence: results/error_analysis_deberta_gold_*.md · results/gold_fn_handread.md · results/embedding_vs_deberta_fp_modes.csv · analysis/rebucket_errors.py</span>

---

## Crossing the two techniques — where each one is trustworthy

The topic model and the classifier were built separately. Projecting all **772 holdout rows** onto
the **frozen NMF K=6 topics** shows they succeed on **opposite halves of the vocabulary**.

| NMF topic | rows | gold hazard rate | alerts | **precision** | share of the 197 FPs |
|---|---:|---:|---:|---:|---:|
| **1 — gluten / coeliac** <span class="small">(NPMI +0.43, lift 5.28)</span> | 202 | 12.9% | 69 | <span class="bad">**0.304**</span> | <span class="bad">24.4%</span> |
| **4 — food poisoning** <span class="small">(lift 1.39, the diffuse mass)</span> | 104 | 96.2% | 102 | <span class="win">**0.961**</span> | 2.0% |

<span class="big">The one topic we found cleanly is where the classifier is least trustworthy — and the mass we failed to subdivide is where it is near-perfect.</span>

- **Two independent instruments agree**, which is what makes this more than a coincidence: 37 of 63
  `neutral_allergen_mention` false positives land in topic 1 (**OR 15.9, Fisher p = 7.4e-14**), and
  53 of 64 `illness_mentioned_not_caused_here` land in topics 0 and 5 with **none in topic 4**
  (OR 3.8, p = 2.2e-4). Both survive Bonferroni over all 54 cells.
- ⚠️ **Own the confound:** per-topic precision tracks the topic's base rate at **ρ = +0.89**, so it
  measures how far the *vocabulary* settles the question, not model skill alone. Recall is far
  flatter (spread 0.19 vs 0.66) — coverage is much less vocabulary-dependent than trustworthiness.
  Post-hoc, single run.

### The business payoff — a *zero-label* routing signal

Of **513 alerts**: the **102 in topic 4 are right 96% of the time** → escalate on light review.
The **69 in topic 1 are right 30% of the time** → send to a human. Delivered by the technique whose
headline evaluation was a **negative** result.

<span class="ev">Evidence: analysis/topic_error_integration.py · results/topic_error_integration.csv · results/topic_error_integration_crosstabs.txt</span>

---

## Limitations we report rather than hide

| # | Limitation | Our disclosure |
|---:|---|---|
| 1 | **Run-to-run noise up to 0.054 gold PR-AUC**, single seed (42) | Report weight-response *shape*, not a ranking; multi-seed study out of budget (~1 GPU-hr/cell) |
| 2 | **Deployed config is not the grid leader** — 3rd of 8; 0.019 behind `focal@15`, 0.008 behind `pos_weight@1` | Weight was fixed *before* the grid ran; both margins are inside our own noise floor, so we report the grid rather than retrofit the deployment |
| 3 | **Three gold metrics, three winners.** Deployed config is 3/8 on PR-AUC but **8/8 on F2** and 7/8 on cost | F2 and cost are computed at a **fixed** th=0.20 and reward over-flagging (`focal@5` flags **75%** of the holdout). We select on the **threshold-free** metric. |
| 4 | **th=0.20 does not transfer.** Gold flag rate 0.659–0.750 vs a 46.0% base rate; gold precision 0.57–0.63 | Threshold was tuned against the heuristic label. Same models flag 18.6–20.8% on validation. **One more instance of the central thesis** — re-tune per distribution. |
| 5 | **Hyperparameters**: 3-trial sweep only; reported system trained at lr 1.814e-05, sweep best 1.835e-05 | A 1.1% difference, far inside the 0.054 noise floor. `analysis.py` now warns on the mismatch. |
| 6 | **LDA irreproducible** across machines; **46.0% is a funnel rate**; `fn_gated` never run | All three stated on their own slides, unprompted |
| 7 | **No interval on PR-AUC, and none on the headline baseline gap** | No checkpoint is committed and only error-row probabilities are stored, so PR-AUC stays a point estimate; the baseline's per-row errors were never persisted, so "116 vs 39 missed" cannot be paired-tested. The bootstrap we *do* have holds the trained model fixed and **composes** with row 1's 0.054. |
| 8 | **A *frozen* MiniLM matches the fine-tune on recall** — +7 missed [−7, +21], p = 0.39 | Fine-tuning wins on **precision** (+0.104 [+0.078, +0.131]) and on the operating point as a whole, not on recall. It flags **81.7%** of the holdout to get there. We reworded our own claim rather than keep the flattering one. |
| 9 | **Two defects we found in our own tooling** | `gensim`'s `infer_vector` ignores the model's `hashfxn`, so inference was never reproducible (0.6825 / 0.6833 / 0.6836 across interpreters) — patched, and the guard now spawns **subprocesses** because the in-process test passed the whole time the defect was live. And 3 duplicate texts in the enriched CSV, one pair straddling train/test: 1 of 1,500 test rows, bounding the error at **0.07%**. Gold-holdout overlap remains exactly **0**. |

<!--
SPEAKER: This slide is the DEFAULT CUT at 17 content slides — every item also lives on its
home slide, and dropping it lands the deck at ~13.8 min. Present it only if you are ahead of
time; as a block it reads as maturity. Ten seconds per row, do not read verbatim.
-->

---

## Conclusions

### The methodology arc — this is what we are actually submitting

built a heuristic label → **audited it** (85.8% agreement, 27% one-sided over-flagging) →
explained *why* it errs, bucket by bucket → **caught our gold set 89% contaminated** → built a
verified zero-overlap holdout (772 rows) → trained 8 loss configurations under one protocol →
**found we could not tell them apart on our own label at all** (spread 0.0030 vs 0.0992, ρ = −0.24)
→ **measured our own run-to-run noise** and let it forbid ranking → reported *shape*, not
leaderboard → **isolated where the transformer's advantage does *not* come from** (four frozen
representations, none separable from a bag of words) → showed the model's residual false alarms
**are the label's own inherited blind spots**, as a dose–response across all five models → **crossed
the two techniques** to say which hazard vocabularies the classifier can be trusted on → put
**error bars** on the headline

### Answers to the two research questions

- **① Detection: yes, and the honest number is lower than the flattering one.** Gold PR-AUC
  **0.804** vs 0.987 against our own label. Recall **0.890 [0.857, 0.922]** at 100:1 cost,
  **$204,850** risk cost against the baseline's **$585,300**, and **−29 missed hazards [−44, −15]**
  against a text-only control. What the transformer buys is **precision at a given recall** — a
  frozen encoder matches its recall and pays 81.7% flag rate for it.
- **② Discovery: partially, and the failure is informative.** Allergen/celiac hazards separate
  cleanly and unsupervised (lift **5.28**, NPMI **+0.43**); the food-poisoning mass (69%) does not.
  The negative result then became **operationally useful**: the topics predict *where the classifier
  is trustworthy* (precision 0.961 vs 0.304).

### Business recommendation

Deploy as a **screening funnel**, not a verdict — threshold **informed by** the 100:1 asymmetry but
capped well above its literal optimum of 0.01, human triage downstream, the over-flagging rate
quoted honestly to whoever staffs that triage, and **alerts routed by topic** so the 96%-precision
stream and the 30%-precision stream get different amounts of human attention.

**Future work:** frozen-DeBERTa ablation to isolate fine-tuning · multi-seed error bars ·
per-row probabilities persisted so PR-AUC gets an interval · token-attribution XAI · ordinal risk
tiers · cross-platform transfer

---

<!-- _class: lead -->

# Thank you

### Questions?

<span class="small">Every number in this deck traces to a committed artifact in <code>results/</code>.<br>
Repo: code · notebooks · <b>30 unit tests</b> · 3 course techniques · 8-cell loss grid · 3-trial sweep ·
two gold sets · full error taxonomies · bootstrap intervals</span>

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
We measured it rather than estimating: 20,000-resample bootstrap gives recall **0.890
[0.857, 0.922]**, **39 missed [28, 51]**, precision 0.616 [0.574, 0.658]. So ±3 points on recall —
adequate for every claim we make, and labelling is resumable. Two caveats we volunteer: **PR-AUC has
no interval** (no checkpoint committed, only error-row probabilities stored), and the interval holds
the trained model **fixed**, so it composes with the 0.054 training non-determinism. One
duplicate-text pair and 5 sub-`high`-confidence rows are documented in
`results/holdout_integrity.md` and deliberately **not** repaired — every committed number is
computed on these exact 772 rows.

**6. "Why no word/document embeddings?"**
We have them — technique #3, `src/embedding_model.py`. Four text-only representations behind one
identical class-balanced logistic head, and **the result is negative, which is why it is worth
presenting**: no frozen embedding is distinguishable from a bag of words on gold (TF-IDF 0.711,
frozen MiniLM 0.709, Doc2Vec 0.683), and the only variant above the XGBoost baseline is **LSA — a
linear rotation of the same TF-IDF matrix** — by 0.029, inside the CI. A frozen transformer encoder
is level with TF-IDF while the fine-tuned one reaches 0.804, so **dense representation is not where
the advantage comes from**. We stop short of "therefore it's fine-tuning": MiniLM differs in
freezing, capacity and pretraining objective at once. Doc2Vec's failure *is* attributed — LSA sees
the identical 4,800 documents and beats it by 0.074, so it is data volume, not domain.

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
Same models flag 18.6–20.8% on validation. One more instance of the central thesis.

**10. "Did the collapse come from your w=50 loss?"**
Not on its own — the weight is demonstrably **not sufficient**: the 8-cell grid runs w=50 in both
formulations at the tuned learning rate and `collapsed=False` in all 8 rows (`pos_weight@50`: flag
rate 0.197, recall 0.921, precision 0.936). But we will not claim more than that. All three sweep
trials used `focal_asymmetric@50`, so there is no within-sweep contrast on the loss, and we never
ran a low-weight/high-lr cell — so a learning-rate × weight *interaction* is not excluded, and it is
mechanistically plausible (a ×50 positive-class weight amplifies gradients exactly where a high lr
with batch 4 is already unstable).

There is a **third data point**, and it adds the axis that makes sense of the other two. Early in
the project we retrained both towers directly on a small LLM-labelled file — **905 train / 227
test** — with ω=50 held fixed (`asymetic_loss_atenuation.py`, DeBERTa-v3-**small**). There the model
**did** collapse outright: recall 100%, precision **37.4%** — exactly that split's base rate —
PR-AUC 0.580, and 142 false positives, i.e. *every* negative row. The documented fix was to dial ω
to 5.0 and **anneal it 1.0 → 5.0 across epochs**, which restored a boundary (recall 90.59%,
precision 72.64% on that split). ⚠️ **Label this exploratory when you say it:** that 1,132-row CSV
is not among the committed datasets, so the evidence is the console logs plus the committed script,
and its numbers are **not** comparable to the 772-row holdout figures.

Read all three together and the honest statement gets *stronger*, not weaker: **ω=50 collapses at
905 training rows, does not collapse at 4,800, and collapses again at 4,800 when the learning rate
is raised — so collapse is an interaction between penalty magnitude, data volume and optimiser step
size, not a property of the weight.** What we still have not done is isolate the weight's individual
contribution: no low-weight cell was ever run at low volume or at the high learning rate. One extra
cell (`pos_weight@1` at lr 3.8e-05, batch 4) would settle it.

**11. "Your baseline nearly matches DeBERTa on your own label — was the transformer worth it?"**
That is exactly the trap the gold set exists to expose. On the heuristic label the gap is 0.009;
on gold it is 0.077. TF-IDF was re-learning our regex, not doing the task.

**12. "A frozen sentence encoder gets your recall for free — why fine-tune?"**
The sharpest question available, and the answer is measured. On **missed hazards a frozen MiniLM is
not separable** from our fine-tune (+7 [−7, +21], exact McNemar p = 0.39) and misses *fewer* at the
point estimate, 32 vs 39. What separates them is the other side of the ledger: it flags **81.7%** of
the holdout to get there against our 66.5%, and the fine-tune wins on **precision** (+0.104
[+0.078, +0.131]), on F1 (+0.073 [+0.046, +0.100]) and on overall correctness (McNemar p = 6.6e-13).
So the defensible claim is that **fine-tuning buys precision at a given recall**, not recall itself —
and we reworded our own slide rather than keep the flattering version.

**13. "Your two techniques never touch each other."**
They do now — `analysis/topic_error_integration.py`, slide 17. Projecting all 772 holdout rows onto
the frozen NMF topics shows the classifier is **least** trustworthy on the one topic the sweep found
cleanly (gluten/coeliac, precision 0.304, 24.4% of all false positives) and **near-perfect** on the
diffuse mass the topic model could not subdivide (food poisoning, precision 0.961). Two independent
instruments agree: 37 of 63 `neutral_allergen_mention` false positives sit in that gluten topic
(Fisher p = 7.4e-14, Bonferroni-safe over 54 cells). Confound owned: per-topic precision tracks base
rate at ρ = +0.89, so it measures how far the vocabulary settles the question, not skill alone.

---

# APPENDIX (not presented) — Figure export checklist

- [ ] **Slide 5** — 2×2 confusion table, heuristic vs LLM (549 / 201 / 12 / 738)
- [ ] **Slide 6** — horizontal bar chart of the 201 FP buckets
- [ ] **Slide 7** — contamination diagram: 1,334/1,500 overlap → 0/772 clean
- [ ] **Slide 9** — 3-trial table with trial 1 highlighted red (flag rate 1.000)
- [ ] **Slide 10** — grouped bars: heuristic vs gold PR-AUC, baseline vs DeBERTa (the payoff figure)
- [ ] **Slide 11** — dot plot: 8 configs on validation (clustered) vs gold (spread) — shows 33× visually
- [ ] **Slide 12** — weight-response curve, `pos_weight` monotone vs `focal` flat, with a ±0.054 noise band
- [ ] **Slide 14** — lift table heat-strip (5.28 highlighted)
- [ ] **Slide 15** — embedding ladder: 6 models on gold PR-AUC with CI whiskers, fine-tuned bar
      separated from the four frozen ones (the "representation is not the answer" figure)
- [ ] **Slide 16** — FP bucket comparison: rule vs model, side by side (the "inherited blind spots"
      figure), with the 5-model dose–response strip beneath it
- [ ] **Slide 17** — topic × precision plot: 6 NMF topics, precision against base rate, topics 1 and
      4 labelled (the "crossing the techniques" figure)
- [ ] Reuse existing: `results/pr_curve_gold_*.png`, `results/cost_curve_gold_*.png`,
      `results/topic_model_selection.png`, `results/pr_curve_embedding_gold_*.png`

# Rehearsal notes

- Target **14:00** to leave buffer, which means **cutting slide 18 (Limitations) by default** —
  17 content slides is already ~14.2 min. Present it only if you are ahead.
- Differentiators: **5–7** (the two audits), **9–12** (metric, payoff, saturation, weight response),
  **15** (the negative embedding result) and **17** (crossing the techniques). Do not rush these.
- ⚠️ **Never cut slides 15, 16 or 17** — they carry three of the four sentences below, and 15 is the
  only slide that shows a third course technique.
- Open with the one-sentence version. Close on the arc, not on a metric.
- The four sentences that earn the top band, say them slowly:
  1. *"We could not tell our own models apart on our own label."*
  2. *"Bare recall would have selected the broken model — here it is."*
  3. *"65% of the model's false alarms are the label's own blind spots."*
  4. *"The one topic we found cleanly is where our classifier is least trustworthy."*
- If asked why the transformer wins, the precise answer is **precision at a given recall** — not
  recall. A frozen encoder matches our recall (Q&A 12). Say it before someone finds it.
