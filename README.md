# Food Safety Compass

**Detecting food-safety hazards hidden inside Yelp restaurant reviews.**
TAU Text Mining 2026 final project — Iris Kronfeld & Tal Edrehy.

---

## 1. The problem, and the cost asymmetry that shapes everything

A restaurant can hold a 4.5-star average and still have a handful of reviews describing an
allergic reaction, food poisoning, or unsafe handling. Those reviews are the signal a health
inspector or a platform-safety team wants, and star ratings bury them. We frame this as binary
text classification over review text: `is_hazard`.

The two error types are not symmetric. A missed hazard is a public-health liability; a false
alarm is one analyst spending a few minutes reading a review. The evaluation model in
[`analysis/evaluation_pipeline.py`](analysis/evaluation_pipeline.py) prices this at **$5,000 per
missed hazard against $50 per false alarm — a 100:1 ratio**, and that ratio drives every design
choice downstream: a recall-weighted **F2** for checkpoint selection, a **decision threshold
lowered to 0.20**, an asymmetric training loss, and PR-AUC (threshold-free) as the metric we
compare models on.

## 2. The intellectual spine: two ground truths

The dataset is labelled by a heuristic — a severe-hazard keyword match plus a low star rating.
That label is cheap and scales, but **a keyword heuristic cannot be evaluated against itself**:
any text model that learns to recognise the keywords scores near-perfectly while learning
nothing about hazards. So the project carries two independent ground truths and reports both.

1. **The heuristic label**, audited rather than trusted. An independent LLM judge relabelled a
   1,500-row sample ([`labeling/gold_dataset.csv`](labeling/gold_dataset.csv),
   [`results/label_quality.json`](results/label_quality.json)): **85.8% agreement, heuristic
   precision 73.2%, heuristic recall 97.9%**. The disagreement is starkly one-sided — **201
   over-flags against only 12 misses**. This set is 89% training data, so it is valid for judging
   *the rule* and useless for judging *a model*.
2. **A verified zero-overlap gold holdout** —
   [`labeling/gold_dataset_holdout.csv`](labeling/gold_dataset_holdout.csv), 772 LLM-labelled
   reviews drawn by [`labeling/build_holdout_pool.py`](labeling/build_holdout_pool.py) from
   reviews that never entered the pipeline, with text overlap against the training set verified
   to be zero. 355 of the 772 are hazards — a **46.0% hazard rate *within the keyword-screened
   funnel***, not in Yelp at large, where the true rate is plausibly 2–5%.

Every headline model number in this README is measured on ground truth #2.

## 3. Headline results

Gold holdout, from [`results/performance_gold_llm_label_fresh_holdout.csv`](results/performance_gold_llm_label_fresh_holdout.csv)
and [`results/ground_truth_comparison.csv`](results/ground_truth_comparison.csv):

| Model | PR-AUC vs heuristic label | PR-AUC vs gold holdout | Delta |
|---|---:|---:|---:|
| XGBoost + TF-IDF (baseline) | 0.979 | 0.728 | **−0.251** |
| DeBERTa-v3 (deployed, th=0.20) | 0.987 | 0.804 | **−0.183** |

**This table is the main result.** Against the heuristic label the two models look
indistinguishable and both look excellent. Against real ground truth the baseline loses a quarter
of its score: most of what TF-IDF had learned was fidelity to a keyword rule, not hazard
detection. The transformer loses far less, which is the evidence that it learned something the
rule does not encode.

At the deployed operating point (th=0.20) DeBERTa reaches **recall 0.890, precision 0.616,
F2 0.817**, with **39 missed hazards and 197 false alarms — a total risk cost of $204,850**
against the baseline's **116 misses and $585,300, 2.9× higher**.

## 4. Techniques compared

| | Role | Where |
|---|---|---|
| **Fine-tuned DeBERTa-v3-base** | Course technique #1 — transfer learning, with a custom `AsymmetricSafetyLoss` (`focal_asymmetric`, w=50) and th=0.20 | [`src/sota_model.py`](src/sota_model.py), [`src/losses.py`](src/losses.py) |
| **Topic modelling (LDA + NMF)** | Course technique #2 — unsupervised discovery of *hazard types* | [`src/topic_model.py`](src/topic_model.py) |
| **TF-IDF + XGBoost** | **Baseline, not a course technique** — the thing the transformer has to beat | [`src/baseline_model.py`](src/baseline_model.py) |

**The custom loss.** [`src/losses.py`](src/losses.py) implements three formulations behind one
interface: `pos_weight` (plain `BCEWithLogitsLoss(pos_weight=w)`, the honest control),
`focal_asymmetric` (weight scales with `(1−p)^γ`, so the penalty is keyed on the *error* rather
than the label), and `fn_gated` (penalty only where `p < τ`). All three are exercised in
[`tests/test_losses.py`](tests/test_losses.py), including a regression guard asserting an
all-positive model is caught by F2/PR-AUC/flag-rate.

They were then run head to head — 8 configurations at 3 epochs,
[`results/grid_search_loss_variants.csv`](results/grid_search_loss_variants.csv) — and the grid
produced the sharpest methodological finding in the project. **Validation PR-AUC spread across
all 8 configurations: 0.0030. Gold-holdout PR-AUC spread across the same 8: 0.0992 — 33× larger.**
Rank correlation between the two ground truths is **Spearman −0.24**: on the heuristic label the
models are not merely close, their ordering is essentially unrelated to their ordering on real
data. We could not tell our own models apart until we built the holdout. No configuration
collapsed to all-positive at any weight.

**Topic modelling.** Fit on the 1,500 heuristic-flagged hazard reviews, K swept over
{2,3,4,5,6,8,10}, coherence by in-corpus NPMI, K selected by NMI against the LLM-assigned
`hazard_type` (coherence alone just picks the largest K, and NMF at small K is degenerate).
Headline model: **NMF, K=6**. Topic 1 — *gluten, gluten free, celiac, cross contamination, gf* —
has **per-topic NPMI +0.43**, by far the most coherent topic found, and **lift 5.28** for
`allergic_reaction`. Cluster purity is **0.697** (LDA K=4: 0.690, exactly the majority-class
baseline), so type recovery overall is weak — and the *why* is the finding: the technique
isolates the rare, lexically distinct hazard type and cannot subdivide the dominant,
lexically diffuse food-poisoning mass.
See [`results/topic_model_type_lift.csv`](results/topic_model_type_lift.csv),
[`results/topic_model_topics.csv`](results/topic_model_topics.csv),
[`results/topic_model_sweep.csv`](results/topic_model_sweep.csv).

**Error analysis.** [`analysis/error_analysis.py`](analysis/error_analysis.py) applies one
rule-based failure-mode taxonomy to both the labelling rule and the model, which lets errors be
traced across the two. The result: **65% of the model's false positives on the gold holdout fall
into the labelling rule's own top two failure modes** — 32.5% `illness_mentioned_not_caused_here`
plus 32.0% `neutral_allergen_mention`. The model inherited its teacher's blind spots. A keyword
rule cannot represent causation, only co-occurrence, and the transformer trained on that rule
reproduces exactly that confusion.

## 5. Limitations we are stating ourselves

- **The 46.0% hazard rate is inside the keyword-screened funnel**, not the Yelp population rate
  (plausibly 2–5%). Precision figures here are not transferable to unfiltered review streams.
- **The heuristic's 97.9% recall is circular** — measured inside the rule's own keyword filter,
  so it can only miss what the filter already let through. It is optimistic by construction; the
  holdout is the honest estimate.
- **LDA is environment-sensitive.** With identical code, data and seed, LDA fits differ across
  machines and library versions. NMF (deterministic `nndsvda` init) reproduces exactly, which is
  why NMF leads every headline topic claim; LDA numbers are quoted only from the committed
  artifacts in `results/`.
- **The custom loss shows robustness, not superiority.** It did not beat plain class weighting on
  peak gold PR-AUC. What the grid shows is that `focal_asymmetric` is far less sensitive to the
  penalty magnitude than `pos_weight` — which is what a decaying `(1−p)^γ` penalty is supposed to
  do, and is a mechanistic result rather than a win.
- The gold labels come from an LLM judge, not a domain expert, so ground truth #2 is *independent
  of* the heuristic but not infallible.

## 6. Repository map

| Path | Contents |
|---|---|
| [`preprocessing/`](preprocessing) | `final_project_preprocessing.ipynb` — filters Yelp businesses to food/restaurants, streams and keyword-filters reviews, builds the heuristic `is_hazard` label |
| [`postprocessing/`](postprocessing) | `final_project_postprocessing.ipynb` — lexicon/tabular enrichment (medical-lexicon density, VADER negative intensity, negation-window flag); the committed `enriched_allergy_hazard_dataset.csv` and three EDA figures |
| [`src/`](src) | Modelling core: [`data_pipeline.py`](src/data_pipeline.py) (64/16/20 splits), [`features.py`](src/features.py) (enrichment lifted out of the notebook so fresh data can be scored identically), [`baseline_model.py`](src/baseline_model.py), [`sota_model.py`](src/sota_model.py), [`losses.py`](src/losses.py), [`topic_model.py`](src/topic_model.py) |
| [`labeling/`](labeling) | LLM-as-judge pipeline: [`build_holdout_pool.py`](labeling/build_holdout_pool.py), [`create_gold_dataset.py`](labeling/create_gold_dataset.py), and the two gold sets plus the candidate pool |
| [`analysis/`](analysis) | [`evaluation_pipeline.py`](analysis/evaluation_pipeline.py) (both ground truths, cost model, PR and cost curves) and [`error_analysis.py`](analysis/error_analysis.py) |
| [`results/`](results) | Every committed artifact: performance tables, `ground_truth_comparison.csv`, `label_quality.json`, the loss grid, topic-model sweep/topics/lift, error-analysis summaries and detail CSVs, PR and cost curves, run logs |
| [`config/`](config) | [`settings.py`](config/settings.py) — paths, feature allow-list (leakage exclusions documented inline), loss variant, threshold, metric choices |
| [`tests/`](tests) | [`test_losses.py`](tests/test_losses.py) — 9 tests over the three loss formulations |
| [`slides/`](slides) | [`SLIDES_SKELETON.md`](slides/SLIDES_SKELETON.md) — presentation outline, each claim traced to a file in `results/` |

Entry points at the root: [`main.py`](main.py) (Optuna sweep), [`analysis.py`](analysis.py) (final
evaluation, produces every artifact in `results/`),
[`grid_search_analysis.py`](grid_search_analysis.py) (loss-variant grid, resumable),
[`verify_setup.py`](verify_setup.py) (30-second preflight).

## 7. Running it

```bash
pip install -r requirements.txt
export WANDB_MODE=offline     # required: see the caveats below
python verify_setup.py        # preflight — dependencies, data files, holdout integrity

python analysis.py            # final evaluation: both models, both ground truths, all artifacts
python -m src.topic_model     # LDA + NMF sweep
python grid_search_analysis.py    # loss-variant grid (resumable; skips completed cells)
python main.py                # optional Optuna hyperparameter sweep (8-12h on GPU)
pytest tests/                 # loss-function tests
```

**Two known reproducibility gaps, documented rather than papered over:**

- [`requirements.txt`](requirements.txt) is incomplete and stale. It is **missing
  `sentencepiece`** (hard crash at DeBERTa tokenizer load), **`matplotlib`**, **`seaborn`** and
  **`accelerate`** (`TrainingArguments` will refuse to construct), and it pins
  `transformers==4.40.0` / `torch==2.2.2`, versions nobody on this project actually runs — the
  code was developed and verified against transformers 4.57. Install the four missing packages
  manually and ignore the pins.
- **Weights & Biases is a hard dependency**: `report_to="wandb"` is hardcoded at
  [`src/sota_model.py:187`](src/sota_model.py). Without `WANDB_MODE=offline` the run will block
  on a login prompt.

Raw Yelp JSON and `data/` are gitignored; `postprocessing/enriched_allergy_hazard_dataset.csv`,
both gold sets and everything in `results/` are committed, so all reported numbers can be
re-derived without a GPU.
