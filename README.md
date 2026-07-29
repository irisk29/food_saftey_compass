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
| **Document embeddings (Doc2Vec / LSA / frozen MiniLM)** | Course technique #3 — the middle rung between lexical and contextual, four representations behind one classifier head | [`src/embedding_model.py`](src/embedding_model.py) |
| **TF-IDF + XGBoost** | **Baseline, not a course technique** — the thing the transformer has to beat | [`src/baseline_model.py`](src/baseline_model.py) |

**Document embeddings.** The comparison used to jump straight from sparse lexical to
fine-tuned contextual, which leaves the obvious question unasked: how much of DeBERTa's
advantage is *dense representation* and how much is *fine-tuning*? Four text-only
representations feed one identical class-balanced `LogisticRegression`, so the table varies the
representation and nothing else — TF-IDF (sparse control), **LSA** 300d, **Doc2Vec PV-DBOW**
300d trained on our own corpus, and **frozen `all-MiniLM-L6-v2`** 384d, which is fitted on our
data not at all. Gold-holdout PR-AUC: **LSA 0.757, TF-IDF 0.711, MiniLM 0.709, Doc2Vec 0.684**,
against the committed baseline's 0.728 and DeBERTa's 0.804.

Two findings, and the second is the one that matters:

- **No frozen embedding beats the baseline by a resolvable margin, and none comes near
  DeBERTa.** Only LSA — a linear rotation of the same TF-IDF matrix — is above the baseline at
  all, by 0.029, which is well inside the stratified bootstrap CI on a 772-row holdout
  (widest 95% CI 0.093 wide; [`results/embedding_gold_pr_auc_bootstrap.csv`](results/embedding_gold_pr_auc_bootstrap.csv)).
  Learned paragraph vectors are the *worst* representation here: 4,800 training documents is one
  to three orders of magnitude below what PV-DBOW needs, and the cause is data volume rather than
  corpus mismatch, because LSA sees the identical 4,800 documents and beats it by 0.074.
  **Dense distributed representation is not where the transformer's advantage comes from** — a
  frozen transformer encoder scores 0.709, statistically level with plain TF-IDF, and the
  fine-tuned one scores 0.804. We did not isolate fine-tuning as the sole cause: MiniLM-L6-v2 is
  also 8× smaller than DeBERTa-v3-base (6 layers / 22M vs 12 / 184M) and was pretrained on a
  sentence-similarity objective, so capacity and pretraining objective are confounded with
  freezing. The defensible claim is the negative one: *frozen* dense embeddings do not close
  the gap.
- **The ground-truth gap shrinks monotonically as a representation loses the ability to
  memorise the keyword rule.** Heuristic→gold PR-AUC deltas: Doc2Vec **−0.098**, MiniLM
  **−0.148**, LSA **−0.169**, DeBERTa **−0.183**, TF-IDF+LogReg **−0.213**, XGBoost baseline
  **−0.251**. The sharpest single fact: TF-IDF and frozen MiniLM land on *statistically
  indistinguishable* gold scores (0.711 vs 0.709) from heuristic scores 0.066 apart — so the
  heuristic metric awards TF-IDF 0.066 of PR-AUC for skill that is worth exactly nothing
  out of sample. That is the project's central thesis, reproduced on a new axis with a new
  model family.

See [`results/embedding_technique_comparison.csv`](results/embedding_technique_comparison.csv)
for all three techniques on both ground truths in one table, and `CLAUDE.md` for the full
reading including the caveats.

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
reproduces exactly that confusion. (Precisely: those two modes are 45.8% and 11.4% of the
*rule's* own 201 false flags, 57.2% together — the ordering differs, but both of the model's
dominant error modes are errors its teacher makes.)

**Crossing the two techniques.**
[`analysis/topic_error_integration.py`](analysis/topic_error_integration.py) projects all 772
gold-holdout reviews onto the frozen NMF K=6 topics and reconstructs the deployed model's
predictions from the committed error-detail CSV, so neither model is retrained. The two
techniques succeed on **opposite halves of the hazard vocabulary**: in topic 1 (gluten/coeliac —
the one topic the sweep found cleanly, NPMI +0.43, lift 5.28) the classifier scores precision
**0.304** over 69 alerts; in topic 4 (the diffuse food-poisoning mass the topic model could not
subdivide, lift 1.39) it scores **0.961** at recall 0.980. Per-topic precision correlates with
per-topic base rate at ρ = +0.89, so quote both. The regex taxonomy independently agrees on
where the errors sit: 37 of 63 `neutral_allergen_mention` false positives land in topic 1
(odds ratio 15.9, Fisher p = 7.4e-14, post-hoc). Usable reading: topic assignment is a
zero-label **routing** signal, which is a real job even though hazard-type classification was a
negative result. See [`results/topic_error_integration.csv`](results/topic_error_integration.csv)
and [`results/topic_error_integration_crosstabs.txt`](results/topic_error_integration_crosstabs.txt).

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
- **The embedding comparison does not resolve its own top of table.** LSA's 0.029 lead over the
  XGBoost baseline is smaller than half the bootstrap CI width on a 772-row holdout, so the
  correct statement is "no frozen embedding is *distinguishable* from TF-IDF on gold", not "LSA
  wins". The claims that survive the interval are the large ones: Doc2Vec is genuinely worse
  (−0.074 against LSA) and DeBERTa is genuinely better (+0.047 against LSA, +0.095 against
  Doc2Vec). The logistic head's `C` was left at 1.0 for every variant and never tuned — a
  deliberate choice, since the project already showed validation cannot resolve differences this
  small, but it does mean each representation is reported at an untuned operating point.
  The claims that do survive the interval: Doc2Vec is genuinely worse than LSA (−0.074) and
  DeBERTa is genuinely better than every embedding variant (+0.047 over LSA, +0.095 over frozen
  MiniLM, +0.121 over Doc2Vec).
- **Frozen MiniLM posts the lowest total risk cost of any model in the project** ($175,400 vs
  DeBERTa's $204,850) and this is *not* evidence it is the better model. It flags 81.7% of a
  holdout whose hazard rate is 46%. Under a 100:1 cost ratio over-flagging is nearly free, which
  is the same pathology the loss grid already found in `focal_asymmetric@5`. PR-AUC ranks MiniLM
  0.095 below DeBERTa and is the metric to believe.
- **The enriched dataset contains 3 duplicate review texts** (7,497 unique of 7,500), one of
  which straddles the train/test boundary, so 1 of 1,500 test-split rows shares its text with a
  training row. Found while building the embedding leakage guard. It affects the baseline and
  DeBERTa identically, bounds any test-split metric error at 0.07%, and does not touch the gold
  holdout, whose overlap with training text remains verified at exactly zero.

## 6. Repository map

| Path | Contents |
|---|---|
| [`preprocessing/`](preprocessing) | `final_project_preprocessing.ipynb` — filters Yelp businesses to food/restaurants, streams and keyword-filters reviews, builds the heuristic `is_hazard` label |
| [`postprocessing/`](postprocessing) | `final_project_postprocessing.ipynb` — lexicon/tabular enrichment (medical-lexicon density, VADER negative intensity, negation-window flag); the committed `enriched_allergy_hazard_dataset.csv` and three EDA figures |
| [`src/`](src) | Modelling core: [`data_pipeline.py`](src/data_pipeline.py) (64/16/20 splits), [`features.py`](src/features.py) (enrichment lifted out of the notebook so fresh data can be scored identically), [`baseline_model.py`](src/baseline_model.py), [`sota_model.py`](src/sota_model.py), [`losses.py`](src/losses.py), [`topic_model.py`](src/topic_model.py), [`embedding_model.py`](src/embedding_model.py) |
| [`labeling/`](labeling) | LLM-as-judge pipeline: [`build_holdout_pool.py`](labeling/build_holdout_pool.py), [`create_gold_dataset.py`](labeling/create_gold_dataset.py), and the two gold sets plus the candidate pool |
| [`analysis/`](analysis) | [`evaluation_pipeline.py`](analysis/evaluation_pipeline.py) (both ground truths, cost model, PR and cost curves) and [`error_analysis.py`](analysis/error_analysis.py) |
| [`results/`](results) | Every committed artifact: performance tables, `ground_truth_comparison.csv`, `label_quality.json`, the loss grid, topic-model sweep/topics/lift, error-analysis summaries and detail CSVs, PR and cost curves, run logs |
| [`config/`](config) | [`settings.py`](config/settings.py) — paths, feature allow-list (leakage exclusions documented inline), loss variant, threshold, metric choices |
| [`tests/`](tests) | [`test_losses.py`](tests/test_losses.py) — 9 tests over the three loss formulations; [`test_embedding_model.py`](tests/test_embedding_model.py) — 21 tests over leakage, class-balance parity with the baseline, cross-process determinism and the text-only guarantee |
| [`slides/`](slides) | **[`DECK.md`](slides/DECK.md) — the presentation** (Marp; 18 presented slides + Q&A appendix, every number traced to a file in `results/`), rendered to `deck.pdf` / `deck.html`. [`SLIDES_SKELETON.md`](slides/SLIDES_SKELETON.md) is the superseded planning outline. |
| [`docs/reviews/`](docs/reviews) | Archived external review passes (`PROFESSOR_REVIEW*.md`) and `final_work_overview.md`, moved out of the repo root |

Entry points at the root: [`main.py`](main.py) (Optuna sweep), [`analysis.py`](analysis.py) (final
evaluation, produces every artifact in `results/`),
[`grid_search_analysis.py`](grid_search_analysis.py) (loss-variant grid, resumable),
[`verify_setup.py`](verify_setup.py) (30-second preflight).

## 7. Running it

```bash
pip install -r requirements.txt
export WANDB_MODE=disabled    # optional — omit if you have a W&B account
python verify_setup.py        # preflight — dependencies, data files, holdout integrity

python analysis.py            # final evaluation: both models, both ground truths, all artifacts
python -m src.topic_model     # LDA + NMF sweep

# Document embeddings (course technique #3). CPU-only, ~6 min for all four variants
# including the 3-seed Doc2Vec spread study. Writes only results/*embedding* files
# and overwrites nothing that analysis.py produced.
python -m src.embedding_model --seeds 42 43 44
python -m src.embedding_model --variants tfidf_lr tfidf_lsa doc2vec_dbow   # no network needed

python grid_search_analysis.py    # loss-variant grid (resumable; skips completed cells)
python main.py                # optional Optuna hyperparameter sweep (8-12h on GPU)
pytest tests/                 # loss-function tests
```

Build the presentation (needs Node, no Python):

```bash
npx @marp-team/marp-cli@latest slides/DECK.md -o slides/deck.pdf --allow-local-files
npx @marp-team/marp-cli@latest -s slides/        # live preview at localhost:8080
```

**Both former reproducibility gaps are fixed as of 2026-07-28 — a clean clone now runs:**

- ✅ [`requirements.txt`](requirements.txt) **rewritten.** It previously omitted `sentencepiece`
  and `protobuf` (hard crash at DeBERTa tokenizer load), `accelerate` (`TrainingArguments`
  refuses to construct), `matplotlib`, `seaborn` and `scipy` — and pinned
  `transformers==4.40.0` / `torch==2.2.2`, versions nobody on this project ran. All six missing
  packages are added, and the wrong exact pins are replaced by minimum floors, because *floors
  that install beat pins that are precise and wrong*. The file documents the two-machine reality
  (macOS/MPS for training, Windows/CPU for analysis) which is the known cause of the LDA
  reproducibility caveat in §5.
- ✅ **Weights & Biases is now optional.** `report_to` is resolved by `_report_to()` in
  [`src/sota_model.py`](src/sota_model.py) and `main.py` guards its import and login, so a clone
  with no W&B account degrades to local logging instead of blocking on a prompt. Set
  `WANDB_MODE=disabled` to silence it entirely; the CSVs in `results/` are written either way.

**One provenance caveat, disclosed and made safe:** `results/best_hyperparameters.json` names lr
`1.8346e-05`, while every committed artifact was trained at `1.8140e-05` (bs 16 matches) — a 1.1%
difference, far inside the measured 0.054 gold-PR-AUC run-to-run noise floor. `analysis.py`
therefore **defaults to the as-reported values**, so `python analysis.py` reproduces the committed
CSVs rather than silently overwriting them at a different learning rate. Set
`FSC_USE_SWEPT_HPARAMS=1` to train at the newer swept value instead — it prints a
`PROVENANCE WARNING` and *will* overwrite every performance CSV and figure.

Raw Yelp JSON and `data/` are gitignored; `postprocessing/enriched_allergy_hazard_dataset.csv`,
both gold sets and everything in `results/` are committed, so all reported numbers can be
re-derived without a GPU.
