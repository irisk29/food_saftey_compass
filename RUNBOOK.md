# Runbook — what to do now

Updated **2026-07-28**, after the full 8-cell grid re-run. Every GPU step in the original
pre-execution guide has now been run, so this is a plan for the time that is left.

**Deadline: 2026-08-02 midnight. Presentations: 2026-08-03. You have ~4.5 days.**

The headline: **all modelling is done — there is no GPU work left on the critical path, and
the results are good.** What remains is a presentation that does not exist yet, a pass of
corrections, and repository hygiene. Nothing on the critical path requires new research.

---

## Where the project actually stands

| Piece | State |
|---|---|
| Holdout gold set | ✅ 772 rows, 355 hazard (46.0% within the keyword funnel), 767 high-confidence, committed |
| Topic modelling | ✅ Run and committed; NPMI bug fixed. One caveat — see Step 4 |
| Final evaluation (`analysis.py`) | ✅ Run 2026-07-27; every performance artifact exists |
| Model error analysis | ✅ Both ground truths, bucketed |
| Loss-variant grid | ✅ Complete; 8 cells, all 3-epoch, **all 8 gold-scored**. Produced the saturation finding (Step 1) |
| Optuna sweep (`main.py`) | ❌ Never run under the fixed config (Step 6 — optional) |
| **Presentation** | ❌ **Skeleton only. 15% of the grade, and the largest remaining risk** |
| README | ❌ Still one line |
| `requirements.txt` / W&B | ❌ Still cannot reproduce from a clean clone (Step 7) |

Your results, for reference while you build slides — all verified against the committed
CSVs:

| | vs heuristic label | vs gold holdout | delta |
|---|---:|---:|---:|
| XGBoost baseline — PR-AUC | 0.979 | 0.728 | **−0.251** |
| DeBERTa-v3 — PR-AUC | 0.987 | 0.804 | **−0.183** |
| DeBERTa @0.20 — recall | 0.947 | 0.890 | −0.057 |

DeBERTa on the holdout: 39 missed hazards vs the baseline's 116, and a total risk cost of
$204,850 vs $585,300 — **2.9× lower**.

---

## Step 1 — ✅ DONE AND COMPLETE: the full loss grid ran (2026-07-28, commit `5fc0d88`)

**No GPU work is required any more.** All eight cells were re-trained from scratch under
`--force`, **all at 3 epochs** (900 optimiser steps, verified from `results/grid_search.log`),
and **all eight are now scored on the gold holdout**. The mixed 2-/3-epoch problem is gone and
every number from the previous grid is superseded. The result:

| Ground truth | Configs | PR-AUC spread |
|---|---:|---:|
| Heuristic keyword label (validation) | 8 | **0.0030** |
| Gold LLM label (holdout) | 8 | **0.0992** |

The heuristic label cannot tell your models apart; the holdout separates them by **33×**. The
sharper statistic, now computable with eight points: the **rank correlation between the two
ground truths is Spearman ρ = −0.24**. The validation ordering is not just low-resolution — it
carries no information about out-of-sample ordering. That is the headline.

Nothing collapsed at any weight in either formulation (`collapsed=False` in all 8 rows), so the
old "w=50 goes all-positive" claim is positively refuted, not merely unevidenced.

**⚠️ Three things you must carry into the deck, because they change what you may claim:**

1. **The noise floor is ~0.054, not 0.008.** Four configurations have been trained twice under
   identical settings; three reproduced to within 0.006 gold PR-AUC, one (`pos_weight@5`) moved
   **0.054**. Do not present the eight-row gold ordering as a ranking — only the top-to-bottom
   gap (0.099) clearly exceeds noise. And note no `seed=` is ever set, so every run uses the HF
   default 42; that 0.054 is non-determinism at a *fixed* seed, a lower bound.
2. **Report the shape, not a winner.** `pos_weight` declines **monotonically** across
   w = 1, 5, 15, 50 (0.8096 → 0.7956 → 0.7905 → 0.7219, range 0.088); `focal_asymmetric` is
   **non-monotone** and wanders in a band half as wide (0.7913, 0.7781, 0.8211, 0.8021, range
   0.043). Robustness, not peak performance. Do **not** quote the old "7× less sensitive", and do
   not quote the endpoint-only "8×" either.
3. **Your headline number reproduces.** `focal_asymmetric @ w=50` has been trained three times
   across two code paths: gold PR-AUC 0.7961, 0.8021, 0.8045. Put that in a footnote — it
   pre-empts "is that number stable?" for one line of text.

See `PROFESSOR_REVIEW_V6.md` for the full reading, including the list of V5 numbers that are now
stale and must not reach a slide.

---

## Step 2 — Build the presentation (start today, in parallel with Step 1)

**This is now the biggest risk in the project.** 15% of the grade, presentations are the
day after the deadline, and only `slides/SLIDES_SKELETON.md` exists. The skeleton is good
— it has the spine, per-slide evidence lines, and timing — but it is not a deck.

Nothing in it is blocked on data any more, and **the skeleton's stale numbers have now been
corrected in place** (2026-07-28): 772/355 and 46.0% "within the keyword-screened funnel", NMF
first with purity 0.697 / lift 5.28 / NPMI +0.43 and the LDA reproducibility caveat, the refuted
w=50 collapse deleted from Slide 8, and the Q&A section extended with the noise-floor,
PR-AUC-vs-cost and over-flagging answers.

**What is left is the actual build.** Two new slides are specified in the skeleton's checklist
and are the ones this grid earned — the saturation slide (0.0030 vs 0.0992, 33×, ρ = −0.24) and
the weight-response slide (monotone `pos_weight` vs flat `focal_asymmetric`). Use the real
classifier numbers from the table above, never the smoke-test figures.

The spine still holds, and it is a strong one: *we built a hazard detector, then spent the
project proving our own ground truth wrong — twice — before trusting a single number.*

---

## Step 3 — ✅ DONE: README (2026-07-28)

Rewritten from one line to a full page: problem and cost asymmetry, the two-ground-truths
design and *why* it exists, the headline table, the technique comparison, a self-stated
limitations section, a repo map, and how to run — including honest documentation of the two
reproducibility gaps that Step 7 still has to fix. Every number in it was recomputed from a
committed artifact rather than copied from a doc.

---

## Step 4 — Corrections pass (~1 hour, pure Q&A insurance)

Each of these is a question you can take away from a grader for free. The first three are
**new as of the full grid re-run** and are the highest value ones — the first is the only item
on this list that fixes an actual incorrectness rather than adding a caveat.

- **🔴 Purge the stale V5 grid numbers.** The re-run overwrote the CSV, so anything quoted from
  the old four-row table is now wrong. Dead: gold spread **0.1294** and "17×" (now 0.0992 and
  **33×**); `pos_weight@5` gold PR-AUC **0.8497** (now 0.7956, and it is no longer the best
  cell — `focal_asymmetric@15` at 0.8211 is); "best on validation is worst on gold" (use
  **Spearman ρ = −0.24** instead); "**7× less sensitive**" (use the 2× range ratio and the
  monotone/non-monotone contrast); noise floor **0.0084** and "15× noise" (see below). Check
  `CLAUDE.md`, `RUNBOOK.md` and `slides/SLIDES_SKELETON.md` — the first two are already done.
- **🔴 Own the noise floor.** One sentence, and it is worth more than any other sentence
  available: *"All runs use the HuggingFace default seed (42). Re-training four configurations
  under identical settings produced gold PR-AUC discrepancies of 0.00005–0.054, so we treat gold
  differences below ~0.05 as unresolved and report the weight-response shape rather than a
  ranking. A multi-seed study was out of budget at ~1 GPU-hour per cell."* Present the eight-row
  gold ordering *without* this and the first Q&A question takes the result away from you.
- **Threshold non-transfer.** The 0.20 threshold was tuned against the heuristic label and does
  not transfer: on gold, every configuration flags 0.659–0.750 of the holdout against a 46.0%
  funnel hazard rate, with precision 0.57–0.63 — while the same models flag only 0.186–0.208 on
  validation. One sentence, and it is another instance of your central thesis.
- **PR-AUC vs cost disagree, and say why first.** Gold F2@0.20 and risk cost@0.20 both crown
  `focal_asymmetric@5`, which flags **75%** of the holdout; under a 100:1 cost ratio
  over-flagging is nearly free. Your deployed config is last of eight on both of those columns
  and 3rd of eight on PR-AUC. Select on the threshold-free metric, re-tune the threshold per
  model, and say so before a grader notices.
- **Deployment weight.** *"The deployment weight was fixed before the variant grid was run. The
  grid subsequently ranked two other configurations ahead of it on the gold holdout, by margins
  (0.008 and 0.019 PR-AUC) smaller than the run-to-run discrepancy we measured for a single
  configuration (up to 0.054), so we report the grid rather than retrofit the deployment to it."*
- **LDA is not reproducible across machines.** Re-running the fixed topic sweep in a
  second environment with the same seed produced different LDA fits (allergen lift 4.18 vs
  the committed 2.52; purity 0.716 vs 0.690; coherence peak K=4 vs K=8). NMF reproduced
  exactly. Quote LDA numbers only from the committed artifacts, lead with NMF, and put one
  sentence in the write-up owning it — "we caught our own topic model being irreproducible
  and moved the headline claims to the algorithm that reproduces" reads as rigour, not
  weakness. Do **not** re-run the topic model expecting the old numbers back.
- ✅ **Hyperparameter provenance — RESOLVED 2026-07-28** (`441c436`). `results/optuna_trials.csv`
  and `results/best_hyperparameters.json` are now committed: a 3-trial sweep, objective =
  validation PR-AUC, loss fixed at `focal_asymmetric@50`, lr ∈ [1e-5, 5e-5] log, batch ∈ {4,8,16}.
  Two things came out of it, and the second is a *result*, not housekeeping:
  - **Provenance, with one caveat to disclose.** Best params: lr **1.8346e-05**, bs 16. The
    committed artifacts were trained at lr **1.8140e-05** (bs 16 matches) — a 1.1% relative
    difference, far inside the 0.054 gold-PR-AUC noise floor. But `analysis.py` prefers the JSON,
    so a re-run now trains at the new rate and will **not** reproduce the committed CSVs. It
    prints a PROVENANCE WARNING on mismatch; `FSC_USE_REPORTED_HPARAMS=1` pins the as-reported
    values. Disclosure sentence: *"hyperparameters come from a 3-trial PR-AUC sweep
    (`results/optuna_trials.csv`); the reported system was trained at lr 1.814e-05 from an
    earlier sweep of the same space, a 1.1% difference well inside our measured run-to-run
    noise."*
  - 🔴 **The sweep caught a collapse, and bare recall would have selected it.** Trial 1
    (lr=3.80e-05, bs=4) went all-positive: flag rate **1.000**, recall **1.000**, precision
    **0.200** (= the base rate), PR-AUC **0.196**. Ranked by candidate objective: **recall picks
    trial 1**; PR-AUC puts it last by 5×; F2 and F1 also reject it. The metric decision argued a
    priori since V3 is now **measured**. Attribute the collapse to lr/batch, *not* to w=50 — the
    grid shows `collapsed=False` at w=50 in both formulations. This belongs on Slide 8 and
    replaces the deleted un-evidenced claim.
- ✅ **Funnel wording — done.** `src/data_pipeline.py` and `verify_setup.py` now print "hazard
  rate within the keyword-screened funnel" instead of "hazard base rate".
- Test-split numbers no longer need an "in-selection" footnote: checkpoint selection ran
  on the validation split, so they are out-of-selection. Say so.

---

## Step 5 — ✅ DONE: error analysis polish (2026-07-28, no GPU)

- ✅ **`NEGATED_HAZARD` widened** (`analysis/error_analysis.py:38`) to cover `haven't/hasn't/
  hadn't/don't/doesn't/isn't/aren't/nobody/none` and their expanded forms. All artifacts were
  re-bucketed without retraining via the new `analysis/rebucket_errors.py`, which reloads full
  review text (the detail CSVs store only a 400-char excerpt), reconstructs predictions from the
  recorded FP/FN indices, and was validated to reproduce the committed buckets exactly before
  the regex changed. **9 rows moved, all false positives** (+4 `negated_hazard` on gold, +5
  in-sample). Zero rows left either residual bucket — the new cues fired on rows already claimed
  by higher-priority rules, so the fix improved bucket *accuracy* without shrinking the residual.
- ✅ **All 23 unexplained gold FNs hand-read** → `results/gold_fn_handread.md` (+ `.csv`).
  **100% are now named**, in four roughly equal modes (5 each): `implicit_hazard`,
  `second_hand_report`, `contamination_novel_phrasing` (new — real contamination phrased outside
  the rule's vocabulary, e.g. "hair baked into cheese topping"), and `label_questionable`; plus
  2 `explicit_hazard_missed` (plain model error) and 1 `mild_or_hedged`.
- ⚠️ **The truncation hypothesis is wrong, and this is the useful finding.** Measured with the
  real `DebertaV2TokenizerFast` at `max_length=256` (not a proxy): only **1 of 23** has its
  hazard cue beyond the window, and only 2 of 23 exceed 256 tokens at all. Median hazard-cue
  position is token **39**. Raising `max_length` would recover at most one FN — and that one is
  a `label_questionable` slip-and-fall. **Do not claim truncation as a failure mode.**
- ⚠️ **`unexplained_fn` was a taxonomy artifact, not a data mystery.** 17 of the 23 *do* contain
  an `EXPLICIT_HAZARD`/`ILLNESS_WORD` term, but every FN rule is conditioned on `not
  has_explicit`, so a short low-starred review with a clear hazard word matches nothing and falls
  through. That is a better slide than the original one: *we audited our own error taxonomy and
  found the residual bucket was our rule's blind spot, not the model's.*
- 💡 **Two examples worth a slide.** In 5 cases the only hazard keyword is a disgust idiom
  ("made me want to VOMIT") while the real hazard — grease in the water, an insect, a hair — has
  no keyword at all; the idiom then *suppresses* the `contamination_no_illness` bucket via its
  `ILLNESS_WORD` guard. And 5 of 23 gold labels are arguable at LLM confidence *high* (a
  slip-and-fall, a steak-doneness mix-up, flavour revulsion read as illness) — honest evidence
  that ground truth #2 is independent but not infallible.
- The FP story is already excellent and should be a slide: **65% of the model's gold false
  positives are the labelling rule's own two top failure modes** (32.5%
  `illness_mentioned_not_caused_here` + 32.0% `neutral_allergen_mention`). The model
  inherited its teacher's blind spots. Best single example in the file — a review
  *praising* a shop's cross-contamination prevention, flagged at p=0.995.

---

## Step 5b — ✅ DONE: document embeddings, the third course technique (2026-07-29, CPU-only)

Closes the largest remaining gap against the brief: the report claimed two of the five course
techniques and the embedding family had been explicitly declined. `src/embedding_model.py` adds
it, and the run needs **no GPU** — the whole thing is ~6 minutes on the Windows/CPU box.

```bash
pip install -r requirements.txt          # adds gensim>=4.4, sentence-transformers>=2.7
python -m src.embedding_model --seeds 42 43 44
python tests/test_embedding_model.py     # 21 tests
```

If the HuggingFace hub is unreachable, drop the pretrained variant — everything else still runs:

```bash
python -m src.embedding_model --variants tfidf_lr tfidf_lsa doc2vec_dbow
```

**It is additive by construction.** It writes only `results/*embedding*` files, reads the
committed `performance_*.csv` rather than recomputing the baseline or DeBERTa, and touches no
existing module. `analysis.py`, the loss grid and the topic model were **not** re-run.

Four text-only representations, one identical class-balanced `LogisticRegression`, both ground
truths, same `score_variant` and same cost model as everything else:

| Representation (gold holdout) | PR-AUC | 95% CI | heuristic→gold delta |
|---|---:|---:|---:|
| DeBERTa-v3 (committed) | **0.804** | — | −0.183 |
| LSA 300d + LogReg | 0.757 | 0.724–0.793 | −0.169 |
| XGBoost baseline (committed) | 0.728 | — | −0.251 |
| TF-IDF + LogReg (text-only control) | 0.711 | 0.667–0.754 | −0.213 |
| MiniLM-L6-v2 frozen 384d + LogReg | 0.709 | 0.666–0.750 | −0.148 |
| Doc2Vec PV-DBOW 300d + LogReg | 0.683 | 0.639–0.732 | −0.098 |

**Three things to carry into the deck:**

1. **The result is a negative one, and it is better than a win would have been.** No frozen
   embedding is distinguishable from TF-IDF on gold, and all four lose to DeBERTa. A frozen
   transformer encoder scores **0.709** — level with plain TF-IDF at 0.711 — while the
   fine-tuned one scores **0.804**. So **dense distributed representation is not where the
   transformer's advantage comes from.** ⚠️ Do not upgrade that to "the advantage is
   fine-tuning": MiniLM-L6-v2 is also 8× smaller than DeBERTa-v3-base (6 layers / 22M vs
   12 / 184M) and pretrained on a sentence-similarity objective, so freezing, capacity and
   pretraining objective are confounded. The negative claim is clean; the causal one is not.
2. **The delta column is the money column.** It orders the six models almost perfectly by how
   little each can memorise the keyword rule. Best single sentence available: TF-IDF and frozen
   MiniLM reach the same gold PR-AUC (0.711 vs 0.709, well inside one CI) from heuristic scores
   0.066 apart — the heuristic label pays TF-IDF for skill worth nothing out of sample. Do note
   the delta is *partly* mechanical (a lower heuristic score has less room to fall), which is
   why the equal-gold/unequal-heuristic pair is the version to quote: it has no such artifact.
3. **Every model inherits the label's blind spots, and DeBERTa inherits them least.** Share of
   gold false positives sitting in the labelling rule's own top two failure modes: DeBERTa
   **64.5%** (the same figure CLAUDE.md rounds to 65%), TF-IDF 69.1%, MiniLM 71.8%, LSA 75.9%,
   Doc2Vec 76.5%
   (`results/embedding_vs_deberta_fp_modes.csv`). This upgrades the existing "the model inherits
   its teacher's blind spots" slide from an observation about one model to a **dose-response
   across five**.

⚠️ **Do not quote MiniLM's risk cost as a win.** It is the lowest in the project ($175,400 vs
DeBERTa's $204,850) because it flags 81.7% of the holdout — the identical over-flagging pathology
the loss grid already found in `focal_asymmetric@5`. Select on PR-AUC.

---

## Step 6 — Optuna sweep (optional; only if Steps 1–5 are done)

`python main.py` runs 3 trials at 4 epochs — **8–12 hours**. It buys you honest
hyperparameter provenance (Step 4) and the PR-AUC-vs-F2 selection comparison across
trials. It is a nice-to-have. If you run it, `analysis.py` would then need re-running to
pick up `best_hyperparameters.json`, which is another 2–3 hours — so realistically this is
only worth it if you have a completely free day. **The one-sentence disclosure is the
better trade.**

---

## Step 7 — Repository hygiene (1 hour total, do before submitting)

- **`requirements.txt` is still broken.** Missing `sentencepiece` (hard crash at tokenizer
  load), `matplotlib`, `seaborn`, `accelerate`. It also pins `transformers==4.40.0` and
  `torch==2.2.2`, which nobody in this project runs. A grader cloning the repo cannot
  reproduce anything. Fix the file rather than documenting the workaround.
- **W&B is a hard dependency**: `report_to="wandb"` is hardcoded in
  `src/sota_model.py:187`. Either set `report_to=os.getenv("REPORT_TO", "none")` or
  document `export WANDB_MODE=offline` prominently.
- **Three defective holdout rows**: one exact duplicate-text pair and one
  self-contradictory row (`llm_is_hazard=1` with `hazard_type="none"`, low confidence,
  benign rationale). Dropping 3 of 772 changes nothing statistically — drop them or add a
  footnote.
- **Archive the review files.** There are now four `PROFESSOR_REVIEW*.md` plus
  `final_work_overview.md` in the repo root. Self-reviews quoting grade estimates invite
  anchoring. Move them to an `archive/` directory or delete them before submission.
  (`final_work_overview.md:31` also still claims the w=50 collapse was confirmed, which
  the evidence no longer supports.)
- **Add the instructor and Tal as collaborators** — this is a submission requirement.

---

## Step 8 — Commit

`results/` is not gitignored; `model_outputs/` is (leave the checkpoints out).

```bash
git add results/ slides/ README.md
git commit -m "Add loss-variant gold scoring, presentation, README"
git push
```

---

## Reference — if something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError` / tokenizer conversion error at startup | `sentencepiece` missing | `pip install sentencepiece` |
| `TrainingArguments` refuses to construct | `accelerate` missing | `pip install accelerate` |
| MPS out of memory | `deberta-v3-base` at seq 256 on 8 GB | Lower `max_length` to 192 in `tokenize_split`, or pass `batch_size=8` |
| `RuntimeError: ... 'square_i64' ... MPS` | label dtype | Already handled (float32 labels); if it reappears, `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `nan` loss after a few steps | fp16 on MPS | `fp16=False` is already set; check the learning rate is not above 5e-5 |
| Training absurdly slow | fell back to CPU | Preflight `[2] Compute device` must say `mps`, not `cpu` |
| wandb prompts for a login | `report_to="wandb"` | `export WANDB_MODE=offline` |
| `Flag Rate` ≈ 100% in a grid cell | that configuration collapsed to all-positive | **Keep the output — it is evidence.** This is exactly what Step 1 is testing for at w=50. Do not "fix" it by lowering the weight; record it |

`grid_search_analysis.py` is now resumable — it writes incrementally *and* skips completed
`(variant, weight)` cells on restart, so an interrupted sweep costs you nothing. `analysis.py`
still restarts from scratch.

Preflight is still worth 30 seconds before any long run:

```bash
python verify_setup.py
```

---

## What to report

Build the results section around these five, in this order:

1. **The ground-truth delta** (`results/ground_truth_comparison.csv`) — the baseline
   collapses from PR-AUC 0.979 to 0.728 on real ground truth, a −0.251 drop, while DeBERTa
   drops −0.183. **This is the headline**: most of the baseline's original score was
   fidelity to a flawed keyword proxy, not hazard-detection skill. TF-IDF trivially
   recovers a keyword rule; that is now demonstrated, not asserted.
2. **PR-AUC on the gold holdout**, baseline 0.728 vs DeBERTa 0.804 — plus the deployment
   framing: 39 missed hazards vs 116, risk cost 2.9× lower. Lead with PR-AUC (threshold-free,
   so the two models' different class weighting cannot distort it).
3. **Label quality** (`label_quality.json`): 85.8% agreement, heuristic precision 73.2%,
   and the one-sidedness — 201 over-flags against 12 misses.
4. **The two error taxonomies together.** The rule over-flags because a keyword cannot
   represent causation (48% `illness_mentioned_not_caused_here`), and the *model* inherits
   exactly that blind spot (65% of its gold false positives are the rule's own top two
   modes). Tracing model errors back to label pathology is the strongest analytical move
   in the project.
5. **The topic-model lift table** — NMF isolates the rare, lexically-distinct allergen type
   (lift 5.28, the most coherent topic found at NPMI +0.43) and cannot subdivide the
   dominant food-poisoning mass. Say *why*, and a null result becomes a finding.

State these caveats yourself rather than letting a grader find them:

- The 46.0% holdout hazard rate is **within the keyword-screened funnel**, not the Yelp
  population rate (~2–5%).
- The heuristic's 97.9% recall is measured **inside its own keyword filter** — circular
  and optimistic.
- **LDA topic results are environment-sensitive**; NMF is not, which is why NMF leads.
- The loss-variant comparison **did not separate the variants** on the saturated heuristic
  metric — which is why it was re-scored on the gold holdout.
- **Run-to-run noise is large and we measured it.** All runs use the HuggingFace default seed
  (42); re-training four configurations under identical settings moved gold PR-AUC by
  0.00005–0.054. Gold gaps below ~0.05 are unresolved, so we report the weight-response *shape*
  rather than a ranking, and a multi-seed study was out of budget at ~1 GPU-hour per cell.
- **The 0.20 threshold does not transfer across ground truths** — it was tuned on the heuristic
  label, and on gold every configuration flags 66–75% of a holdout whose funnel hazard rate is
  46%. This is why PR-AUC, not fixed-threshold cost or F2, is the selection metric.
- Topic-model K was selected by NMI against the LLM types and the same NMI is reported as
  validation; and ~27% of the fit corpus is LLM-benign, which is visible as the
  service-complaint topics.

Naming your own limitations is worth more than hiding them, and every one of these has a
one-sentence honest framing.
