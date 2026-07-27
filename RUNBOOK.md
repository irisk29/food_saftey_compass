# Runbook — what to do now

Rewritten **2026-07-27 (evening)**, after the stage-4 and stage-5 runs. The original
version of this file was a pre-execution guide; every GPU step in it has now been run,
so this is a fresh plan for the time that is left.

**Deadline: 2026-08-02 midnight. Presentations: 2026-08-03. You have ~6 days.**

The headline: **the modelling is essentially done and the results are good.** What is
left is one unattended GPU run, a presentation that does not exist yet, and a pass of
corrections. Nothing on the critical path requires new research.

---

## Where the project actually stands

| Piece | State |
|---|---|
| Holdout gold set | ✅ 772 rows, 355 hazard (46.0% within the keyword funnel), 767 high-confidence, committed |
| Topic modelling | ✅ Run and committed; NPMI bug fixed. One caveat — see Step 4 |
| Final evaluation (`analysis.py`) | ✅ Run 2026-07-27; every performance artifact exists |
| Model error analysis | ✅ Both ground truths, bucketed |
| Loss-variant grid | ⚠️ Partially run — 4 of the cells that matter are missing (Step 1) |
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

## Step 1 — Finish the loss grid (start this TONIGHT, unattended)

This is the only remaining GPU work, and it is the only item that is both blocking and
slow, so it goes first — start it before you do anything else, then work on slides while
it runs.

**Why it needs re-running.** The four cells you have are scored only against the
heuristic label on the validation split, where every configuration sits at ceiling:
PR-AUC spans 0.9818–0.9882, a spread of 0.0065, which is single-seed noise. The grid as it
stands cannot rank the variants. Two further gaps: **w=50 was never tested**, and w=50 is
the configuration every headline number in your project came from
(`focal_asymmetric, w=50, gamma=2.0`). So the deployed config was compared against nothing.

`grid_search_analysis.py` now scores every cell on the **gold holdout** as well
(`gold_*` columns), where models actually differ, and it resumes — the 4 finished cells
are skipped, not re-paid for.

```bash
export WANDB_MODE=offline
python grid_search_analysis.py --quick 2>&1 | tee -a results/grid_search.log
```

That runs exactly **4 new fine-tunes**: `pos_weight @ {5, 50}` and
`focal_asymmetric @ {5, 50}`. Budget an overnight. It gives you a clean 2×2 on the gold
holdout (two variants × two penalties) plus answers to both open questions:

- Does `pos_weight @ w=50` actually collapse? Your docs used to assert it did; no artifact
  ever showed it, and the deployed focal model at the same weight does **not** (flag rate
  0.207). Watch the `collapsed` and `gold_collapsed` columns.
- Does the asymmetric loss beat the unweighted control *where it matters*? On the
  heuristic label it does not — the best PR-AUC in the current grid is `pos_weight @ w=1`,
  i.e. no asymmetry at all.

**Note the existing 4 rows will keep blank `gold_*` cells** — that means *not measured*,
not zero. If you have a second free night and want the full gold table, re-run everything
with `--force --quick` (8 fine-tunes). If you are badly short on time, the minimum viable
version is `--weights 50` (2 runs), which still covers the deployed config.

**If the result is "no variant wins," report that.** "We ran the comparison and it did not
separate, and here is why the metric was saturated" is a genuine finding, and the rubric
rewards understanding why a technique fails. Do not quietly drop the experiment.

---

## Step 2 — Build the presentation (start today, in parallel with Step 1)

**This is now the biggest risk in the project.** 15% of the grade, presentations are the
day after the deadline, and only `slides/SLIDES_SKELETON.md` exists. The skeleton is good
— it has the spine, per-slide evidence lines, and timing — but it is not a deck.

Nothing in it is blocked on data any more. **Correct these numbers as you build:**

- Holdout is **772 rows / 355 hazards**, not 744/342.
- Base rate **46.0%**, and say "within the keyword-screened funnel" out loud — the raw
  Yelp rate is plausibly 2–5%.
- **Topic model: lead with NMF, not LDA.** The skeleton quotes purity 0.716 and LDA lift
  4.18; those came from a run whose artifacts no longer exist (see Step 4). Use NMF K=6,
  allergen lift **5.28**, per-topic NPMI **+0.43** — reproducible and the stronger result.
- Use the real classifier numbers from the table above, not the smoke-test figures.

The spine still holds, and it is a strong one: *we built a hazard detector, then spent the
project proving our own ground truth wrong — twice — before trusting a single number.*

---

## Step 3 — README (1–2 hours)

Still one line. It is the grader's first impression and the content already exists in
`CLAUDE.md` and `IMPLEMENTATION_NOTES.md`. One page: the problem, the cost asymmetry, the
two-ground-truths design, the headline table above, and an index of what lives in
`results/`.

---

## Step 4 — Corrections pass (1 hour, pure Q&A insurance)

Each of these is a question you can take away from a grader for free.

- **LDA is not reproducible across machines.** Re-running the fixed topic sweep in a
  second environment with the same seed produced different LDA fits (allergen lift 4.18 vs
  the committed 2.52; purity 0.716 vs 0.690; coherence peak K=4 vs K=8). NMF reproduced
  exactly. Quote LDA numbers only from the committed artifacts, lead with NMF, and put one
  sentence in the write-up owning it — "we caught our own topic model being irreproducible
  and moved the headline claims to the algorithm that reproduces" reads as rigour, not
  weakness. Do **not** re-run the topic model expecting the old numbers back.
- **Hyperparameter provenance.** No `optuna_trials.csv` or `best_hyperparameters.json`
  exists, so `analysis.py` fell back to its hardcoded `lr=1.814e-05, bs=16` — values from
  a sweep run *before* the metric fixes and the validation split. The results are clean
  (evaluation never touched selection data); only the provenance is impure. Either run
  Step 6 or add one sentence: *"hyperparameters were carried over from an earlier sweep;
  the final configuration was re-trained and evaluated under the fixed protocol."*
- **Funnel wording.** `src/data_pipeline.py:86` still prints "hazard base rate" — say
  "hazard rate within the keyword-screened funnel."
- Test-split numbers no longer need an "in-selection" footnote: checkpoint selection ran
  on the validation split, so they are out-of-selection. Say so.

---

## Step 5 — Error analysis polish (1–2 hours, no GPU)

- **`NEGATED_HAZARD` still omits `haven't/hasn't/hadn't`** (`analysis/error_analysis.py:38`),
  so the committed error-analysis files were generated with a known-flawed regex. Fix it
  and re-run `analyze_errors` — the detail CSVs already carry the probabilities, so no
  retraining is needed.
- **23 of 39 gold false negatives are `unexplained_fn`** (59%). Hand-read those 23 in
  `results/error_analysis_deberta_gold_llm_label_fresh_holdout_detail.csv`; several
  visible examples state the hazard in the review's last sentence, which is consistent
  with the 256-token truncation. Turning "59% unexplained" into a named failure mode is an
  80s→90s discriminator.
- The FP story is already excellent and should be a slide: **66% of the model's gold false
  positives are the labelling rule's own two top failure modes** (34%
  `illness_mentioned_not_caused_here` + 32% `neutral_allergen_mention`). The model
  inherited its teacher's blind spots. Best single example in the file — a review
  *praising* a shop's cross-contamination prevention, flagged at p=0.995.

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
   exactly that blind spot (66% of its gold false positives are the rule's own top two
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
- Topic-model K was selected by NMI against the LLM types and the same NMI is reported as
  validation; and ~27% of the fit corpus is LLM-benign, which is visible as the
  service-complaint topics.

Naming your own limitations is worth more than hiding them, and every one of these has a
one-sentence honest framing.
