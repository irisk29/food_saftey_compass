# Runbook — final training and evaluation on the MacBook

Written for the state as of 2026-07-26: all code changes are done and smoke-tested, the
holdout pool is built, and the LLM labelling of the holdout set is finishing on the
Windows machine. What remains is the GPU work.

Deadline: **2026-08-02 midnight**, presentations 2026-08-03.

---

## Step 0 — Before you leave the Windows machine

The holdout gold set is being produced **on Windows**. It has to reach the MacBook.

**0a. Wait for the labelling to finish.** Check progress:

```bash
python -c "import pandas as pd; d=pd.read_csv('labeling/gold_dataset_holdout.csv'); print(len(d), 'rows,', f'{d.llm_is_hazard.mean():.1%} hazard rate')"
```

Target is 800 rows. If the process died, **re-run the exact same command** — it resumes
from what is already on disk and never re-labels a row:

```bash
python labeling/create_gold_dataset.py --source labeling/holdout_candidate_pool.csv --n 800
```

If you run short on time or API quota, **anything above ~500 rows is enough to report on**.
Below 300 the preflight check will block you, because the confidence intervals get too
wide to say anything.

**0b. Commit and push both new data files.** Neither is gitignored and both are small.

```bash
git add labeling/holdout_candidate_pool.csv labeling/gold_dataset_holdout.csv
git add src/ analysis/ tests/ config/ main.py analysis.py grid_search_analysis.py \
        verify_setup.py CLAUDE.md IMPLEMENTATION_NOTES.md PROFESSOR_REVIEW_V2.md RUNBOOK.md
git commit -m "Add fresh holdout gold set, loss variants, topic modeling, error analysis"
git push
```

You do **not** need the 5.3 GB raw Yelp JSON on the MacBook. Every step below reads only
the committed CSVs.

---

## Step 1 — Environment on the MacBook

```bash
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then install the three packages `requirements.txt` is missing. **`sentencepiece` is not
optional** — `AutoTokenizer.from_pretrained(..., use_fast=False)` fails without it, and
it fails *after* the model download, several minutes into a run:

```bash
pip install sentencepiece matplotlib seaborn
python -c "import nltk; nltk.download('vader_lexicon')"
```

Consider adding those three to `requirements.txt` while you are there, so the grader can
reproduce the project from a clean clone.

**Weights & Biases.** `main.py` calls `wandb.login()`, and the Trainer is configured with
`report_to="wandb"`. If you do not want to deal with it, disable it for the session —
nothing in the results depends on W&B, since everything is now written to `results/`:

```bash
export WANDB_MODE=offline
```

---

## Step 2 — Preflight (30 seconds, do not skip)

```bash
python verify_setup.py
```

It checks dependencies, the vader lexicon, MPS availability, all three data files, that
the gold holdout has **zero text overlap** with training data, that `src/features.py`
still reproduces the notebook's features exactly, that the selection metrics are not bare
recall, and that the 9 unit tests pass.

Fix anything marked `[FAIL]` before continuing. `[warn]` lines are safe to ignore.

---

## Step 3 — Topic modeling (2 minutes, CPU only)

Do this first: it is fast, it needs no GPU, and it secures your second course technique.

```bash
python -c "from src.topic_model import run_topic_modeling; run_topic_modeling()"
```

**Produces:** `results/topic_model_sweep.csv`, `topic_model_topics.csv`,
`topic_model_type_lift.csv`, `topic_model_crosstabs.txt`, `topic_model_selection.png`.

Results are already known from the Windows run and should reproduce exactly
(`random_state=42`): LDA K=4 and NMF K=6 selected, purity 0.716 against a 0.690
majority-class baseline, and the allergen topic at lift 5.28. Re-run it on the Mac anyway
so the committed artifacts come from one machine.

---

## Step 4 — Final evaluation run (the important one)

**This is the single run you cannot skip.** It trains both models and produces every
number in your results section, under both ground truths.

```bash
python analysis.py 2>&1 | tee results/analysis_run.log
```

**Estimated 2–3 hours** on an M1/M2 Pro (one 3-epoch DeBERTa fine-tune on 6,000 reviews
at `max_length=256`, plus inference on the test split and the gold holdout). On a base M1
with 8 GB it may be closer to 4–5 hours — see the memory note in Step 7.

**Produces:**

| File | What it is |
|---|---|
| `results/label_quality.json` | heuristic vs LLM agreement (85.8%, precision 73.2%) |
| `results/error_analysis_heuristic_label.md` | why the labelling rule errs, bucketed |
| `results/performance_heuristic_label_test_split.csv` | metrics vs the heuristic label |
| `results/performance_gold_llm_label_fresh_holdout.csv` | **metrics vs gold — the headline** |
| `results/ground_truth_comparison.csv` | the delta between the two |
| `results/error_analysis_deberta_*.md` | model failure modes, bucketed |
| `results/pr_curve_*.png`, `cost_curve_*.png` | figures for the slides |
| `results/checkpoint_selection.json` | whether F2 and PR-AUC picked the same epoch |

Watch the **`Flag Rate`** column in the printed tables. If it reads 100%, the model has
collapsed to predicting hazard for everything — see Step 7.

---

## Step 5 — Loss-variant comparison (overnight)

This is the variant experiment that answers "is the asymmetric loss actually doing
anything, or is it just class weighting?"

The full grid is 21 fine-tunes and would take **days**. Do not run it. Run this instead —
4 fine-tunes at 2 epochs, roughly **4–6 hours**, which still contains the comparison that
matters (`pos_weight` vs `focal_asymmetric`, at a low and a high penalty):

```bash
python grid_search_analysis.py \
    --variants pos_weight focal_asymmetric \
    --weights 1 15 \
    --epochs 2 2>&1 | tee results/grid_search.log
```

**Produces:** `results/grid_search_loss_variants.csv`, written incrementally after every
run — so if it dies at 3am you still keep the completed rows.

`w=1` is the unweighted control; it tells you how much the asymmetry buys at all.

**Skip this entirely if you are short on time.** Step 4 is worth far more.

---

## Step 6 — Hyperparameter sweep (optional, only if time allows)

`main.py` runs 3 Optuna trials at 4 epochs each — **8–12 hours**. You already have tuned
values from a previous sweep (`lr=1.814e-05`, `batch_size=16`), which `analysis.py` uses
by default.

The one thing this adds that you do not already have is the PR-AUC-vs-F2 selection
comparison across trials. That is a nice-to-have, not a requirement.

```bash
python main.py 2>&1 | tee results/sweep.log
```

If you run it, run it **before** Step 4 — `analysis.py` will then pick up
`results/best_hyperparameters.json` automatically instead of the hardcoded defaults.

---

## Step 7 — If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError` / tokenizer conversion error at startup | `sentencepiece` missing | `pip install sentencepiece` |
| `TypeError: __init__() got an unexpected keyword argument 'eval_strategy'` | shouldn't happen — the code detects this — but if it does, transformers is very old | `pip install -U transformers` |
| MPS out of memory | `deberta-v3-base` at seq 256 on 8 GB | Lower `max_length` to 192 in `tokenize_split`, or pass `batch_size=8` to `run_sota_training` |
| `RuntimeError: ... 'square_i64' ... MPS` | label dtype | Already handled (labels cast to float32); if it reappears, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `nan` loss after a few steps | fp16 on MPS | `fp16=False` is already set; check the learning rate is not above 5e-5 |
| Training takes absurdly long | fell back to CPU | Check preflight `[2] Compute device` says `mps`, not `cpu` |
| **`Flag Rate` = 100%, recall = 100%, precision ≈ base rate** | the model collapsed to all-positive | Lower `ASYMMETRIC_WEIGHT` in `config/settings.py` from 50 to 10–15 and re-run. This is a real, documented failure at w=50 — if it happens, **keep the output**, it is evidence for the write-up |
| wandb prompts for a login | `report_to="wandb"` | `export WANDB_MODE=offline` |

Every long run above is safe to `tee` and safe to interrupt; only `grid_search_analysis.py`
resumes partially (via its incremental CSV writes). `analysis.py` restarts from scratch.

---

## Step 8 — Commit the results

`results/` is not gitignored. `model_outputs/` is (checkpoints are large — leave them out).

```bash
git add results/
git commit -m "Add final evaluation results, topic model artifacts, error analysis"
git push
```

Then add the instructor and Tal as repository collaborators, if that is not already done.

---

## Step 9 — What to actually report

Once Step 4 finishes, the numbers you build the presentation around are:

1. **`results/ground_truth_comparison.csv`** — the same model scored against the heuristic
   label vs the independent gold label. In the smoke test the baseline dropped from
   PR-AUC 0.986 to 0.717. If the real run shows a gap of that shape, **that is your
   headline finding**: most of the original score was fidelity to a flawed proxy, not
   hazard-detection skill.

2. **PR-AUC on the gold holdout**, baseline vs DeBERTa. Lead with PR-AUC, not accuracy and
   not recall — it is threshold-free, so it is not distorted by the two models' different
   class weighting.

3. **The label-quality numbers** (`label_quality.json`): 85.8% agreement, heuristic
   precision 73.2%, and the fact that the errors are one-sided (201 over-flags vs 12
   misses).

4. **The error-analysis buckets** — especially that 48% of the labelling rule's false
   hazards are `illness_mentioned_not_caused_here`. A keyword rule cannot represent
   causation, only co-occurrence.

5. **The topic-model lift table** — the models isolate the rare, lexically-distinct
   allergen type (lift 5.28) and fail to subdivide the dominant food-poisoning mass. Say
   *why*, and it becomes a strength rather than a null result.

Two caveats to state out loud rather than let a grader find:

- The holdout hazard base rate (~42%) is the rate **among keyword-flagged candidates**,
  not among all Yelp reviews — the pool is pre-filtered.
- The heuristic's 97.9% recall is measured **inside an already keyword-filtered dataset**,
  so it is circular and optimistic.

Naming your own limitations is worth more than hiding them, and both of these have a
one-sentence honest framing.
