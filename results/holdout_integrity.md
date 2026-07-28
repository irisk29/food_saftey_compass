# Gold holdout integrity check — `labeling/gold_dataset_holdout.csv`

Verified 2026-07-28 on the committed file. **Nothing here is repaired**, deliberately:
every committed model number (`results/performance_*.csv`, `results/error_analysis_*`,
all eight `gold_*` columns of `results/grid_search_loss_variants.csv`) is computed on
these exact 772 rows. Dropping rows now would invalidate every artifact in the repo to
buy a change of ~0.3% in the denominator. The defects are disclosed instead.

## What was checked

| Property | Result |
|---|---|
| Rows | 772 |
| Positive (`llm_is_hazard = 1`) | 355 — **46.0%** funnel hazard rate |
| Overlap with training data | **0 / 772** (independently re-verified in `verify_setup.py` preflight) |
| Duplicate `review_id` | 0 |
| Empty / NaN `text` | 0 |
| Empty `llm_rationale` | 0 |
| `word_count` range | 9 – 767 |

## Known defects (2 + 5 rows, all disclosed, none repaired)

**1. One duplicate-text group — 2 rows.**

| `source_index` | `review_id` | stars | words | `llm_is_hazard` | confidence |
|---:|---|---:|---:|---:|---|
| 2293 | `XbZj8XHs1j3dcaKz6kC8Xw` | 1 | 130 | 1 | high |
| 1127 | `Qk8Yjm2Z70MYSdFFrKy4Zw` | 1 | 130 | 1 | high |

Distinct `review_id`s with byte-identical text — a genuine duplicate posting in the
source Yelp corpus, not a pipeline bug. Both received the same label from the judge,
so the pair adds one row of redundant weight to a 772-row set (0.13%). No metric in
the repo moves at the reported precision.

**2. Five rows labelled below `high` confidence.** 767 rows are `high`, 3 are `medium`,
2 are `low`. These are retained because excluding low-confidence judgements would be
selecting the evaluation set on the judge's own certainty, which biases the holdout
toward easy cases — exactly the circularity this holdout exists to avoid.

## Related, and more consequential than either defect above

The hand-read audit of all 23 residual false negatives (`results/gold_fn_handread.md`)
found **5 of 23 gold labels arguable at LLM confidence `high`** — slip-and-fall
mislabelled as a food hazard, a steak-doneness mix-up, flavour revulsion read as
illness. That is the honest headline on holdout quality: ground truth #2 is
*independent* of the heuristic label, which is what makes it useful, but it is not
infallible. The 46.0% hazard rate is also a **funnel** rate — 100% of these rows passed
the keyword screen — against an estimated ~2–5% population rate.
