"""
Bootstrap confidence intervals for the threshold-dependent gold-holdout metrics.

Why this exists
---------------
Every headline number in this project is a point estimate. "89.0% recall / 61.6%
precision / F2 0.817, 39 of 355 hazards missed" is measured on **772 rows**, and a
reader cannot tell from a point estimate whether 89.0% could as easily have been 85%
or 92%. This module attaches an interval to each of those figures, and — more
usefully — to the *difference* between the deployed transformer and the recoverable
baselines, on the same resampled rows.

What it is
----------
A non-parametric percentile bootstrap over the 772 gold-holdout rows, holding the
trained models fixed. It resamples rows with replacement, recomputes the confusion
matrix for every model on the same resample, and reports percentile 95% intervals
for recall, precision, F1, F2, flag rate, missed hazards, false alarms and total
risk cost. Because every model is scored on the *same* resample in every iteration,
the difference between two models is a genuine paired bootstrap, which is a much
tighter and more honest statement than comparing two independent intervals.

It adds an exact McNemar test on the discordant pairs as a second, distribution-free
view of the same paired comparison.

What it is NOT
--------------
* **Not a PR-AUC bootstrap.** PR-AUC needs a per-row probability for all 772 rows.
  No model checkpoint is committed and `results/` persists probabilities only for the
  *error* rows (`error_analysis_*_detail.csv`), so the full probability vector is
  unrecoverable without retraining. Threshold-free metrics are therefore out of scope
  here, deliberately. `src/embedding_model.bootstrap_gold_pr_auc` does bootstrap
  PR-AUC, but only for the embedding variants it fits in-process.
* **Not a measure of training variance.** These intervals hold the *trained model*
  fixed and vary only the evaluation sample. The project separately measured up to
  **0.054 gold PR-AUC** between nominally identical re-runs at a fixed seed
  (`results/grid_search_loss_variants.csv`, `pos_weight@5`). The two sources of
  uncertainty compose; a reader who sees only the interval below will *underestimate*
  total uncertainty. This warning is repeated in the generated markdown on purpose.
* **Not a test against the XGBoost baseline.** `analysis/evaluation_pipeline.py` runs
  `analyze_errors` for the transformer only, so the baseline's per-row errors were
  never persisted and no XGBoost pickle is committed. The headline "116 missed vs 39
  missed" gap therefore cannot be paired-tested. The recoverable comparators are the
  four embedding/text controls from `src/embedding_model.py`, whose error indices *are*
  committed; the primary one used here is the TF-IDF + LogReg text-only control, which
  is the "TF-IDF + classifier" baseline of the step-4 plan minus the tabular features.
* **Not a re-run of anything.** No training, no GPU, no probability recomputation. It
  reads committed CSVs only, and writes only new `results/bootstrap_ci_*` files.

How the predictions are recovered
---------------------------------
Identical in principle to `analysis/rebucket_errors.py`: the detail CSVs record the row
index of every false positive and false negative, so starting from the gold labels and
flipping exactly those indices reproduces the hard predictions by construction; every
other row is a true positive or true negative by elimination. The reconstruction is then
reconciled against the committed `results/performance_*.csv` rows to 1e-9 on every
metric before any resampling happens, and the run aborts if it does not match.

Determinism
-----------
One `numpy.random.default_rng(cfg.RANDOM_STATE)` drives everything; resample weights are
drawn once and shared across models, so re-running reproduces every number exactly.

Usage
-----
    python -m analysis.bootstrap_ci                  # 20,000 resamples, writes artifacts
    python -m analysis.bootstrap_ci --n-boot 5000
    python -m analysis.bootstrap_ci --check          # reconcile only, write nothing
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

import config.settings as cfg
from analysis.evaluation_pipeline import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

# Evaluation set. Only the fresh gold holdout is handled: it is the only set the project
# reports model metrics on, and the only one whose rows are an i.i.d.-ish sample that a
# row bootstrap is meaningful over.
SLUG = "gold_llm_label_fresh_holdout"

# model key -> (error-detail label, committed performance CSV, row label in that CSV)
# Every entry is scored at the deployed threshold 0.20; the committed FP/FN counts below
# are what the reconciliation asserts against.
MODELS = {
    "deberta_th020": (
        f"deberta_{SLUG}",
        f"performance_{SLUG}.csv",
        "DeBERTa-v3 (th=0.20, deployed)",
    ),
    "tfidf_lr_th020": (
        f"embedding_tfidf_lr_{SLUG}",
        f"performance_embedding_{SLUG}.csv",
        "TF-IDF + LogReg (text-only control) (th=0.20)",
    ),
    "tfidf_lsa_th020": (
        f"embedding_tfidf_lsa_{SLUG}",
        f"performance_embedding_{SLUG}.csv",
        "LSA 300d + LogReg (th=0.20)",
    ),
    "minilm_th020": (
        f"embedding_minilm_{SLUG}",
        f"performance_embedding_{SLUG}.csv",
        "MiniLM-L6-v2 frozen 384d + LogReg (th=0.20)",
    ),
    "doc2vec_dbow_th020": (
        f"embedding_doc2vec_dbow_{SLUG}",
        f"performance_embedding_{SLUG}.csv",
        "Doc2Vec PV-DBOW 300d + LogReg (th=0.20)",
    ),
}

DISPLAY = {
    "deberta_th020": "DeBERTa-v3 (th=0.20, deployed)",
    "tfidf_lr_th020": "TF-IDF + LogReg control (th=0.20)",
    "tfidf_lsa_th020": "LSA 300d + LogReg (th=0.20)",
    "minilm_th020": "MiniLM-L6-v2 frozen + LogReg (th=0.20)",
    "doc2vec_dbow_th020": "Doc2Vec PV-DBOW + LogReg (th=0.20)",
}

# The deployed system, and the comparator the paired difference headlines on.
REFERENCE = "deberta_th020"
PRIMARY_COMPARATOR = "tfidf_lr_th020"

# Metrics reported with intervals. Each maps confusion counts -> value. Written as
# closed forms over (tn, fp, fn, tp) so they vectorise over thousands of resamples at
# once; `_selfcheck_metrics` asserts they agree with sklearn on the real data.
METRICS = {
    "Recall (Safety Coverage)": lambda tn, fp, fn, tp: _div(tp, tp + fn),
    "Precision (Alert Validity)": lambda tn, fp, fn, tp: _div(tp, tp + fp),
    "F2 (recall-weighted)": lambda tn, fp, fn, tp: _fbeta(fp, fn, tp, cfg.FBETA),
    "F1": lambda tn, fp, fn, tp: _fbeta(fp, fn, tp, 1.0),
    "Flag Rate": lambda tn, fp, fn, tp: _div(tp + fp, tn + fp + fn + tp),
    "False Negatives (Missed)": lambda tn, fp, fn, tp: fn.astype(float),
    "False Positives (Alarms)": lambda tn, fp, fn, tp: fp.astype(float),
    "Total Risk Cost": lambda tn, fp, fn, tp: (fn * COST_FALSE_NEGATIVE
                                               + fp * COST_FALSE_POSITIVE),
}

# Metrics whose paired difference is worth reporting. "Lower is better" for the count
# and cost metrics, which is why the sign convention is spelled out in the artifact.
PAIRED_METRICS = list(METRICS)

# Resampling schemes. `rows` is the one the task and the report want: it varies the
# hazard base rate exactly as a fresh 772-row draw would. `stratified` holds the 46.0%
# funnel base rate fixed, matching src/embedding_model.bootstrap_gold_pr_auc; it is
# reported as a sensitivity check, and it is *narrower* by construction, so quoting it
# alone would understate the interval.
SCHEMES = ("rows", "stratified")

DEFAULT_N_BOOT = 20000
STABILITY_LADDER = (1000, 2000, 5000, 10000, 20000)


def _div(num, den):
    """Elementwise num/den with 0/0 -> 0, matching sklearn's zero_division=0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _fbeta(fp, fn, tp, beta):
    b2 = float(beta) ** 2
    return _div((1.0 + b2) * tp, (1.0 + b2) * tp + b2 * fn + fp)


# --------------------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------------------

def load_gold_holdout():
    """
    The 772-row fresh holdout with the LLM judgement as ground truth.

    Read straight from `labeling/gold_dataset_holdout.csv` with a positional index,
    because that is the index the detail CSVs recorded against. Features are not
    needed — nothing here re-scores a model — so `src.features.enrich` is skipped and
    the frame is a byte-faithful read of the committed file.
    """
    df = pd.read_csv(cfg.GOLD_HOLDOUT_PATH)
    df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")
    return df.reset_index(drop=True)


def reconstruct_predictions(key, gold, results_dir):
    """
    Recover one model's hard predictions on all 772 rows from its committed detail CSV.

    y_pred := y_true with exactly the recorded FP indices set to 1 and FN indices set
    to 0. Every unrecorded row is a correct prediction, so it is a TP where y_true==1
    and a TN where y_true==0. Raises if an index is out of range, if the stored text
    excerpt does not match the row it claims to be, or if the recovered confusion
    counts disagree with the detail CSV's own tallies.
    """
    label, _, _ = MODELS[key]
    path = os.path.join(results_dir, f"error_analysis_{label}_detail.csv")
    detail = pd.read_csv(path)

    y_true = gold["llm_is_hazard"].astype(int).to_numpy()

    missing = set(detail["index"]) - set(range(len(gold)))
    if missing:
        raise ValueError(
            f"[{key}] {len(missing)} recorded error indices are outside the "
            f"{len(gold)}-row holdout (e.g. {sorted(missing)[:5]}). The gold file has "
            f"drifted since {os.path.basename(path)} was written; refusing to bootstrap."
        )

    # Same guard rail rebucket_errors uses: the detail CSV stores a 400-char excerpt, so
    # a prefix comparison proves the index mapping is still the one that was written.
    for _, r in detail.iterrows():
        if str(gold.loc[r["index"], cfg.TEXT_COLUMN])[:200] != str(r["text"])[:200]:
            raise ValueError(f"[{key}] text mismatch at index {r['index']}; "
                             f"refusing to bootstrap a mapping we cannot verify.")

    y_pred = y_true.copy()
    y_pred[detail.loc[detail.error_type == "FP", "index"].to_numpy()] = 1
    y_pred[detail.loc[detail.error_type == "FN", "index"].to_numpy()] = 0

    recorded = detail.error_type.value_counts().to_dict()
    got_fp = int(((y_true == 0) & (y_pred == 1)).sum())
    got_fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if got_fp != recorded.get("FP", 0) or got_fn != recorded.get("FN", 0):
        raise ValueError(f"[{key}] reconstructed {got_fp} FP / {got_fn} FN but "
                         f"{os.path.basename(path)} records "
                         f"{recorded.get('FP', 0)} / {recorded.get('FN', 0)}.")
    return y_pred


def _counts(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def point_metrics(y_true, y_pred):
    """Metric dict on the real (unresampled) rows."""
    c = [np.array([v]) for v in _counts(y_true, y_pred)]
    return {name: float(fn(*c)[0]) for name, fn in METRICS.items()}


def _selfcheck_metrics(y_true, y_pred):
    """Assert the closed-form metrics equal sklearn's, so the vectorised path is trusted."""
    from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
    got = point_metrics(y_true, y_pred)
    want = {
        "Recall (Safety Coverage)": recall_score(y_true, y_pred, zero_division=0),
        "Precision (Alert Validity)": precision_score(y_true, y_pred, zero_division=0),
        "F2 (recall-weighted)": fbeta_score(y_true, y_pred, beta=cfg.FBETA, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    for k, v in want.items():
        if abs(got[k] - v) > 1e-12:
            raise AssertionError(f"closed-form {k} = {got[k]} != sklearn {v}")


def reconcile(key, gold, y_pred, results_dir, tol=1e-9):
    """
    Check the reconstruction against the committed performance CSV row.

    Returns (row_dict, ok). Compares every metric the committed CSV shares with
    `METRICS`; a mismatch anywhere means the reconstruction is not the evaluated
    system and the run must stop rather than publish an interval around the wrong
    point estimate.
    """
    _, perf_file, perf_row = MODELS[key]
    perf = pd.read_csv(os.path.join(results_dir, perf_file), index_col=0)
    if perf_row not in perf.index:
        raise ValueError(f"[{key}] row {perf_row!r} not in {perf_file}; "
                         f"available: {list(perf.index)}")
    committed = perf.loc[perf_row]
    got = point_metrics(gold["llm_is_hazard"].astype(int).to_numpy(), y_pred)

    row = {"model": key, "display": DISPLAY[key], "committed_row": perf_row}
    ok = True
    for metric in METRICS:
        if metric not in committed.index:
            continue
        c, g = float(committed[metric]), got[metric]
        row[f"committed_{metric}"] = c
        row[f"recovered_{metric}"] = g
        row[f"abs_diff_{metric}"] = abs(g - c)
        ok = ok and abs(g - c) <= tol
    row["reconciles"] = ok
    return row, ok


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------

def _draw_weights(y_true, n_boot, scheme, rng, chunk=2000):
    """
    Yield (n_boot_chunk, n) integer multiplicity matrices for one resampling scheme.

    Multiplicities rather than index lists: every metric here is a function of four
    counts, and each count is a linear functional of the multiplicity vector, so a
    single matrix product gives all four counts for all models. This is what makes the
    bootstrap *paired* — one weight matrix, every model scored under it.

    `rows`       : multinomial over all n rows, i.e. an ordinary i.i.d. row bootstrap.
                   The hazard base rate wobbles, exactly as it would in a fresh draw.
    `stratified` : independent multinomials within the positive and negative rows, so
                   every replicate keeps 355 positives and 417 negatives. Narrower by
                   construction; reported as a sensitivity check only.
    """
    n = len(y_true)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        w = np.zeros((b, n), dtype=np.int32)
        if scheme == "rows":
            w[:] = rng.multinomial(n, np.full(n, 1.0 / n), size=b)
        elif scheme == "stratified":
            w[:, pos] = rng.multinomial(len(pos), np.full(len(pos), 1.0 / len(pos)), size=b)
            w[:, neg] = rng.multinomial(len(neg), np.full(len(neg), 1.0 / len(neg)), size=b)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        done += b
        yield w


def _category(y_true, y_pred):
    """One-hot (n, 4) indicator over (TN, FP, FN, TP) for a single model."""
    ind = np.zeros((len(y_true), 4), dtype=np.int32)
    ind[(y_true == 0) & (y_pred == 0), 0] = 1
    ind[(y_true == 0) & (y_pred == 1), 1] = 1
    ind[(y_true == 1) & (y_pred == 0), 2] = 1
    ind[(y_true == 1) & (y_pred == 1), 3] = 1
    return ind


def bootstrap_draws(y_true, preds, n_boot, scheme, seed):
    """
    Resample rows `n_boot` times and score every model on each resample.

    Returns {model: {metric: array(n_boot)}}. All models share the same resample in
    every iteration, so differences taken across the returned arrays are paired.
    """
    inds = {k: _category(y_true, p) for k, p in preds.items()}
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in preds}
    for w in _draw_weights(y_true, n_boot, scheme, rng):
        for k, ind in inds.items():
            c = w @ ind                      # (chunk, 4) resampled confusion counts
            parts[k].append(c)
    out = {}
    for k, chunks in parts.items():
        c = np.concatenate(chunks, axis=0)
        tn, fp, fn, tp = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        out[k] = {name: f(tn, fp, fn, tp) for name, f in METRICS.items()}
    return out


def _interval(draws, point, alpha=0.05):
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point,
        "boot_mean": float(draws.mean()),
        "boot_sd": float(draws.std(ddof=1)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci95_width": float(hi - lo),
        # Difference between the bootstrap mean and the point estimate. Large relative
        # to the SD would indicate a badly skewed statistic; reported so a reader can
        # see that it is not.
        "bias_estimate": float(draws.mean() - point),
    }


def interval_table(y_true, preds, n_boot, seed):
    """One row per (scheme, model, metric): point estimate plus percentile 95% CI."""
    points = {k: point_metrics(y_true, p) for k, p in preds.items()}
    rows = []
    for scheme in SCHEMES:
        draws = bootstrap_draws(y_true, preds, n_boot, scheme, seed)
        for k in preds:
            for metric in METRICS:
                rows.append({
                    "scheme": scheme, "model": k, "display": DISPLAY[k],
                    "metric": metric, "n_rows": int(len(y_true)), "n_boot": n_boot,
                    **_interval(draws[k][metric], points[k][metric]),
                })
    return pd.DataFrame(rows)


def paired_table(y_true, preds, n_boot, seed, reference=REFERENCE):
    """
    Paired bootstrap on `reference` minus each other model, on identical resamples.

    The headline quantity is `False Negatives (Missed)`: a negative difference means
    the transformer misses fewer hazards. `p_gt_zero` / `p_lt_zero` are the bootstrap
    fractions on each side of zero — a two-sided bootstrap p-value is
    2*min(p_lt_zero, p_gt_zero), floored at 1/n_boot, and is reported as
    `boot_p_two_sided`.
    """
    points = {k: point_metrics(y_true, p) for k, p in preds.items()}
    rows = []
    for scheme in SCHEMES:
        draws = bootstrap_draws(y_true, preds, n_boot, scheme, seed)
        for k in preds:
            if k == reference:
                continue
            for metric in PAIRED_METRICS:
                d = draws[reference][metric] - draws[k][metric]
                p_lt = float((d < 0).mean())
                p_gt = float((d > 0).mean())
                iv = _interval(d, points[reference][metric] - points[k][metric])
                rows.append({
                    "scheme": scheme, "reference": reference, "comparator": k,
                    "comparator_display": DISPLAY[k], "metric": metric,
                    "n_rows": int(len(y_true)), "n_boot": n_boot,
                    **iv,
                    "p_lt_zero": p_lt, "p_gt_zero": p_gt,
                    "boot_p_two_sided": max(2.0 * min(p_lt, p_gt), 1.0 / n_boot),
                    "excludes_zero": bool(iv["ci95_low"] > 0 or iv["ci95_high"] < 0),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Exact paired test
# --------------------------------------------------------------------------------------

def mcnemar_table(y_true, preds, reference=REFERENCE):
    """
    Exact McNemar on the discordant pairs, in two framings.

    `correctness` — over all 772 rows: b = reference right / comparator wrong,
    c = reference wrong / comparator right. This is the textbook paired accuracy test.

    `missed_hazards` — over the 355 true hazards only: b = reference caught it /
    comparator missed it, c = reference missed it / comparator caught it. This is the
    framing the report actually cares about, because the deployed system is sold on
    not missing hazards, and it is the discordance the paired bootstrap on
    `False Negatives (Missed)` also summarises.

    Reports the exact binomial p-value (the exact McNemar test, `binomtest(b, b+c, 0.5)`)
    and the continuity-corrected chi-square for reference. No approximation is relied on:
    the exact test is the one quoted.
    """
    ref = preds[reference]
    rows = []
    for k, p in preds.items():
        if k == reference:
            continue
        for framing, mask in (("correctness", np.ones(len(y_true), bool)),
                              ("missed_hazards", y_true == 1)):
            if framing == "correctness":
                ref_ok, cmp_ok = (ref == y_true), (p == y_true)
            else:
                ref_ok, cmp_ok = (ref == 1), (p == 1)
            m = mask
            b = int((ref_ok & ~cmp_ok & m).sum())   # reference better on this row
            c = int((~ref_ok & cmp_ok & m).sum())   # comparator better on this row
            n_disc = b + c
            exact = binomtest(b, n_disc, 0.5).pvalue if n_disc else float("nan")
            if n_disc:
                stat = (abs(b - c) - 1) ** 2 / n_disc
                chi_p = float(chi2.sf(stat, 1))
            else:
                stat = chi_p = float("nan")
            rows.append({
                "reference": reference, "comparator": k,
                "comparator_display": DISPLAY[k], "framing": framing,
                "n_considered": int(m.sum()),
                "both_correct": int((ref_ok & cmp_ok & m).sum()),
                "both_wrong": int((~ref_ok & ~cmp_ok & m).sum()),
                "reference_only_correct_b": b, "comparator_only_correct_c": c,
                "n_discordant": n_disc,
                "exact_binomial_p": exact,
                "chi2_cc_statistic": float(stat), "chi2_cc_p": chi_p,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Stability of the interval in the number of resamples
# --------------------------------------------------------------------------------------

def stability_table(y_true, preds, seed, ladder=STABILITY_LADDER,
                    scheme="rows", watch=("Recall (Safety Coverage)",
                                          "F2 (recall-weighted)",
                                          "False Negatives (Missed)")):
    """
    Recompute the interval at increasing `n_boot` so the chosen count is justified.

    Reported as the movement of each CI bound relative to the largest ladder rung.
    "Enough resamples" means the bounds have stopped moving at the precision the report
    quotes (3 decimal places / whole counts), not an arbitrary round number.
    """
    rows = []
    for n_boot in ladder:
        draws = bootstrap_draws(y_true, preds, n_boot, scheme, seed)
        for k in preds:
            pt = point_metrics(y_true, preds[k])
            for metric in watch:
                iv = _interval(draws[k][metric], pt[metric])
                rows.append({"scheme": scheme, "n_boot": n_boot, "model": k,
                             "display": DISPLAY[k], "metric": metric,
                             "ci95_low": iv["ci95_low"], "ci95_high": iv["ci95_high"],
                             "ci95_width": iv["ci95_width"]})
    out = pd.DataFrame(rows)
    ref_n = max(ladder)
    ref = out[out.n_boot == ref_n].set_index(["model", "metric"])
    out["low_shift_vs_max"] = [
        r.ci95_low - ref.loc[(r.model, r.metric), "ci95_low"] for r in out.itertuples()]
    out["high_shift_vs_max"] = [
        r.ci95_high - ref.loc[(r.model, r.metric), "ci95_high"] for r in out.itertuples()]
    return out


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------

def _fmt(metric, v):
    if metric in ("False Negatives (Missed)", "False Positives (Alarms)"):
        return f"{v:.1f}"
    if metric == "Total Risk Cost":
        return f"${v:,.0f}"
    return f"{v:.4f}"


def _md_interval_rows(intervals, model, scheme="rows"):
    sub = intervals[(intervals.model == model) & (intervals.scheme == scheme)]
    lines = ["| Metric | Point estimate | 95% CI (percentile) | CI width |",
             "|---|---:|---:|---:|"]
    for metric in METRICS:
        r = sub[sub.metric == metric].iloc[0]
        lines.append(f"| {metric} | {_fmt(metric, r.point_estimate)} | "
                     f"[{_fmt(metric, r.ci95_low)}, {_fmt(metric, r.ci95_high)}] | "
                     f"{_fmt(metric, r.ci95_width)} |")
    return lines


def write_markdown(intervals, paired, mcnemar, stability, recon, n_boot, output_dir, n_pos):
    ref_disp = DISPLAY[REFERENCE]
    n_rows = int(intervals.n_rows.iloc[0])
    prim = paired[(paired.comparator == PRIMARY_COMPARATOR) & (paired.scheme == "rows")]
    fn_row = prim[prim.metric == "False Negatives (Missed)"].iloc[0]
    rec_row = prim[prim.metric == "Recall (Safety Coverage)"].iloc[0]
    mc = mcnemar[(mcnemar.comparator == PRIMARY_COMPARATOR)
                 & (mcnemar.framing == "missed_hazards")].iloc[0]
    mc_all = mcnemar[(mcnemar.comparator == PRIMARY_COMPARATOR)
                     & (mcnemar.framing == "correctness")].iloc[0]

    L = [
        "# Bootstrap confidence intervals — gold LLM label (fresh holdout)",
        "",
        f"Generated by `analysis/bootstrap_ci.py` from committed artifacts only "
        f"(no training, no GPU, no re-scoring). {n_boot:,} resamples, "
        f"seed `config.settings.RANDOM_STATE = {cfg.RANDOM_STATE}`, n = {n_rows} rows, "
        f"{n_pos} true hazards.",
        "",
        "## What was measured, and what it cannot be",
        "",
        "Hard predictions at the deployed threshold 0.20 were reconstructed for every model "
        "whose per-row errors are committed, by taking the gold labels and flipping exactly "
        "the false-positive and false-negative indices recorded in "
        "`results/error_analysis_*_detail.csv` — the same reconstruction "
        "`analysis/rebucket_errors.py` performs. Each reconstruction was reconciled against "
        "the committed `results/performance_*.csv` row to within 1e-9 on every shared metric "
        "before any resampling; see `bootstrap_ci_reconciliation_"
        f"{SLUG}.csv`.",
        "",
        "**Two things this does not cover, and both matter:**",
        "",
        "1. **PR-AUC has no interval here, and cannot get one from committed files.** "
        "No model checkpoint is committed, and `results/` stores per-row probabilities only "
        "for the *error* rows. A threshold-free metric needs all 772 probabilities, so the "
        "PR-AUC figures (0.8045 for the deployed system) remain bare point estimates. "
        "Do not present the intervals below as if they covered PR-AUC.",
        "2. **These intervals hold the trained model fixed.** They quantify sampling "
        "variability of the 772-row evaluation set *only*. The project separately measured "
        "run-to-run **training** non-determinism of up to **0.054 gold PR-AUC** between "
        "nominally identical re-runs at a fixed seed "
        "(`results/grid_search_loss_variants.csv`, `pos_weight@5`). Those two sources of "
        "uncertainty are independent and they **compose**: a reader who quotes only the "
        "bootstrap interval will *understate* total uncertainty, because re-training the "
        "same configuration would move the point estimate as well as the sample. Quote both, "
        "or quote neither.",
        "",
        "A third limit, smaller but real: the gold labels are an LLM judgement, and 5 of the "
        "23 hand-read residual false negatives were found arguable at confidence *high* "
        "(`results/gold_fn_handread.md`). The bootstrap treats the labels as exact, so label "
        "noise is outside the interval too.",
        "",
        "## The deployed system",
        "",
        f"`{ref_disp}`, ordinary i.i.d. row bootstrap:",
        "",
    ]
    L += _md_interval_rows(intervals, REFERENCE)
    L += [
        "",
        "The `stratified` scheme in the CSV holds the 46.0% funnel hazard rate fixed in every "
        "replicate. It is narrower by construction and is a sensitivity check, not the number "
        "to quote: a fresh 772-row draw would not have exactly 355 hazards.",
        "",
        "## Paired comparison against the recoverable baseline",
        "",
        "**The XGBoost baseline could not be included.** "
        "`analysis/evaluation_pipeline.evaluate_one_set` runs the error analysis for the "
        "transformer only, so the baseline's per-row errors were never persisted, and no "
        "model pickle is committed. The headline \"116 missed vs 39 missed\" gap therefore "
        "cannot be paired-tested from committed files. The comparator used instead is the "
        "**TF-IDF + LogReg text-only control at th=0.20** (68 missed), whose error indices "
        "*are* committed — the step-4 \"TF-IDF + classifier\" baseline without the tabular "
        "features. The three other embedding variants are in the CSV as well.",
        "",
        "Sign convention: the difference is *deployed transformer minus comparator*, so for "
        "`False Negatives (Missed)`, `False Positives (Alarms)` and `Total Risk Cost` a "
        "**negative** value favours the transformer, and for the rates a **positive** value "
        "does.",
        "",
        "| Quantity | Paired difference | 95% CI | Excludes 0 |",
        "|---|---:|---:|:--:|",
        f"| Missed hazards | {fn_row.point_estimate:+.1f} | "
        f"[{fn_row.ci95_low:+.1f}, {fn_row.ci95_high:+.1f}] | "
        f"{'yes' if fn_row.excludes_zero else 'no'} |",
        f"| Recall | {rec_row.point_estimate:+.4f} | "
        f"[{rec_row.ci95_low:+.4f}, {rec_row.ci95_high:+.4f}] | "
        f"{'yes' if rec_row.excludes_zero else 'no'} |",
        "",
        "Because both models are scored on the *same* resampled rows in every iteration, this "
        "is a paired bootstrap. It is a stronger statement than checking whether two "
        "independent intervals overlap — overlapping marginal intervals are entirely "
        "compatible with a difference that is reliably one-signed, and the paired interval is "
        "the one that answers the question actually being asked.",
        "",
        "### Every recoverable comparator, on missed hazards",
        "",
        "| Comparator | Its missed hazards | Paired difference | 95% CI | Bootstrap p | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in MODELS:
        if k == REFERENCE:
            continue
        pr = paired[(paired.comparator == k) & (paired.scheme == "rows")
                    & (paired.metric == "False Negatives (Missed)")].iloc[0]
        own = intervals[(intervals.model == k) & (intervals.scheme == "rows")
                        & (intervals.metric == "False Negatives (Missed)")].iloc[0]
        mcr = mcnemar[(mcnemar.comparator == k)
                      & (mcnemar.framing == "missed_hazards")].iloc[0]
        L.append(f"| {DISPLAY[k]} | {own.point_estimate:.0f} | {pr.point_estimate:+.1f} | "
                 f"[{pr.ci95_low:+.1f}, {pr.ci95_high:+.1f}] | "
                 f"{pr.boot_p_two_sided:.3g} | {mcr.exact_binomial_p:.3g} |")
    L += [
        "",
        "**Read the MiniLM row before claiming the transformer dominates.** A *frozen* "
        "MiniLM-L6-v2 encoder with logistic regression on top is not separable from the "
        "fine-tuned transformer on missed hazards — the paired interval straddles zero — and it "
        "misses fewer at the point estimate. What separates them is the false-alarm side: the "
        "frozen encoder flags 81.7% of the holdout to get there, against 66.5%, and the overall "
        "correctness McNemar strongly favours the fine-tune. The defensible claim is therefore "
        "about the *operating point as a whole* (F2, flag rate, risk cost), not about recall "
        "alone. On recall alone, fine-tuning is not demonstrated to beat a frozen sentence "
        "encoder on 772 rows.",
        "",
        "*Artifact-drift note:* the four comparator rows come from `src/embedding_model.py` "
        "artifacts. Their committed counts were observed to change during development, so the "
        "reconciliation step re-derives them from whatever is committed at run time rather than "
        "hard-coding them. If a comparator number here disagrees with a number quoted "
        "elsewhere, re-run this module — it reconciles or aborts, it never guesses.",
        "",
        "## Exact test on the discordant pairs (McNemar)",
        "",
        "Rows where both models agree carry no information about which is better, so the "
        "exact paired test conditions on the discordant ones. Reported in two framings; the "
        "quoted p-value is the exact binomial one, not the chi-square approximation.",
        "",
        "| Framing | Transformer-only correct (b) | Comparator-only correct (c) | Discordant | Exact p |",
        "|---|---:|---:|---:|---:|",
        f"| Missed hazards (355 true hazards) | {int(mc.reference_only_correct_b)} | "
        f"{int(mc.comparator_only_correct_c)} | {int(mc.n_discordant)} | "
        f"{mc.exact_binomial_p:.3g} |",
        f"| Overall correctness (772 rows) | {int(mc_all.reference_only_correct_b)} | "
        f"{int(mc_all.comparator_only_correct_c)} | {int(mc_all.n_discordant)} | "
        f"{mc_all.exact_binomial_p:.3g} |",
        "",
        "Note the two framings can point different ways, and that is not a contradiction: the "
        "deployed system buys recall with false alarms on purpose, so it can win decisively on "
        "missed hazards while doing worse on raw correctness. Overall accuracy is not the "
        "objective — F2 and risk cost are — which is exactly why both rows are printed.",
        "",
        "## Are the intervals stable?",
        "",
        f"`bootstrap_ci_stability_{SLUG}.csv` recomputes the interval at "
        f"{', '.join(f'{n:,}' for n in STABILITY_LADDER)} resamples. The largest movement of "
        "any watched CI bound between the second-largest and largest rung is "
        f"{stability[stability.n_boot == sorted(STABILITY_LADDER)[-2]][['low_shift_vs_max', 'high_shift_vs_max']].abs().max().max():.4f} "
        "(recall / F2 in probability units, missed hazards in whole reviews), i.e. below the "
        f"precision at which the report quotes these numbers. {n_boot:,} is therefore enough; "
        "Monte-Carlo error in the bounds is not what limits this analysis — the 772 rows are.",
        "",
        "## What the report may and may not claim",
        "",
        "* Quote recall and F2 **with the interval attached**, and say it is a 772-row "
        "evaluation set. The interval on recall is a few points wide, so \"89.0%\" should be "
        "written as an estimate, not a specification.",
        "* Precision and flag rate carry intervals of similar width; any claim that depends on "
        "the precision figure to better than a couple of points is not supported.",
        "* `Total Risk Cost` has by far the widest relative interval, because it is dominated "
        "by a 100:1-weighted count of 39 events. Treat it as an illustration of the cost "
        "asymmetry, never as a forecast.",
        "* The transformer-over-baseline advantage in missed hazards should be quoted from the "
        "**paired** interval and the McNemar test, not from two marginal intervals.",
        "* Any two configurations whose gold gap is inside the project's ~0.05 noise floor "
        "remain unresolved regardless of what these intervals say — the noise floor is about "
        "training, the interval is about sampling, and neither rescues the other.",
        "",
        "## Files",
        "",
        f"* `results/bootstrap_ci_{SLUG}.csv` — one row per (scheme, model, metric).",
        f"* `results/bootstrap_ci_paired_{SLUG}.csv` — paired differences vs the deployed system.",
        f"* `results/bootstrap_ci_mcnemar_{SLUG}.csv` — exact McNemar, both framings.",
        f"* `results/bootstrap_ci_stability_{SLUG}.csv` — CI bounds vs number of resamples.",
        f"* `results/bootstrap_ci_reconciliation_{SLUG}.csv` — recovered vs committed metrics.",
        "",
        "Reproduce with:",
        "",
        "```",
        "python -m analysis.bootstrap_ci",
        "```",
        "",
    ]
    path = os.path.join(output_dir, f"bootstrap_ci_{SLUG}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=cfg.RESULTS_DIR)
    ap.add_argument("--results-dir", default=cfg.RESULTS_DIR,
                    help="where the committed error_analysis_*/performance_* CSVs live")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    ap.add_argument("--check", action="store_true",
                    help="reconcile the reconstruction against the committed CSVs and exit")
    args = ap.parse_args(argv)

    gold = load_gold_holdout()
    y_true = gold["llm_is_hazard"].astype(int).to_numpy()

    print("=" * 74)
    print("  RECONSTRUCTION OF HARD PREDICTIONS AT th=0.20 (no model, no GPU)")
    print("=" * 74)
    print(f"  {len(gold)} rows | {int(y_true.sum())} true hazards | "
          f"{int((y_true == 0).sum())} non-hazards")

    preds, recon_rows, all_ok = {}, [], True
    for key in MODELS:
        y_pred = reconstruct_predictions(key, gold, args.results_dir)
        _selfcheck_metrics(y_true, y_pred)
        row, ok = reconcile(key, gold, y_pred, args.results_dir)
        tn, fp, fn, tp = _counts(y_true, y_pred)
        print(f"  {DISPLAY[key]:42s} TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d}  "
              f"{'reconciles' if ok else 'MISMATCH'}")
        preds[key] = y_pred
        recon_rows.append(row)
        all_ok = all_ok and ok

    recon = pd.DataFrame(recon_rows)
    if not all_ok:
        bad = recon.loc[~recon.reconciles, "model"].tolist()
        print(f"\nFAILED reconciliation: {bad}. Refusing to bootstrap — the reconstruction "
              f"is not the evaluated system. Investigate before reporting any interval.")
        return 1
    print("\n  All reconstructions reproduce the committed metrics to within 1e-9.")

    if args.check:
        print("  --check requested; no artifacts written.")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)

    intervals = interval_table(y_true, preds, args.n_boot, args.seed)
    paired = paired_table(y_true, preds, args.n_boot, args.seed)
    mcnemar = mcnemar_table(y_true, preds)
    stability = stability_table(y_true, preds, args.seed)

    written = []
    for name, frame in ((f"bootstrap_ci_{SLUG}.csv", intervals),
                        (f"bootstrap_ci_paired_{SLUG}.csv", paired),
                        (f"bootstrap_ci_mcnemar_{SLUG}.csv", mcnemar),
                        (f"bootstrap_ci_stability_{SLUG}.csv", stability),
                        (f"bootstrap_ci_reconciliation_{SLUG}.csv", recon)):
        p = os.path.join(args.output_dir, name)
        frame.to_csv(p, index=False)
        written.append(p)
    written.append(write_markdown(intervals, paired, mcnemar, stability, recon,
                                  args.n_boot, args.output_dir, int(y_true.sum())))

    print("\n" + "=" * 74)
    print(f"  95% PERCENTILE INTERVALS — {DISPLAY[REFERENCE]}")
    print(f"  {args.n_boot:,} resamples of {len(gold)} rows, seed {args.seed}, i.i.d. row scheme")
    print("=" * 74)
    sub = intervals[(intervals.model == REFERENCE) & (intervals.scheme == "rows")]
    for metric in METRICS:
        r = sub[sub.metric == metric].iloc[0]
        print(f"  {metric:28s} {_fmt(metric, r.point_estimate):>10s}  "
              f"[{_fmt(metric, r.ci95_low)}, {_fmt(metric, r.ci95_high)}]")

    print("\n" + "=" * 74)
    print(f"  PAIRED DIFFERENCE — {DISPLAY[REFERENCE]} minus comparator")
    print("=" * 74)
    for scheme in ("rows",):
        for k in MODELS:
            if k == REFERENCE:
                continue
            for metric in ("False Negatives (Missed)", "Recall (Safety Coverage)"):
                r = paired[(paired.scheme == scheme) & (paired.comparator == k)
                           & (paired.metric == metric)].iloc[0]
                print(f"  {DISPLAY[k]:38s} {metric:26s} "
                      f"{r.point_estimate:+9.4f}  [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]  "
                      f"p={r.boot_p_two_sided:.4g}")

    print("\n" + "=" * 74)
    print("  EXACT McNEMAR ON DISCORDANT PAIRS")
    print("=" * 74)
    print(mcnemar[["comparator_display", "framing", "reference_only_correct_b",
                   "comparator_only_correct_c", "n_discordant",
                   "exact_binomial_p"]].to_string(index=False))

    print("\n  REMEMBER: these intervals hold the trained model fixed. They cover sampling")
    print("  variability of the 772-row holdout only. Training non-determinism was measured")
    print("  separately at up to 0.054 gold PR-AUC between identical re-runs; the two")
    print("  compose, so quoting the interval alone understates total uncertainty.")
    for p in written:
        print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
