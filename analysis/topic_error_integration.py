"""
Cross the project's two course techniques: project the transformer's gold-holdout
errors onto the NMF topic space.

Why this exists
---------------
Topic modelling (`src/topic_model.py`) and the fine-tuned classifier (`src/sota_model.py`)
were built as separate deliverables and never met. That is a real weakness: two techniques
sitting side by side is not the same as two techniques informing each other. This module
closes it with the one question that needs both of them at once:

    *Which hazard vocabularies does the classifier handle, and which does it not?*

Topic modelling supplies the vocabularies — the NMF K=6 fit is the project's headline
unsupervised model, and its topics are lexical clusters over hazard language. The
classifier supplies the errors. Crossing them turns "197 false positives" into a
statement about *language*, which is what the business problem is actually about.

How it works, and the one caveat
--------------------------------
1. Refit NMF K=6 exactly as `src/topic_model.py` does — same vectoriser settings, same
   `nndsvda` init, same seed, same fit corpus (the 1,500 heuristic-flagged hazard rows of
   the enriched dataset). The refit reproduces the committed
   `results/topic_model_topics.csv` exactly on 5 of the 6 topics; topic 4 differs only in
   the ordering of its top-word tail (a near-tie in term loadings across sklearn builds)
   and its per-topic NPMI moves 0.011 -> -0.001. The topic *structure* is identical, and
   the assignment below depends on the factorisation, not on the word ordering. We record
   the discrepancy rather than suppress it.

2. `transform` all 772 gold-holdout reviews through the frozen vectoriser and NMF. Note
   what this is and is not: the holdout is out-of-corpus for the topic model, so this is a
   projection onto topics learned elsewhere, not a refit. Terms absent from the fit
   vocabulary are dropped by the vectoriser. Rows whose projection is all-zero (no fit
   vocabulary present at all) are reported as an explicit `unassigned` bucket rather than
   forced into topic 0.

3. Reconstruct the deployed model's predictions on those 772 rows from
   `results/error_analysis_deberta_gold_llm_label_fresh_holdout_detail.csv`: `y_pred` is
   `y_true` with exactly the recorded FP/FN indices flipped, which reproduces the
   committed confusion matrix (197 FP, 39 FN) by construction. This is the same
   reconstruction trick `analysis/rebucket_errors.py` uses, and it needs no GPU.

4. Report per topic: support, base rate, precision, recall, and the share of the
   project's FP and FN mass that lands there. Then cross the topic assignment against
   the failure-mode taxonomy of `analysis/error_analysis.py`, which is an independent
   instrument: if the topic the model was never shown and the hand-written regex
   taxonomy agree on where the errors concentrate, that is two instruments on one
   phenomenon rather than one instrument restated.

Nothing here re-trains the classifier and nothing here re-selects a topic model. Both
inputs are frozen artifacts.

Usage
-----
    python -m analysis.topic_error_integration
"""

import os
import sys

import numpy as np
import pandas as pd

import config.settings as cfg
from src.topic_model import _vectorizers, fit_one

NMF_K = 6
DETAIL_PATH = "error_analysis_deberta_gold_llm_label_fresh_holdout_detail.csv"
OUTPUT_NAME = "topic_error_integration.csv"
GOLD_LABEL = "llm_is_hazard"


def fit_reference_nmf(k=NMF_K):
    """Refit the committed NMF K=6 model on its original fit corpus."""
    enriched = pd.read_csv(cfg.INPUT_DATA_PATH)
    hazards = enriched[enriched[cfg.TARGET_COLUMN] == 1]
    texts = hazards[cfg.TEXT_COLUMN].fillna("").tolist()

    count_vec, tfidf_vec = _vectorizers()
    X_count = count_vec.fit_transform(texts)
    X_tfidf = tfidf_vec.fit_transform(texts)
    fit = fit_one(texts, k, "nmf", count_vec, tfidf_vec, X_count, X_tfidf)
    return fit, tfidf_vec, len(texts)


def check_reproduction(fit, output_dir):
    """Compare the refit topics against the committed artifact, term by term."""
    path = os.path.join(output_dir, "topic_model_topics.csv")
    committed = pd.read_csv(path)
    committed = committed[(committed.algorithm == "NMF") & (committed.k == NMF_K)]
    exact = 0
    for i, words in enumerate(fit["top_words"]):
        row = committed[committed.topic == i]
        if len(row) and ", ".join(words) == row.iloc[0].top_words:
            exact += 1
    print(f"Refit reproduces {exact}/{len(fit['top_words'])} committed NMF K={NMF_K} "
          f"topics exactly on their top-{len(fit['top_words'][0])} term lists")
    return exact


def reconstruct_predictions(holdout, output_dir):
    """
    Rebuild the deployed model's hard predictions on the 772 holdout rows from the
    committed error-detail CSV. `y_pred` = `y_true` with the recorded errors flipped.
    """
    detail = pd.read_csv(os.path.join(output_dir, DETAIL_PATH))
    y_true = holdout[GOLD_LABEL].astype(int).to_numpy()
    y_pred = y_true.copy()
    fp_idx = detail.loc[detail.error_type == "FP", "index"].to_numpy()
    fn_idx = detail.loc[detail.error_type == "FN", "index"].to_numpy()
    y_pred[fp_idx] = 1
    y_pred[fn_idx] = 0

    assert ((y_pred == 1) & (y_true == 0)).sum() == len(fp_idx)
    assert ((y_pred == 0) & (y_true == 1)).sum() == len(fn_idx)
    print(f"Reconstructed predictions: {len(fp_idx)} FP, {len(fn_idx)} FN "
          f"on {len(y_true)} rows (base rate {y_true.mean():.3f})")
    return y_true, y_pred


def assign_topics(holdout, tfidf_vec, model):
    """Project holdout reviews onto the frozen topics; flag all-zero projections."""
    X = tfidf_vec.transform(holdout[cfg.TEXT_COLUMN].fillna(""))
    W = model.transform(X)
    topic = W.argmax(axis=1).astype(object)
    unassigned = W.sum(axis=1) == 0
    topic[unassigned] = "unassigned"
    print(f"Projected {len(holdout)} holdout rows; {int(unassigned.sum())} carried no "
          f"fit-corpus vocabulary and are reported as unassigned")
    return topic


def build_table(topic, y_true, y_pred, fit, output_dir):
    n_fp = int(((y_pred == 1) & (y_true == 0)).sum())
    n_fn = int(((y_pred == 0) & (y_true == 1)).sum())
    committed = pd.read_csv(os.path.join(output_dir, "topic_model_topics.csv"))
    committed = committed[(committed.algorithm == "NMF") & (committed.k == NMF_K)]

    rows = []
    for t in sorted(set(topic), key=lambda x: (x == "unassigned", x)):
        m = topic == t
        yt, yp = y_true[m], y_pred[m]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        if t == "unassigned":
            words, npmi = "", np.nan
        else:
            r = committed[committed.topic == t].iloc[0]
            words, npmi = r.top_words, float(r.coherence_npmi)
        rows.append({
            "topic": t,
            "top_words": words,
            "topic_npmi": npmi,
            "n_holdout": int(m.sum()),
            "gold_hazard_rate": float(yt.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "share_of_all_fp": fp / n_fp if n_fp else np.nan,
            "share_of_all_fn": fn / n_fn if n_fn else np.nan,
        })
    return pd.DataFrame(rows)


def mode_crosstab(topic, output_dir):
    """Failure mode (regex taxonomy) x NMF topic, for the 197 FPs and 39 FNs."""
    detail = pd.read_csv(os.path.join(output_dir, DETAIL_PATH))
    detail["topic"] = [topic[i] for i in detail["index"]]
    out = {}
    for kind in ("FP", "FN"):
        sub = detail[detail.error_type == kind]
        ct = pd.crosstab(sub.primary_mode, sub.topic)
        ct["total"] = ct.sum(axis=1)
        out[kind] = ct.sort_values("total", ascending=False)
    return out


def concentration_tests(fp_ct, pairs=(("neutral_allergen_mention", (1,)),
                                      ("illness_mentioned_not_caused_here", (0, 5)))):
    """
    Is a named failure mode over-represented in a named topic set, relative to all other
    false positives? A 2x2 Fisher exact test, run post-hoc: these cells were chosen after
    reading the crosstab, so the Bonferroni threshold over all mode x topic cells is
    reported alongside the raw p-value.
    """
    from scipy.stats import fisher_exact

    total = int(fp_ct["total"].sum())
    n_cells = (fp_ct.shape[0]) * (fp_ct.shape[1] - 1)
    lines = []
    for mode, topics in pairs:
        if mode not in fp_ct.index:
            continue
        mode_total = int(fp_ct.loc[mode, "total"])
        in_mode_in_topics = int(sum(fp_ct.loc[mode, t] for t in topics))
        in_topics = int(sum(fp_ct[t].sum() for t in topics))
        table = [[in_mode_in_topics, mode_total - in_mode_in_topics],
                 [in_topics - in_mode_in_topics,
                  total - mode_total - (in_topics - in_mode_in_topics)]]
        odds, p = fisher_exact(table)
        lines.append(
            f"{mode} in topic(s) {list(topics)}: {in_mode_in_topics}/{mode_total} "
            f"({in_mode_in_topics / mode_total:.1%}) vs "
            f"{table[1][0]}/{total - mode_total} of all other FPs; "
            f"odds ratio {odds:.2f}, p = {p:.2e} "
            f"(Bonferroni threshold over {n_cells} cells = {0.05 / n_cells:.1e})")
    return lines


def run(output_dir=None):
    output_dir = output_dir or cfg.RESULTS_DIR
    fit, tfidf_vec, n_fit = fit_reference_nmf()
    check_reproduction(fit, output_dir)

    holdout = pd.read_csv(cfg.GOLD_HOLDOUT_PATH)
    y_true, y_pred = reconstruct_predictions(holdout, output_dir)
    topic = assign_topics(holdout, tfidf_vec, fit["model"])
    table = build_table(topic, y_true, y_pred, fit, output_dir)

    out = os.path.join(output_dir, OUTPUT_NAME)
    table.to_csv(out, index=False)
    cols = ["topic", "n_holdout", "gold_hazard_rate", "tp", "fp", "fn",
            "precision", "recall", "share_of_all_fp", "share_of_all_fn"]
    print()
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Precision tracks the per-topic base rate, so quote both or neither.
    fin = table[table.topic != "unassigned"]
    rho = fin["gold_hazard_rate"].corr(fin["precision"], method="spearman")
    print(f"\nSpearman rho(per-topic gold hazard rate, per-topic precision) = {rho:.3f} "
          f"over {len(fin)} topics — precision is not independent of base rate")

    cts = mode_crosstab(topic, output_dir)
    tests = concentration_tests(cts["FP"])
    txt = os.path.join(output_dir, "topic_error_integration_crosstabs.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"Failure mode x NMF K={NMF_K} topic, deployed DeBERTa-v3 @ th=0.20 "
                f"on the 772-row gold holdout\n")
        f.write(f"Spearman rho(gold hazard rate, precision) across topics = {rho:.3f}\n")
        for kind, ct in cts.items():
            f.write(f"\n=== {kind} ===\n{ct.to_string()}\n")
            print(f"\n=== {kind}: mode x topic ===\n{ct.to_string()}")
        f.write("\n=== Post-hoc concentration tests (Fisher exact, 2x2, two-sided) ===\n")
        f.write("Selected after inspecting the table above, so treat as post-hoc; a\n")
        f.write("Bonferroni correction over all 9 modes x 6 topics (54 tests) is noted.\n")
        for line in tests:
            f.write(line + "\n")
            print(line)

    print(f"\nFit corpus: {n_fit} heuristic-flagged hazard reviews")
    print(f"Wrote {out}\nWrote {txt}")
    return table


if __name__ == "__main__":
    run()
    sys.exit(0)
