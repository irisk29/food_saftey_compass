"""
Evaluation suite.

The central change from the first version: metrics are computed against two different
ground truths and reported side by side.

  1. The heuristic `is_hazard` label on the held-out test split. Measures how well the
     model reproduces the keyword+stars rule. Inflated by construction — the rule is
     only 73% precise against expert judgement — so it is reported as a *reference*,
     never as the headline.
  2. The LLM-judged label on the fresh holdout set (reviews that never entered the
     pipeline). Measures hazard detection. This is the number that goes in the slides.

The gap between the two is the most informative quantity the project produces: it says
how much of the original score was fidelity to a flawed proxy rather than skill.

PR-AUC leads every table. It is threshold-free, so it does not depend on the 0.20
operating point, and it is robust to the class-weighting asymmetry between the two
models (the baseline is class-balanced via scale_pos_weight, the transformer via the
asymmetric loss; comparing them at a shared fixed threshold would not be meaningful).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

import config.settings as cfg
from analysis.error_analysis import analyze_errors

# Financial parameters for our custom business metric.
# The 100:1 ratio is the justification for the lowered decision threshold; note it is
# NOT the same as the loss weight (cfg.ASYMMETRIC_WEIGHT = 50), and the write-up should
# say why: the loss weight is a training-time regulariser tuned empirically, while this
# ratio is a deployment-time cost model. Conflating them invites an obvious question.
COST_FALSE_NEGATIVE = 5000.0  # Public health liability per missed hazard
COST_FALSE_POSITIVE = 50.0    # Operational overhead per manual compliance audit


def score_variant(y_true, probs, th):
    """Full metric row at one threshold. Handles degenerate all-one-class predictions."""
    preds = (probs >= th).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        # Threshold-free and therefore the primary comparison metric.
        "PR-AUC": average_precision_score(y_true, probs),
        "Recall (Safety Coverage)": recall_score(y_true, preds, zero_division=0),
        "Precision (Alert Validity)": precision_score(y_true, preds, zero_division=0),
        "F2 (recall-weighted)": fbeta_score(y_true, preds, beta=cfg.FBETA, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        # Exposes the degenerate all-positive solution that perfect recall would hide.
        "Flag Rate": float(preds.mean()),
        "False Negatives (Missed)": int(fn),
        "False Positives (Alarms)": int(fp),
        "Total Risk Cost": (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE),
    }


def _format_table(raw):
    """Percentages and currency for display; the CSV keeps raw floats."""
    out = raw.copy()
    for col in ("Recall (Safety Coverage)", "Precision (Alert Validity)", "Flag Rate"):
        if col in out:
            out[col] = out[col].map(lambda v: f"{v * 100:.1f}%")
    for col in ("PR-AUC", "F2 (recall-weighted)", "F1"):
        if col in out:
            out[col] = out[col].map(lambda v: f"{v:.4f}")
    if "Total Risk Cost" in out:
        out["Total Risk Cost"] = out["Total Risk Cost"].map(lambda v: f"${v:,.0f}")
    return out


def sota_probabilities(hf_trainer, tokenized):
    logits = hf_trainer.predict(tokenized).predictions
    return 1 / (1 + np.exp(-np.asarray(logits).flatten()))


def baseline_probabilities(pipeline, df):
    return pipeline.predict_proba(df[cfg.TABULAR_FEATURES + [cfg.TEXT_COLUMN]])[:, 1]


def evaluate_one_set(name, df, y_true, base_probs, sota_probs,
                     optimal_th=None, output_dir=None, run_error_analysis=True):
    """
    Scores both models on one dataset against one ground truth.

    Returns (summary_df, dict of probability arrays) so callers can compose.
    """
    optimal_th = cfg.DECISION_THRESHOLD if optimal_th is None else optimal_th
    output_dir = output_dir or cfg.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)

    print("\n" + "=" * 74)
    print(f"  EVALUATION: {name}")
    print(f"  {len(y_true)} reviews | hazard base rate {y_true.mean():.1%}")
    print("=" * 74)

    columns = {}
    if base_probs is not None:
        columns["Baseline XGBoost (th=0.50)"] = score_variant(y_true, base_probs, 0.50)
    if sota_probs is not None:
        columns["DeBERTa-v3 (th=0.50)"] = score_variant(y_true, sota_probs, 0.50)
        columns[f"DeBERTa-v3 (th={optimal_th:.2f}, deployed)"] = score_variant(y_true, sota_probs, optimal_th)

    summary = pd.DataFrame(columns).T
    print(_format_table(summary).to_string())

    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    summary.to_csv(os.path.join(output_dir, f"performance_{slug}.csv"))

    if run_error_analysis and sota_probs is not None:
        preds = (sota_probs >= optimal_th).astype(int)
        analyze_errors(df, y_true=y_true, y_pred=preds, probs=sota_probs,
                       label=f"deberta_{slug}", output_dir=output_dir)

    return summary, {"baseline": base_probs, "sota": sota_probs}


def plot_comparison(y_true, base_probs, sota_probs, name, optimal_th=None, output_dir=None):
    optimal_th = cfg.DECISION_THRESHOLD if optimal_th is None else optimal_th
    output_dir = output_dir or cfg.RESULTS_DIR
    sns.set_theme(style="whitegrid")
    y_true = np.asarray(y_true).astype(int)
    slug = name.lower().replace(" ", "_")

    # --- PR curves ---
    plt.figure(figsize=(8, 5.5))
    for probs, label, color, lw in ((base_probs, "Baseline XGBoost", "steelblue", 2),
                                    (sota_probs, "DeBERTa-v3", "darkorange", 2.5)):
        if probs is None:
            continue
        p, r, _ = precision_recall_curve(y_true, probs)
        plt.plot(r, p, label=f"{label} (AP = {average_precision_score(y_true, probs):.3f})",
                 color=color, lw=lw)
    plt.axhline(y_true.mean(), color="grey", linestyle="--", lw=1,
                label=f"Random baseline ({y_true.mean():.3f})")
    plt.xlabel("Recall (Safety Coverage)")
    plt.ylabel("Precision (Alert Validity)")
    plt.title(f"Precision–Recall — {name}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pr_curve_{slug}.png"), dpi=300)
    plt.close()

    # --- Business cost curve ---
    if sota_probs is not None:
        th_grid = np.linspace(0.01, 0.99, 100)

        def cost_at(probs, t):
            return (((y_true == 1) & (probs < t)).sum() * COST_FALSE_NEGATIVE
                    + ((y_true == 0) & (probs >= t)).sum() * COST_FALSE_POSITIVE)

        plt.figure(figsize=(9, 5.5))
        if base_probs is not None:
            plt.plot(th_grid, [cost_at(base_probs, t) for t in th_grid],
                     label="Baseline XGBoost", color="steelblue", linestyle="--")
        sota_costs = [cost_at(sota_probs, t) for t in th_grid]
        plt.plot(th_grid, sota_costs, label="DeBERTa-v3", color="darkorange", lw=2.5)

        best_th = th_grid[int(np.argmin(sota_costs))]
        plt.axvline(optimal_th, color="crimson", linestyle=":",
                    label=f"Deployed threshold ({optimal_th:.2f})")
        plt.axvline(best_th, color="green", linestyle=":",
                    label=f"Cost-minimising threshold ({best_th:.2f})")
        plt.xlabel("Classification Probability Threshold")
        plt.ylabel("Total Operational & Liability Cost ($)")
        plt.title(f"Business Liability Cost vs Threshold — {name}")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cost_curve_{slug}.png"), dpi=300)
        plt.close()

        # Reported explicitly: if the empirically cost-minimising threshold differs
        # from the deployed 0.20, that is a finding, not an embarrassment.
        print(f"\n  Cost-minimising threshold on this set: {best_th:.2f} "
              f"(deployed: {optimal_th:.2f})")


def compare_ground_truths(heuristic_summary, gold_summary, output_dir=None):
    """
    The headline artifact: the same model scored against the heuristic label vs the
    independent LLM label. A large drop means the original numbers were measuring
    agreement with the labelling rule rather than hazard-detection skill.
    """
    output_dir = output_dir or cfg.RESULTS_DIR
    rows = []
    for model in heuristic_summary.index:
        if model not in gold_summary.index:
            continue
        for metric in ("PR-AUC", "Recall (Safety Coverage)", "Precision (Alert Validity)",
                       "F2 (recall-weighted)"):
            h, g = heuristic_summary.loc[model, metric], gold_summary.loc[model, metric]
            rows.append({
                "model": model,
                "metric": metric,
                "vs_heuristic_label": h,
                "vs_gold_llm_label": g,
                "delta": g - h,
            })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(output_dir, "ground_truth_comparison.csv"), index=False)

    print("\n" + "=" * 74)
    print("  HEURISTIC LABEL vs INDEPENDENT GOLD LABEL")
    print("=" * 74)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n  A negative delta is the portion of the original score that came from"
          "\n  reproducing the labelling rule rather than detecting hazards.")
    return comparison


def run_production_evaluation(test_df, baseline_pipeline, hf_trainer, test_tokenized,
                              optimal_th=None, gold_df=None, gold_tokenized=None,
                              output_dir=None):
    """
    End-to-end evaluation. If a gold holdout set is supplied, both ground truths are
    scored and compared; otherwise only the heuristic-label evaluation runs, with a
    warning that those numbers are not the ones to report.
    """
    optimal_th = cfg.DECISION_THRESHOLD if optimal_th is None else optimal_th
    output_dir = output_dir or cfg.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    heuristic_summary, probs = evaluate_one_set(
        name="Heuristic label (test split)",
        df=test_df,
        y_true=test_df[cfg.TARGET_COLUMN].values,
        base_probs=baseline_probabilities(baseline_pipeline, test_df),
        sota_probs=sota_probabilities(hf_trainer, test_tokenized),
        optimal_th=optimal_th,
        output_dir=output_dir,
    )
    plot_comparison(test_df[cfg.TARGET_COLUMN].values, probs["baseline"], probs["sota"],
                    "Heuristic label (test split)", optimal_th, output_dir)

    if gold_df is None or gold_tokenized is None:
        print("\n[warning] No gold holdout set supplied. The numbers above measure "
              "agreement with a labelling rule that is 73% precise against expert "
              "judgement — do not report them as detection performance.")
        return heuristic_summary, None, None

    gold_summary, gold_probs = evaluate_one_set(
        name="Gold LLM label (fresh holdout)",
        df=gold_df,
        y_true=gold_df[cfg.TARGET_COLUMN].values,
        base_probs=baseline_probabilities(baseline_pipeline, gold_df),
        sota_probs=sota_probabilities(hf_trainer, gold_tokenized),
        optimal_th=optimal_th,
        output_dir=output_dir,
    )
    plot_comparison(gold_df[cfg.TARGET_COLUMN].values, gold_probs["baseline"],
                    gold_probs["sota"], "Gold LLM label (fresh holdout)", optimal_th, output_dir)

    comparison = compare_ground_truths(heuristic_summary, gold_summary, output_dir)

    print(f"\nAll evaluation artifacts written to: {output_dir}")
    return heuristic_summary, gold_summary, comparison
