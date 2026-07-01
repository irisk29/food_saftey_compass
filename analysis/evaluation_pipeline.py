import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, recall_score, precision_score
import config.settings as cfg

# Financial parameters for our custom business metric
COST_FALSE_NEGATIVE = 5000.0  # Public health liability per missed hazard
COST_FALSE_POSITIVE = 50.0  # Operational overhead cost per manual compliance audit
OUTPUT_DIR = "./results"


def run_production_evaluation(test_df, baseline_pipeline, hf_trainer, test_tokenized, optimal_th=0.20):
    """
    Runs an end-to-end evaluation suite comparing Baseline vs. SOTA.
    Generates performance tables, error analysis, text samples, and charts.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14})

    print("\n" + "=" * 70)
    print("      EXTRACTING PREDICTIONS & GENERATING PROBABILITIES")
    print("=" * 70)

    # Extract ground-truth arrays
    y_true = test_df[cfg.TARGET_COLUMN].values.astype(int)

    # 1. Generate Baseline (XGBoost) probabilities
    print("Extracting Baseline XGBoost probabilities...")
    base_probs = baseline_pipeline.predict_proba(test_df)[:, 1]

    # 2. Generate SOTA (DeBERTa-v3) probabilities via the Trainer object
    print("Extracting SOTA DeBERTa-v3 inference predictions...")
    sota_predictions = hf_trainer.predict(test_tokenized)

    # Handle HuggingFace raw logit arrays and pass through sigmoid transformation
    raw_logits = sota_predictions.predictions.flatten()
    sota_probs = 1 / (1 + np.exp(-raw_logits))

    # -------------------------------------------------------------------------
    # METRIC COMPUTATION & VARIANT DICTIONARY CORE
    # -------------------------------------------------------------------------
    def score_variant(probs, th):
        preds = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, preds)
        _, fp, fn, _ = cm.ravel()

        p_curve, r_curve, _ = precision_recall_curve(y_true, probs)

        return {
            "Recall (Safety Coverage)": f"{recall_score(y_true, preds, zero_division=0) * 100:.1f}%",
            "Precision (Alert Validity)": f"{precision_score(y_true, preds, zero_division=0) * 100:.1f}%",
            "PR-AUC (Minority Metric)": f"{auc(r_curve, p_curve):.4f}",
            "F1-Score": f"{f1_score(y_true, preds, zero_division=0):.4f}",
            "False Negatives (Missed)": int(fn),
            "False Positives (Alarms)": int(fp),
            "Total Risk Cost ($\mathcal{C}_{ops}$)": f"${((fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)):,.2f}"
        }

    print("\nComputing quantitative performance matrices...")
    summary_df = pd.DataFrame({
        "Baseline Model (XGBoost)": score_variant(base_probs, th=0.50),
        "Standard SOTA (DeBERTa th=0.50)": score_variant(sota_probs, th=0.50),
        "Optimized SOTA (DeBERTa th=0.20)": score_variant(sota_probs, th=optimal_th)
    }).T

    # Export metric table overview to console and CSV
    print("\n" + "-" * 70)
    print("                 SUMMARY PERFORMANCE PROFILE TABLE")
    print("-" * 70)
    print(summary_df.to_string())
    print("-" * 70)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance_profile.csv"))

    # -------------------------------------------------------------------------
    # VISUALIZATION GENERATION PLOTS
    # -------------------------------------------------------------------------
    print("\nPlotting presentation graphics charts...")

    # Visual 1: Precision-Recall Curve
    base_p, base_r, _ = precision_recall_curve(y_true, base_probs)
    sota_p, sota_r, _ = precision_recall_curve(y_true, sota_probs)

    plt.figure(figsize=(8, 5.5))
    plt.plot(base_r, base_p, label=f'Baseline XGBoost (AUC = {auc(base_r, base_p):.3f})', color='steelblue', lw=2)
    plt.plot(sota_r, sota_p, label=f'SOTA DeBERTa-v3  (AUC = {auc(sota_r, sota_p):.3f})', color='darkorange', lw=2.5)
    plt.xlabel('Recall (Safety Coverage)')
    plt.ylabel('Precision (Alert Validity)')
    plt.title('Visualization 1: Comparative Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_pr_curve.png"), dpi=300)
    plt.close()

    # Visual 2: Business Cost Curve
    th_grid = np.linspace(0.01, 0.99, 100)
    sota_costs = [(((y_true == 1) & (sota_probs < t)).sum() * COST_FALSE_NEGATIVE + (
                (y_true == 0) & (sota_probs >= t)).sum() * COST_FALSE_POSITIVE) for t in th_grid]
    base_costs = [(((y_true == 1) & (base_probs < t)).sum() * COST_FALSE_NEGATIVE + (
                (y_true == 0) & (base_probs >= t)).sum() * COST_FALSE_POSITIVE) for t in th_grid]

    plt.figure(figsize=(9, 5.5))
    plt.plot(th_grid, base_costs, label='Baseline XGBoost Risk Profile', color='steelblue', linestyle='--')
    plt.plot(th_grid, sota_costs, label='SOTA DeBERTa-v3 Risk Profile', color='darkorange', lw=2.5)
    plt.axvline(optimal_th, color='crimson', linestyle=':', label=f'Optimal Operating Boundary ({optimal_th:.2f})')
    plt.xlabel('Classification Probability Threshold')
    plt.ylabel('Total Operational & Liability Cost ($)')
    plt.title('Visualization 2: Business Liability Cost Optimization Curve')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "business_cost_optimization_curve.png"), dpi=300)
    plt.close()

    # Visual 3: Cost Confusion Matrices
    opt_preds = (sota_probs >= optimal_th).astype(int)
    cm = confusion_matrix(y_true, opt_preds)
    cost_matrix = np.array([[0, cm[0, 1] * COST_FALSE_POSITIVE], [cm[1, 0] * COST_FALSE_NEGATIVE, 0]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 13})
    axes[0].set_title('Panel A: Optimized SOTA Error Counts')
    axes[0].set_xticklabels(['Benign', 'Hazard']);
    axes[0].set_yticklabels(['Benign', 'Hazard'])

    sns.heatmap(cost_matrix, annot=True, fmt=',.0f', cmap='Oranges', ax=axes[1], cbar=False, annot_kws={"size": 13})
    axes[1].set_title('Panel B: Optimized SOTA Financial Impact ($)')
    axes[1].set_xticklabels(['Benign', 'Hazard']);
    axes[1].set_yticklabels(['Benign', 'Hazard'])
    plt.suptitle('Visualization 3: Cost-Weighted Error Diagnostic Matrices', fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_weighted_confusion_matrix.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # QUALITATIVE ERROR EXTRACTION LOGIC
    # -------------------------------------------------------------------------
    print("\nIsolating explicit text rows for qualitative review slides...")
    analysis_df = test_df.copy()
    analysis_df['probs'] = sota_probs
    analysis_df['preds'] = opt_preds

    # Extract True False Positives
    fps = analysis_df[(analysis_df[cfg.TARGET_COLUMN] == 0) & (analysis_df['preds'] == 1)]
    # Extract True False Negatives
    fns = analysis_df[(analysis_df[cfg.TARGET_COLUMN] == 1) & (analysis_df['preds'] == 0)]

    print("\n" + "=" * 70)
    print("                 QUALITATIVE ERROR ANALYSIS SAMPLES")
    print("========================================================")
    print("\n[False Positive Samples - Flagged as Hazard Mistakenly]")
    for i, row in fps.head(2).reset_index().iterrows():
        print(f"Sample {i + 1} (Probability Score: {row['probs']:.4f}):\nText: \"{row[cfg.TEXT_COLUMN]}\"\n")

    print("\n[False Negative Samples - True Safety Hazard Missed]")
    for i, row in fns.head(2).reset_index().iterrows():
        print(f"Sample {i + 1} (Probability Score: {row['probs']:.4f}):\nText: \"{row[cfg.TEXT_COLUMN]}\"\n")
    print("=" * 70)

    print(f"\nExecution success. Analytics saved cleanly to: {OUTPUT_DIR}")