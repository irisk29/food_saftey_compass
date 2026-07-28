"""
Final evaluation run: trains both models on fixed hyperparameters and produces every
artifact the write-up needs, under both ground truths.

Use `main.py` for the hyperparameter sweep; this script consumes its result.
"""

import json
import os

import config.settings as cfg
from src.data_pipeline import (
    heuristic_vs_llm_agreement,
    load_and_split_data,
    load_gold_holdout,
)
from src.baseline_model import train_and_evaluate_baseline
from src.sota_model import run_sota_training, selection_disagreement, tokenize_split
from analysis.evaluation_pipeline import run_production_evaluation
from analysis.error_analysis import analyze_label_disagreement

# -----------------------------------------------------------------------------
# Hyperparameter provenance. Read this before re-running.
#
# REPORTED_LR is the learning rate that actually produced every committed number in
# results/performance_*.csv, results/error_analysis_* and the figures. It came from an
# earlier Optuna sweep whose trial-level output was not persisted at the time.
#
# results/best_hyperparameters.json is now committed (from the sweep in main.py, whose
# trials are in results/optuna_trials.csv) and it names a slightly different rate:
# 1.8346e-05 vs 1.8140e-05, a 1.1% relative difference. Because _load_best_hyperparameters
# prefers the JSON, a re-run of this script trains at the NEW rate and will therefore
# not reproduce the committed CSVs bit-for-bit.
#
# That is the correct default going forward — the committed sweep should win over a
# literal — but it must be visible rather than silent, so the mismatch is warned about
# below. For context on whether it matters: the two rates differ by 1.1%, while
# re-training a single fixed configuration was measured to move gold PR-AUC by up to
# 0.054 (see CLAUDE.md, replicate pairs in the 8-cell grid). The hyperparameter
# difference is far inside that noise floor.
# -----------------------------------------------------------------------------
REPORTED_LR = 1.8140198244240376e-05      # produced the committed artifacts
REPORTED_BATCH_SIZE = 16
DEFAULT_LR = REPORTED_LR                  # fallback when no sweep JSON exists
DEFAULT_BATCH_SIZE = REPORTED_BATCH_SIZE
DEFAULT_EPOCHS = 3


def _load_best_hyperparameters():
    path = os.path.join(cfg.RESULTS_DIR, "best_hyperparameters.json")
    if not os.path.exists(path):
        print(f"No sweep results found; using the as-reported hyperparameters "
              f"lr={DEFAULT_LR:.6e}, bs={DEFAULT_BATCH_SIZE}")
        return DEFAULT_LR, DEFAULT_BATCH_SIZE

    with open(path) as f:
        best = json.load(f).get("best_params", {})
    lr = best.get("learning_rate", DEFAULT_LR)
    bs = best.get("batch_size", DEFAULT_BATCH_SIZE)
    print(f"Using swept hyperparameters from {path}: lr={lr:.6e}, batch_size={bs}")

    # Make any divergence from the as-reported configuration impossible to miss.
    if abs(lr - REPORTED_LR) / REPORTED_LR > 1e-6 or bs != REPORTED_BATCH_SIZE:
        print(
            "\n  [!] PROVENANCE WARNING — this run will NOT reproduce the committed CSVs.\n"
            f"      committed artifacts were trained at lr={REPORTED_LR:.6e}, bs={REPORTED_BATCH_SIZE}\n"
            f"      this run will train at        lr={lr:.6e}, bs={bs}\n"
            "      (relative lr difference: "
            f"{abs(lr - REPORTED_LR) / REPORTED_LR:.2%})\n"
            "      Set FSC_USE_REPORTED_HPARAMS=1 to pin the as-reported values instead.\n"
        )
        if os.getenv("FSC_USE_REPORTED_HPARAMS", "").lower() in {"1", "true", "yes"}:
            print(f"      FSC_USE_REPORTED_HPARAMS set — pinning lr={REPORTED_LR:.6e}, "
                  f"bs={REPORTED_BATCH_SIZE}.\n")
            return REPORTED_LR, REPORTED_BATCH_SIZE

    return lr, bs


def main():
    print("========================================================")
    print("     ALLERGEN & FOOD SAFETY HAZARD COMPASS ENGINE       ")
    print("========================================================\n")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. How trustworthy is the training label? Needs no model, so it runs first.
    # -------------------------------------------------------------------------
    agreement = heuristic_vs_llm_agreement()
    print("\n--- Heuristic label vs independent LLM judge ---")
    for k, v in agreement.items():
        print(f"  {k:44s} {v:.4f}" if isinstance(v, float) else f"  {k:44s} {v}")
    with open(os.path.join(cfg.RESULTS_DIR, "label_quality.json"), "w") as f:
        json.dump(agreement, f, indent=2)

    print("\n--- Why the label errs: categorised disagreement ---")
    analyze_label_disagreement()

    # -------------------------------------------------------------------------
    # 1. Models
    # -------------------------------------------------------------------------
    # Three-way split: checkpoint selection sees only the validation split, so the
    # test-split numbers reported below are out-of-selection (the gold holdout was
    # already out-of-everything). Same test split as the historical two-way split.
    train_df, val_df, test_df = load_and_split_data(with_validation=True)
    lr, batch_size = _load_best_hyperparameters()

    baseline_pipeline = train_and_evaluate_baseline(train_df, test_df)

    trainer, tokenizer, test_tokenized = run_sota_training(
        train_df=train_df,
        test_df=test_df,
        eval_df=val_df,
        epochs=DEFAULT_EPOCHS,
        lr=lr,
        batch_size=batch_size,
        return_tokenized=True,
    )

    disagreement = selection_disagreement(trainer)
    if disagreement:
        print(f"\n--- Checkpoint selection: F2 vs PR-AUC ---\n  {disagreement}")
        with open(os.path.join(cfg.RESULTS_DIR, "checkpoint_selection.json"), "w") as f:
            json.dump(disagreement, f, indent=2)

    # -------------------------------------------------------------------------
    # 2. The fresh holdout set — reviews that never entered the pipeline.
    # -------------------------------------------------------------------------
    gold_df = load_gold_holdout(require=False)
    gold_tokenized = tokenize_split(gold_df, tokenizer) if gold_df is not None else None

    # -------------------------------------------------------------------------
    # 3. Evaluate under both ground truths
    # -------------------------------------------------------------------------
    run_production_evaluation(
        test_df=test_df,
        baseline_pipeline=baseline_pipeline,
        hf_trainer=trainer,
        test_tokenized=test_tokenized,
        optimal_th=cfg.DECISION_THRESHOLD,
        gold_df=gold_df,
        gold_tokenized=gold_tokenized,
    )


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
