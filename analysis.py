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

# Best configuration from the Optuna sweep in main.py. Kept as a literal so a final
# run is reproducible without re-running the sweep; results/best_hyperparameters.json
# takes precedence when it exists.
DEFAULT_LR = 1.8140198244240376e-05
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3


def _load_best_hyperparameters():
    path = os.path.join(cfg.RESULTS_DIR, "best_hyperparameters.json")
    if os.path.exists(path):
        with open(path) as f:
            best = json.load(f).get("best_params", {})
        lr = best.get("learning_rate", DEFAULT_LR)
        bs = best.get("batch_size", DEFAULT_BATCH_SIZE)
        print(f"Using swept hyperparameters from {path}: lr={lr:.3e}, batch_size={bs}")
        return lr, bs
    print(f"No sweep results found; using defaults lr={DEFAULT_LR:.3e}, bs={DEFAULT_BATCH_SIZE}")
    return DEFAULT_LR, DEFAULT_BATCH_SIZE


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
