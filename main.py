import gc
import json
import os

import optuna
import pandas as pd
import torch
import wandb

import config.settings as cfg
from src.data_pipeline import load_and_split_data
from src.baseline_model import train_and_evaluate_baseline
from src.sota_model import run_sota_training, selection_disagreement

# Prevent system warnings from cluttering training feedback logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def main():
    print("========================================================")
    print("     ALLERGEN & FOOD SAFETY HAZARD COMPASS ENGINE       ")
    print("========================================================\n")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Load and partition dataset. The validation split is what checkpoint selection
    # and the Optuna objective see; the test split is never consulted during the
    # sweep, so its numbers stay honest for the final report.
    train_df, val_df, test_df = load_and_split_data(with_validation=True)

    # 1. Train and evaluate your traditional XGBoost baseline
    train_and_evaluate_baseline(train_df, test_df)

    # Initialize your global Weights & Biases workspace session
    wandb.login()

    trial_records = []

    # 2. Define the Optuna hyperparameter optimization trial closure
    def objective(trial):
        trainer = None
        suggested_lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        suggested_batch = trial.suggest_categorical("batch_size", [4, 8, 16])

        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "allergen-safety-compass"),
            name=f"optuna_trial_lr_{suggested_lr:.2e}_bs_{suggested_batch}",
            reinit=True,
            config={"learning_rate": suggested_lr, "batch_size": suggested_batch,
                    "loss_variant": cfg.LOSS_VARIANT, "asymmetric_weight": cfg.ASYMMETRIC_WEIGHT},
        )

        try:
            trainer = run_sota_training(
                train_df=train_df,
                test_df=test_df,
                eval_df=val_df,
                epochs=4,
                lr=suggested_lr,
                batch_size=suggested_batch,
            )

            # Scores the validation split (eval_df), not test — the objective must
            # never see the reporting split.
            eval_metrics = trainer.evaluate()

            # PR-AUC, not recall. Recall alone is maximised by flagging every review,
            # so it cannot be a selection metric regardless of whether any particular
            # run degenerates — an all-positive model is unfalsifiable under it.
            # (A collapse was observed historically under the label-keyed pos_weight
            # loss, but it was never persisted to an artifact and the current
            # focal_asymmetric loss at the deployed weight does not collapse, so no
            # figures are quoted for it here.) PR-AUC is also threshold-free, so the
            # hyperparameter search stays independent of our 0.20 operating point.
            target_score = eval_metrics[f"eval_{cfg.HPO_METRIC}"]

            # Record what the alternative objectives would have picked, so the choice
            # of selection metric is an evidenced decision rather than an assertion.
            record = {
                "trial": trial.number,
                "learning_rate": suggested_lr,
                "batch_size": suggested_batch,
                **{k.replace("eval_", ""): v for k, v in eval_metrics.items()
                   if k.startswith("eval_") and isinstance(v, (int, float))},
            }
            disagreement = selection_disagreement(trainer)
            if disagreement:
                record.update(disagreement)
            trial_records.append(record)

            wandb.log(eval_metrics)

            print(f"    trial {trial.number}: pr_auc={eval_metrics.get('eval_pr_auc', float('nan')):.4f} "
                  f"f2={eval_metrics.get('eval_f2', float('nan')):.4f} "
                  f"recall={eval_metrics.get('eval_recall', float('nan')):.4f} "
                  f"precision={eval_metrics.get('eval_precision', float('nan')):.4f} "
                  f"pos_rate={eval_metrics.get('eval_pred_positive_rate', float('nan')):.3f}")

            return target_score

        except Exception as e:
            print(f"Trial failed due to execution error: {str(e)}")
            return 0.0
        finally:
            run.finish()

            print("\n--- Flushing Backend Allocation Pools ---")
            if trainer is not None:
                try:
                    trainer.model.cpu()
                except Exception:
                    pass
                del trainer

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n--- Hyperparameter Sweep (objective = {cfg.HPO_METRIC}) ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    # -------------------------------------------------------------------------
    # Persist the sweep so the reported numbers live in the repo, not only in W&B.
    # -------------------------------------------------------------------------
    if trial_records:
        trials_df = pd.DataFrame(trial_records).sort_values(cfg.HPO_METRIC, ascending=False)
        trials_path = os.path.join(cfg.RESULTS_DIR, "optuna_trials.csv")
        trials_df.to_csv(trials_path, index=False)
        print(f"\nTrial-level metrics saved to: {trials_path}")

        # Would F2 have picked a different configuration than PR-AUC?
        if "f2" in trials_df.columns:
            by_pr = trials_df.sort_values("pr_auc", ascending=False).iloc[0]
            by_f2 = trials_df.sort_values("f2", ascending=False).iloc[0]
            print("\n--- Selection-metric comparison (across trials) ---")
            print(f"  PR-AUC picks trial {int(by_pr['trial'])}: "
                  f"lr={by_pr['learning_rate']:.2e} bs={int(by_pr['batch_size'])}")
            print(f"  F2     picks trial {int(by_f2['trial'])}: "
                  f"lr={by_f2['learning_rate']:.2e} bs={int(by_f2['batch_size'])}")
            print(f"  Agree: {int(by_pr['trial']) == int(by_f2['trial'])}")

    best = {
        "objective_metric": cfg.HPO_METRIC,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "loss_variant": cfg.LOSS_VARIANT,
        "asymmetric_weight": cfg.ASYMMETRIC_WEIGHT,
        "decision_threshold": cfg.DECISION_THRESHOLD,
    }
    with open(os.path.join(cfg.RESULTS_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump(best, f, indent=2)

    print("\n========================================================")
    print("OPTIMIZATION SWEEP COMPLETE")
    print(f"Best Trial Score ({cfg.HPO_METRIC}): {study.best_value:.4f}")
    print(f"Optimal Hyperparameters Found: {study.best_params}")
    print("========================================================")


if __name__ == "__main__":
    main()
