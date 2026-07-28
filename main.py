import gc
import json
import os

import optuna
import pandas as pd
import torch

import config.settings as cfg
from src.data_pipeline import load_and_split_data
from src.baseline_model import train_and_evaluate_baseline
from src.sota_model import run_sota_training, selection_disagreement

# Prevent system warnings from cluttering training feedback logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# --------------------------------------------------------------------------------
# Weights & Biases is optional. It used to be a hard `import wandb` + unconditional
# `wandb.login()`, so a clean clone with no W&B account could not run the sweep at
# all. Now the sweep degrades to local-only logging; the CSV in results/ is the
# artifact that matters and it is written either way.
#   Disable explicitly with:  WANDB_MODE=disabled
# --------------------------------------------------------------------------------
def _init_wandb():
    if os.getenv("WANDB_MODE", "").lower() == "disabled" or \
       os.getenv("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("W&B disabled by environment — logging locally only.")
        return None
    try:
        import wandb
    except ImportError:
        print("W&B not installed — logging locally only. "
              "`pip install wandb` or set WANDB_MODE=disabled to silence this.")
        # Also stop the Trainer's own WandbCallback from trying.
        os.environ["WANDB_DISABLED"] = "true"
        return None
    try:
        wandb.login()
        return wandb
    except Exception as e:                                    # no key, no network, ...
        print(f"W&B login failed ({e}) — continuing with local logging only.")
        # Critical: without this the Trainer's WandbCallback would still call
        # wandb.init() in this same process and block or crash on the missing key.
        os.environ["WANDB_DISABLED"] = "true"
        return None


class _NullRun:
    """Stand-in so the objective body needs no `if wandb` branches."""

    def finish(self):
        pass


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

    # Initialize the Weights & Biases session, if one is available at all.
    wandb = _init_wandb()

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
        ) if wandb else _NullRun()

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
            #
            # This is no longer only an a-priori argument. THIS SWEEP CAUGHT ONE:
            # results/optuna_trials.csv trial 1 (lr=3.80e-05, batch_size=4) collapsed
            # to all-positive — pred_positive_rate 1.000, recall 1.000, precision
            # 0.200 (= the validation base rate), pr_auc 0.196. Its hard predictions
            # stayed all-positive at every epoch (F2 pinned at the analytic 0.5556),
            # though its ranking partially recovered (best-epoch pr_auc 0.643). Under
            # `eval_recall` Optuna would have crowned it best of three; under
            # `eval_pr_auc` it placed last, 0.196 against 0.988.
            # Attribution, with its limit stated: the weight was focal_asymmetric@50
            # in every trial, and the 8-cell grid shows w=50 does NOT collapse at the
            # tuned learning rate, so the weight is not SUFFICIENT. But with one loss
            # setting in the sweep and no low-weight/high-lr cell anywhere, a
            # learning-rate x weight interaction is not excluded. Say collapse
            # required the high lr; do not claim the lr was isolated.
            #
            # PR-AUC is also threshold-free, so the hyperparameter search stays
            # independent of our 0.20 operating point.
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

            if wandb:
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

        # Which selection metric would have picked which configuration? `recall` is
        # included on purpose: it is the metric this project REJECTED, and showing
        # what it would have chosen is the cheapest possible evidence for that
        # decision. On the committed sweep, recall picks the degenerate trial.
        print("\n--- Selection-metric comparison (across trials) ---")
        for metric in ("pr_auc", "f2", "f1", "recall"):
            if metric not in trials_df.columns:
                continue
            win = trials_df.loc[trials_df[metric].idxmax()]
            flag = ""
            if "pred_positive_rate" in trials_df.columns and win["pred_positive_rate"] > 0.95:
                flag = "   <-- DEGENERATE (flags everything)"
            print(f"  {metric:7s} picks trial {int(win['trial'])}: "
                  f"lr={win['learning_rate']:.2e} bs={int(win['batch_size'])} "
                  f"flag_rate={win.get('pred_positive_rate', float('nan')):.3f}{flag}")

        # Collapse alarm, stated loudly rather than buried in a column.
        if "pred_positive_rate" in trials_df.columns:
            degenerate = trials_df[trials_df["pred_positive_rate"] > 0.95]
            if not degenerate.empty:
                print(f"\n  [!] {len(degenerate)} of {len(trials_df)} trials collapsed to "
                      f"all-positive. Trials: {sorted(degenerate['trial'].astype(int))}. "
                      f"This is why the objective is {cfg.HPO_METRIC}, not recall.")

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
