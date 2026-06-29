import gc
import os
import torch
import wandb
import optuna

import config.settings as cfg
from src.data_pipeline import load_and_split_data
from src.baseline_model import train_and_evaluate_baseline
from src.sota_model import run_sota_training

# Prevent system warnings from cluttering training feedback logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.95"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.70"

def main():
    print("========================================================")
    print("     ALLERGEN & FOOD SAFETY HAZARD COMPASS ENGINE       ")
    print("========================================================\n")

    # Load and partition dataset
    train_df, test_df = load_and_split_data()

    # 1. Train and evaluate your traditional XGBoost baseline
    baseline_pipeline = train_and_evaluate_baseline(train_df, test_df)

    # Initialize your global Weights & Biases workspace session
    wandb.login()

    # 2. Define the Optuna hyperparameter optimization trial closure
    def objective(trial):
        trainer = None
        # Configure search boundaries for hyperparameter tuning
        suggested_lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        suggested_batch = trial.suggest_categorical("batch_size", [4, 8, 16])

        # Initialize W&B nested run logging
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "allergen-safety-compass"),
            name=f"optuna_trial_lr_{suggested_lr:.2e}_bs_{suggested_batch}",
            reinit=True,
            config={"learning_rate": suggested_lr, "batch_size": suggested_batch}
        )

        try:
            # Run deep learning fine-tuning for 2 quick epochs per trial
            trainer = run_sota_training(
                train_df=train_df,
                test_df=test_df,
                epochs=4,
                lr=suggested_lr,
                batch_size=suggested_batch
            )

            # Evaluate target metrics
            eval_metrics = trainer.evaluate()
            target_score = eval_metrics["eval_recall"]

            # Log metrics to your active dashboard
            wandb.log(eval_metrics)
            return target_score

        except Exception as e:
            print(f"Trial failed due to execution error: {str(e)}")
            return 0.0
        finally:
            run.finish()

            print("\n--- Flushing MPS Backend Allocation Pools ---")
            if trainer is not None:
                try:
                    # Explicitly shift model parameters off the GPU before deletion
                    trainer.model.cpu()
                except Exception:
                    pass
                del trainer

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()  # Flush the hardware allocation cache

    print("\n--- Starting Automated Hyperparameter Optimization Space Sweeps ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)  # Runs 3 trials to fit inside your 3-week resource budget

    print("\n========================================================")
    print("OPTIMIZATION SWEEP COMPLETE")
    print(f"Best Trial Score (Recall Optimization Target): {study.best_value:.4f}")
    print(f"Optimal Hyperparameters Found: {study.best_params}")
    print("========================================================")


if __name__ == "__main__":
    main()