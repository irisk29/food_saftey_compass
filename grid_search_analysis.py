import os
import gc

import torch
import pandas as pd

import config.settings as cfg
from src.data_pipeline import load_and_split_data
from src.sota_model import run_sota_training

# Candidate false-negative penalty weights for AsymmetricSafetyLoss.
# cfg.ASYMMETRIC_WEIGHT (50.0) was tuned on the old 7,500-sample dataset;
# on the smaller dataset it appears to push every row toward "hazard"
# (100% recall, 37.5% precision), so we sweep lower values here.
WEIGHT_CANDIDATES = [1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0]

# Keep epochs/lr/batch_size fixed across the sweep so the weight is the only
# thing varying between runs.
EPOCHS = 3
LR = 1.8140198244240376e-05
BATCH_SIZE = 16

OUTPUT_DIR = "./results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "grid_search_asymmetric_weight.csv")


def _free_device_memory():
    gc.collect()
    if cfg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif cfg.DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    print("========================================================")
    print("   GRID SEARCH: ASYMMETRIC SAFETY LOSS PENALTY WEIGHT   ")
    print("========================================================\n")

    train_df, test_df = load_and_split_data()

    results = []
    for w in WEIGHT_CANDIDATES:
        print(f"\n>>> Training with asymmetric_weight = {w}")

        trainer = run_sota_training(
            train_df=train_df,
            test_df=test_df,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            asymmetric_weight=w,
        )

        metrics = trainer.evaluate()
        results.append({
            "weight": w,
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"],
            "f1": metrics["eval_f1"],
        })

        print(f"    -> precision={metrics['eval_precision']:.3f} "
              f"recall={metrics['eval_recall']:.3f} f1={metrics['eval_f1']:.3f}")

        # Release the trainer/model before the next run so repeated fine-tunes
        # don't accumulate GPU/MPS memory across the sweep.
        del trainer
        _free_device_memory()

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "-" * 60)
    print("                 GRID SEARCH RESULTS (by F1)")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)

    best = results_df.iloc[0]
    print(f"\nBest weight by F1: w={best['weight']} "
          f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, f1={best['f1']:.3f})")
    print(f"Full results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
