import os
import config.settings as cfg

from src.data_pipeline import load_and_split_data
from src.baseline_model import train_and_evaluate_baseline
from src.sota_model import run_sota_training
from analysis.evaluation_pipeline import run_production_evaluation


def main():
    print("========================================================")
    print("     ALLERGEN & FOOD SAFETY HAZARD COMPASS ENGINE       ")
    print("========================================================\n")

    # Load and partition your balanced dataset
    train_df, test_df = load_and_split_data()

    # 1. Train your baseline XGBoost model
    baseline_pipeline = train_and_evaluate_baseline(train_df, test_df)

    # 2. Train and extract your optimized SOTA DeBERTa-v3 framework model
    # We pass your optimal tuned learning rate directly to the training run
    trainer = run_sota_training(
        train_df=train_df,
        test_df=test_df,
        epochs=3,  # Run full epochs for final evaluation
        lr=1.8140198244240376e-05,
        batch_size=16
    )

    # Prepare the HuggingFace dataset format to feed predictions cleanly
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

    def tokenize_fn(batch):
        return tokenizer(batch[cfg.TEXT_COLUMN], truncation=True, max_length=256)

    test_ds = Dataset.from_pandas(
        test_df[[cfg.TEXT_COLUMN, cfg.TARGET_COLUMN]].rename(columns={cfg.TARGET_COLUMN: "label"}))
    test_tokenized = test_ds.map(tokenize_fn, batched=True)

    # 3. TRIGGER THE FULL PERFORMANCE & DIAGNOSTIC VISUALIZATION ENGINE
    run_production_evaluation(
        test_df=test_df,
        baseline_pipeline=baseline_pipeline,
        hf_trainer=trainer,
        test_tokenized=test_tokenized,
        optimal_th=0.20  # Use the safety-optimized classification threshold
    )


if __name__ == "__main__":
    # Configure Apple Silicon environment variables
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()