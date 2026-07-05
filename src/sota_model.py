import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import config.settings as cfg
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


class AsymmetricSafetyLoss(nn.Module):
    def __init__(self, weight_fn=cfg.ASYMMETRIC_WEIGHT):
        super().__init__()
        self.weight_fn = weight_fn
        # Use BCEWithLogitsLoss for numerical stability over raw probabilities
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        # Calculate base cross entropy values
        base_loss = self.bce(logits, targets)

        # Apply the 50x multiplier penalty factor specifically to False Negatives
        weight_mask = torch.ones_like(targets)
        weight_mask[targets == 1.0] = self.weight_fn

        return (base_loss * weight_mask).mean()


class CustomSafetyTrainer(Trainer):
    """Overrides the default cross-entropy loss function with our asymmetric safety metric."""

    def __init__(self, *args, asymmetric_weight=cfg.ASYMMETRIC_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.asymmetric_weight = asymmetric_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fn = AsymmetricSafetyLoss(weight_fn=self.asymmetric_weight)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Calculates evaluation metrics using a safety-optimized probability threshold."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits.flatten()))

    # LOWERED THRESHOLD: Flag as a hazard if the probability is greater than 20%
    preds = (probs > 0.20).astype(int)

    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }


def run_sota_training(train_df, test_df, epochs=3, lr=2e-5, batch_size=8, asymmetric_weight=None):
    """
    Tokenizes raw text sequences and runs the custom fine-tuning
    process using the DeBERTa-v3 architecture—patched for Apple Silicon stability.
    """
    if asymmetric_weight is None:
        asymmetric_weight = cfg.ASYMMETRIC_WEIGHT

    print(f"\n--- Initializing SOTA DeBERTa-v3 Base Architecture on {cfg.DEVICE} (w={asymmetric_weight}) ---")
    model_nm = "microsoft/deberta-v3-base"

    tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=False)

    # FIX 1: Explicitly define problem_type to stop HuggingFace from defaulting to MSE Loss
    model = AutoModelForSequenceClassification.from_pretrained(
        model_nm,
        num_labels=1,
        problem_type="binary_classification"
    )
    model.to(cfg.DEVICE)

    # Helper validation closure for tokenization alignment
    def tokenize_fn(batch):
        return tokenizer(batch[cfg.TEXT_COLUMN], truncation=True, max_length=256)

    # FIX 2: Explicitly force labels to float32 to prevent the 'square_i64' MPS crash
    train_df_patched = train_df.copy()
    test_df_patched = test_df.copy()
    train_df_patched[cfg.TARGET_COLUMN] = train_df_patched[cfg.TARGET_COLUMN].astype(np.float32)
    test_df_patched[cfg.TARGET_COLUMN] = test_df_patched[cfg.TARGET_COLUMN].astype(np.float32)

    # Format datasets cleanly using the patched dataframes
    train_ds = Dataset.from_pandas(
        train_df_patched[[cfg.TEXT_COLUMN, cfg.TARGET_COLUMN]].rename(columns={cfg.TARGET_COLUMN: "label"}))
    test_ds = Dataset.from_pandas(
        test_df_patched[[cfg.TEXT_COLUMN, cfg.TARGET_COLUMN]].rename(columns={cfg.TARGET_COLUMN: "label"}))

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    test_tokenized = test_ds.map(tokenize_fn, batched=True)

    # Calculate micro-batching steps dynamically
    # If target batch_size is 16, per_device is 4, accumulation_steps becomes 4
    per_device_batch = 4 if batch_size >= 8 else batch_size
    accum_steps = batch_size // per_device_batch

    # Set up runtime parameters for the deep learning training loop
    training_args = TrainingArguments(
        output_dir=cfg.MODEL_OUTPUT_DIR,
        learning_rate=lr,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=per_device_batch,  # Micro-batch size fed to GPU
        gradient_accumulation_steps=accum_steps,  # Accumulate steps before backpropagation
        eval_accumulation_steps=1,  # Offloads validation tensors step-by-step to host RAM
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="recall",
        report_to="wandb",
        fp16=False  # Keep False on Mac MPS as FP16 can occasionally cause nan gradients on older macOS arms
    )

    trainer = CustomSafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        asymmetric_weight=asymmetric_weight,
    )

    trainer.train()

    return trainer