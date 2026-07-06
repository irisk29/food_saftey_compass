import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import INPUT_DATA_PATH, TEST_SIZE, TARGET_COLUMN, DEVICE
# =====================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =====================================================================
# INPUT_DATA_PATH = "path/to/your/llm_gold_dataset.csv"  # Target dataset source file
# TEST_SIZE = 0.2  # Determines the percentage of the dataset allocated to testing (0.2 -> 20%)
MODEL_NAME = "microsoft/deberta-v3-small"  # Core SOTA sequence encoder base[cite: 1]
MAX_LEN = 256  # Sequence tokenization window[cite: 1]


# =====================================================================
# 1. ATTENUATED ASYMMETRIC LOSS FUNCTION IMPLEMENTATION
# =====================================================================
class AttenuatedAsymmetricBCEWithLogitsLoss(nn.Module):
    """Custom Cost-Sensitive Asymmetric Binary Cross-Entropy Loss with dynamic

    epoch attenuation to protect precision while preserving recall under
    low data volumes[cite: 1].
    """

    def __init__(self, base_omega=5.0, annealing=True, total_epochs=4):
        super(AttenuatedAsymmetricBCEWithLogitsLoss, self).__init__()
        self.base_omega = base_omega
        self.annealing = annealing
        self.total_epochs = total_epochs
        self.current_omega = 1.0 if annealing else base_omega

    def update_omega(self, current_epoch):
        """Linearly transitions omega from a symmetric baseline (1.0) up to the

        calibrated safety multiplier to allow stable early weight convergence[cite: 1].
        """
        if self.annealing:
            fraction = min(current_epoch / max(1, self.total_epochs - 1), 1.0)
            self.current_omega = 1.0 + fraction * (self.base_omega - 1.0)
        else:
            self.current_omega = self.base_omega

    def forward(self, logits, targets):
        """Calculates loss weights dynamically with asymmetric multipliers applied

        explicitly to false negative omissions[cite: 1].
        """
        targets = targets.float()

        # Numerically stable calculation of standard log probabilities
        log_prob_pos = F.logsigmoid(logits)
        log_prob_neg = F.logsigmoid(-logits)

        # Apply active attenuated penalty multiplier directly to the positive class
        loss = -(
            self.current_omega * targets * log_prob_pos
            + (1.0 - targets) * log_prob_neg
        )

        return loss.mean()


# =====================================================================
# 2. LIVE DATA EXTRACTION LAYER
# =====================================================================
class FoodSafetyGoldDataset(Dataset):
    """Real-world Dataset abstraction layer mapping tokenizer features directly

    from the source dataframe records.
    """

    def __init__(self, dataframe, tokenizer, max_len=256):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_content = str(row["text"])  # Independent feature column[cite: 1]
        label_value = int(row[TARGET_COLUMN])  # Dependent target column[cite: 1]

        # Process unstructured text block using the transformer tokenizer
        encoding = self.tokenizer(
            text_content,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_value, dtype=torch.long),
        }


# =====================================================================
# 3. EVALUATION SUBROUTINES
# =====================================================================
def run_evaluation_cycle(model, dataloader, device, decision_threshold=0.50):
    """Runs a complete forward validation pass to compute performance vectors."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)

    probabilities = 1 / (1 + np.exp(-all_logits))
    predictions = (probabilities >= decision_threshold).astype(int)

    precision = precision_score(
        all_labels, predictions, zero_division=0, pos_label=1
    )
    recall = recall_score(all_labels, predictions, zero_division=0, pos_label=1)

    p_curve, r_curve, _ = precision_recall_curve(all_labels, probabilities)
    pr_auc = auc(r_curve, p_curve)

    true_positives = int(np.sum((predictions == 1) & (all_labels == 1)))
    false_negatives = int(np.sum((predictions == 0) & (all_labels == 1)))
    false_positives = int(np.sum((predictions == 1) & (all_labels == 0)))

    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "fns": false_negatives,
        "fps": false_positives,
        "tps": true_positives,
    }


# =====================================================================
# 4. MAIN ORCHESTRATION PIPELINE
# =====================================================================
def main():
    # Validate file presence before triggering ingestion layers
    if not os.path.exists(INPUT_DATA_PATH):
        raise FileNotFoundError(
            f"The specified source file path was not found: {INPUT_DATA_PATH}"
        )

    print(f"[DATA INGESTION] Reading source data from: {INPUT_DATA_PATH}")
    raw_df = pd.read_csv(INPUT_DATA_PATH)

    # Validate internal columns map safely to architectural spec expectations
    assert "text" in raw_df.columns, "Source file must contain a 'text' column."
    assert (
        TARGET_COLUMN in raw_df.columns
    ), f"Source file must contain an 'f{TARGET_COLUMN}' target column."

    # Split dataset based on percentage specified in the TEST_SIZE configuration
    print(
        f"[SPLIT ROUTINE] Executing stratified proportional split (Test Size Percentage: {TEST_SIZE * 100}%)"
    )
    train_df, test_df = train_test_split(
        raw_df,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=raw_df[TARGET_COLUMN],  # Preserves identical class distributions
    )
    print(
        f" -> Split Successful. Train Matrix: {len(train_df)} rows | Test Matrix: {len(test_df)} rows"
    )

    # Initialize Tokenization Layer
    print(f"[TOKENIZER INITIALIZATION] Downloading model assets: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Build PyTorch Dataset pipelines from proportional partitions
    train_dataset = FoodSafetyGoldDataset(train_df, tokenizer, max_len=MAX_LEN)
    test_dataset = FoodSafetyGoldDataset(test_df, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load authentic SOTA DeBERTa sequence classification framework[cite: 1]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Optimization Configuration: Attenuated Asymmetric Loss Settings[cite: 1]
    BASE_OMEGA = 5.0
    TOTAL_EPOCHS = 4
    DECISION_THRESHOLD = 0.50

    criterion = AttenuatedAsymmetricBCEWithLogitsLoss(
        base_omega=BASE_OMEGA, annealing=True, total_epochs=TOTAL_EPOCHS
    )

    print("\n" + "=" * 70)
    print("      LAUNCHING COGNITIVE EXPERIMENTAL PIPELINE FINE-TUNING")
    print("=" * 70)

    for epoch in range(TOTAL_EPOCHS):
        criterion.update_omega(epoch)
        print(
            f"\n[EPOCH {epoch + 1}/{TOTAL_EPOCHS}] Active Asymmetric Multiplier (Omega): {criterion.current_omega:.4f}"
        )

        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * input_ids.size(0)

        average_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f" -> Realized Mean Training Batch Loss: {average_epoch_loss:.4f}")

        # Metrics verification step loop
        metrics = run_evaluation_cycle(
            model, test_loader, DEVICE, decision_threshold=DECISION_THRESHOLD
        )
        print(
            f" -> Validation Diagnostic Profile -- Recall: {metrics['recall']:.2%}, "
            f"Precision: {metrics['precision']:.2%}, PR-AUC: {metrics['pr_auc']:.4f}"
        )
        print(
            f" -> Confusion Vectors: [TPs: {metrics['tps']} | FNs: {metrics['fns']} (Missed) | FPs: {metrics['fps']} (Alarms)]"
        )

    print("\n" + "=" * 70)
    print("                 PIPELINE FINE-TUNING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()