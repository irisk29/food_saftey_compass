import inspect

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

import config.settings as cfg
from src.losses import AsymmetricSafetyLoss


class CustomSafetyTrainer(Trainer):
    """Replaces the default cross-entropy objective with our asymmetric safety loss."""

    def __init__(self, *args, asymmetric_weight=None, loss_variant=None,
                 gamma=None, tau=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = AsymmetricSafetyLoss(
            weight_fn=cfg.ASYMMETRIC_WEIGHT if asymmetric_weight is None else asymmetric_weight,
            variant=cfg.LOSS_VARIANT if loss_variant is None else loss_variant,
            gamma=cfg.FOCAL_GAMMA if gamma is None else gamma,
            tau=cfg.FN_GATE_TAU if tau is None else tau,
        )
        print(f"    loss: AsymmetricSafetyLoss({self.loss_fn.extra_repr()})")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop the labels so the model never runs its own internal loss branch.
        # The previous code passed labels through and relied on an invalid
        # problem_type ("binary_classification") to make HuggingFace silently skip
        # that branch — which worked by accident and only on transformers 4.40.
        # `**kwargs` absorbs `num_items_in_batch`, added to this signature in 4.46.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Metrics at the deployed threshold, plus threshold-free PR-AUC.

    `pred_positive_rate` is reported deliberately: an all-positive model scores
    perfect recall, so the rate is the cheapest way to make that degenerate solution
    visible instead of flattering. It is logged every eval rather than only when
    trouble is suspected — the deployed focal_asymmetric configuration does not
    collapse (test-split flag rate 0.207), and the collapse seen historically under
    the label-keyed pos_weight loss was never persisted to an artifact, so this column
    is the evidence rather than the recollection.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-np.asarray(logits).flatten()))
    labels = np.asarray(labels).flatten().astype(int)

    preds = (probs >= cfg.DECISION_THRESHOLD).astype(int)
    preds_50 = (probs >= 0.50).astype(int)

    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        # F2 weights recall 2x precision — matches the business asymmetry, but unlike
        # bare recall it still punishes a model that flags everything.
        "f2": fbeta_score(labels, preds, beta=cfg.FBETA, zero_division=0),
        # Threshold-free, so hyperparameter search is not entangled with our 0.20 choice.
        "pr_auc": average_precision_score(labels, probs),
        "pred_positive_rate": float(preds.mean()),
        # Same model at the conventional threshold, for the trade-off table.
        "precision_at_50": precision_score(labels, preds_50, zero_division=0),
        "recall_at_50": recall_score(labels, preds_50, zero_division=0),
        "f2_at_50": fbeta_score(labels, preds_50, beta=cfg.FBETA, zero_division=0),
    }


def _build_training_args(**kwargs):
    """
    TrainingArguments across transformers versions.

    `evaluation_strategy` was renamed `eval_strategy` in 4.41. The project pins 4.40
    but is trained on machines that have drifted ahead, so pick whichever the
    installed version actually accepts rather than crashing on import.
    """
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    strategy = kwargs.pop("_eval_strategy")
    key = "eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"
    kwargs[key] = strategy
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})


def _trainer_tokenizer_kwarg(tokenizer):
    """`tokenizer=` was deprecated in favour of `processing_class=` in transformers 4.46."""
    accepted = set(inspect.signature(Trainer.__init__).parameters)
    return {"processing_class": tokenizer} if "processing_class" in accepted else {"tokenizer": tokenizer}


def tokenize_split(df, tokenizer, max_length=256):
    """Tokenized HF dataset from a dataframe. Shared with the evaluation pipeline."""
    patched = df.copy()
    # Labels forced to float32: the head emits a single logit, and int64 labels
    # trigger a 'square_i64' crash on the MPS backend.
    patched[cfg.TARGET_COLUMN] = patched[cfg.TARGET_COLUMN].astype(np.float32)

    ds = Dataset.from_pandas(
        patched[[cfg.TEXT_COLUMN, cfg.TARGET_COLUMN]].rename(columns={cfg.TARGET_COLUMN: "label"}),
        preserve_index=False,
    )
    return ds.map(
        lambda batch: tokenizer(batch[cfg.TEXT_COLUMN], truncation=True, max_length=max_length),
        batched=True,
    )


def load_tokenizer(model_nm="microsoft/deberta-v3-base"):
    return AutoTokenizer.from_pretrained(model_nm, use_fast=False)


def run_sota_training(train_df, test_df, epochs=3, lr=2e-5, batch_size=8,
                      asymmetric_weight=None, loss_variant=None, gamma=None,
                      return_tokenized=False, eval_df=None):
    """
    Fine-tunes DeBERTa-v3 with the asymmetric safety loss.

    Checkpoint selection uses cfg.CHECKPOINT_METRIC (F2), not recall — recall alone
    is maximised by predicting every review a hazard.

    `eval_df` is the split used for checkpoint selection (and whatever
    `trainer.evaluate()` scores afterwards). Pass the validation split here so the
    test split stays out of every selection decision. If None, falls back to
    `test_df` — which makes the test-split numbers selection-biased ("in-selection")
    and should only be used for quick experiments, never reported results.
    """
    if asymmetric_weight is None:
        asymmetric_weight = cfg.ASYMMETRIC_WEIGHT

    print(f"\n--- DeBERTa-v3 Base on {cfg.DEVICE} "
          f"(w={asymmetric_weight}, variant={loss_variant or cfg.LOSS_VARIANT}) ---")
    model_nm = "microsoft/deberta-v3-base"

    tokenizer = load_tokenizer(model_nm)

    # num_labels=1 -> single logit, read through a sigmoid. No problem_type is set:
    # labels are popped before the forward pass, so the model's internal loss branch
    # is never reached and cannot default to MSE.
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
    model.to(cfg.DEVICE)

    if eval_df is None:
        print("    [warn] no eval_df given — checkpoint selection will use the TEST "
              "split, making its numbers in-selection. Pass the validation split.")
        eval_df = test_df

    train_tokenized = tokenize_split(train_df, tokenizer)
    eval_tokenized = tokenize_split(eval_df, tokenizer)

    # Micro-batching: keep the effective batch at `batch_size` while capping the
    # per-device batch at 4 so activations fit on MPS.
    per_device_batch = 4 if batch_size >= 8 else batch_size
    accum_steps = batch_size // per_device_batch

    training_args = _build_training_args(
        output_dir=cfg.MODEL_OUTPUT_DIR,
        learning_rate=lr,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=accum_steps,
        eval_accumulation_steps=1,
        num_train_epochs=epochs,
        weight_decay=0.01,
        _eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.CHECKPOINT_METRIC,
        greater_is_better=True,
        report_to="wandb",
        fp16=False,  # FP16 on MPS can produce nan gradients on older macOS builds
    )

    trainer = CustomSafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        compute_metrics=compute_metrics,
        asymmetric_weight=asymmetric_weight,
        loss_variant=loss_variant,
        gamma=gamma,
        **_trainer_tokenizer_kwarg(tokenizer),
    )

    trainer.train()

    if return_tokenized:
        # The reporting split is always test_df, tokenized separately from the
        # eval split so downstream evaluation never accidentally scores validation.
        test_tokenized = tokenize_split(test_df, tokenizer) if eval_df is not test_df \
            else eval_tokenized
        return trainer, tokenizer, test_tokenized
    return trainer


def selection_disagreement(trainer):
    """
    Which epoch F2 picks vs which epoch PR-AUC picks.

    Reported rather than silently resolved: if the two metrics choose different
    checkpoints, that is a concrete illustration of why the old recall-only objective
    was unsafe, and it costs nothing to extract from the log history.
    """
    history = [h for h in trainer.state.log_history if "eval_f2" in h]
    if not history:
        return None

    def best(metric):
        row = max(history, key=lambda h: h.get(f"eval_{metric}", float("-inf")))
        return {"epoch": row.get("epoch"), "value": row.get(f"eval_{metric}")}

    f2_best, pr_best = best("f2"), best("pr_auc")
    return {
        "f2_choice_epoch": f2_best["epoch"], "f2_choice_value": f2_best["value"],
        "pr_auc_choice_epoch": pr_best["epoch"], "pr_auc_choice_value": pr_best["value"],
        "metrics_agree": f2_best["epoch"] == pr_best["epoch"],
        "epochs_logged": len(history),
    }
