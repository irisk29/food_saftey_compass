"""
Variant experiment: does an error-dependent safety penalty beat plain class weighting?

Two axes are swept together:

  variant — how the penalty is applied
      pos_weight        every hazard example weighted w      (label-dependent)
      focal_asymmetric  weight scales with (1-p)^gamma       (error-dependent, smooth)
      fn_gated          weight applies only where p < tau    (error-dependent, hard)

  weight — how large the penalty is
      w=1 is the unweighted control. w=50 was the original setting and is known to
      collapse to predicting hazard for everything (100% recall, 37.5% precision),
      which is exactly the failure this sweep is meant to characterise rather than
      stumble into.

Selection is on F2 and PR-AUC, never bare recall — the degenerate all-positive model
wins on recall by definition. `pred_positive_rate` is recorded for every run so
collapse is visible in the results table rather than inferred.

Runtime note: the full grid is 3 variants x 7 weights = 21 fine-tunes. Use --variants
or --weights to cut it down; --quick runs a 2x4 subset that still contains the
comparison that matters.
"""

import argparse
import gc
import os

import pandas as pd
import torch

import config.settings as cfg
from src.data_pipeline import load_and_split_data
from src.sota_model import run_sota_training

WEIGHT_CANDIDATES = [1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0]
VARIANTS = ["pos_weight", "focal_asymmetric", "fn_gated"]

QUICK_WEIGHTS = [1.0, 5.0, 15.0, 50.0]
QUICK_VARIANTS = ["pos_weight", "focal_asymmetric"]

# Held fixed so the loss configuration is the only thing varying.
EPOCHS = 3
LR = 1.8140198244240376e-05
BATCH_SIZE = 16

OUTPUT_CSV = os.path.join(cfg.RESULTS_DIR, "grid_search_loss_variants.csv")


def _free_device_memory():
    gc.collect()
    if cfg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif cfg.DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="2 variants x 4 weights instead of the full 21-run grid")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", default=OUTPUT_CSV)
    args = parser.parse_args()

    variants = args.variants or (QUICK_VARIANTS if args.quick else VARIANTS)
    weights = args.weights or (QUICK_WEIGHTS if args.quick else WEIGHT_CANDIDATES)

    print("========================================================")
    print("   GRID SEARCH: ASYMMETRIC SAFETY LOSS FORMULATIONS     ")
    print("========================================================")
    print(f"  variants: {variants}")
    print(f"  weights:  {weights}")
    print(f"  total runs: {len(variants) * len(weights)}\n")

    # The grid compares loss configurations — a selection activity — so it scores
    # the validation split. The test split stays reserved for the final report.
    train_df, val_df, test_df = load_and_split_data(with_validation=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    results = []
    for variant in variants:
        for w in weights:
            print(f"\n>>> variant={variant}  weight={w}")

            trainer = run_sota_training(
                train_df=train_df,
                test_df=test_df,
                eval_df=val_df,
                epochs=args.epochs,
                lr=LR,
                batch_size=BATCH_SIZE,
                asymmetric_weight=w,
                loss_variant=variant,
            )

            m = trainer.evaluate()  # scores the validation split (eval_df)
            row = {
                "variant": variant,
                "weight": w,
                "pr_auc": m.get("eval_pr_auc"),
                "f2": m.get("eval_f2"),
                "f1": m.get("eval_f1"),
                "precision": m.get("eval_precision"),
                "recall": m.get("eval_recall"),
                # The collapse detector: 1.0 means the model flags every review.
                "pred_positive_rate": m.get("eval_pred_positive_rate"),
                "precision_at_50": m.get("eval_precision_at_50"),
                "recall_at_50": m.get("eval_recall_at_50"),
            }
            row["collapsed"] = bool(row["pred_positive_rate"] and row["pred_positive_rate"] > 0.95)
            results.append(row)

            print(f"    -> pr_auc={row['pr_auc']:.4f} f2={row['f2']:.4f} "
                  f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
                  f"flag_rate={row['pred_positive_rate']:.3f}"
                  f"{'  [COLLAPSED]' if row['collapsed'] else ''}")

            # Write incrementally: a 21-run sweep is long enough that losing it to a
            # crash on the last run would be painful.
            pd.DataFrame(results).to_csv(args.out, index=False)

            del trainer
            _free_device_memory()

    df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print("\n" + "-" * 78)
    print("                GRID SEARCH RESULTS (sorted by PR-AUC)")
    print("-" * 78)
    print(df.to_string(index=False))

    print("\n--- Best per variant (by PR-AUC) ---")
    print(df.loc[df.groupby("variant")["pr_auc"].idxmax()].to_string(index=False))

    n_collapsed = int(df["collapsed"].sum())
    if n_collapsed:
        print(f"\n{n_collapsed}/{len(df)} configurations collapsed to flagging >95% of reviews:")
        print(df[df["collapsed"]][["variant", "weight", "recall", "precision"]].to_string(index=False))
        print("These score near-perfect recall while being useless — which is precisely")
        print("why bare recall was abandoned as the selection metric.")

    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
