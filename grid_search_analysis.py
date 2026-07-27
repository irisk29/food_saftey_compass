"""
Variant experiment: does an error-dependent safety penalty beat plain class weighting?

Two axes are swept together:

  variant — how the penalty is applied
      pos_weight        every hazard example weighted w      (label-dependent)
      focal_asymmetric  weight scales with (1-p)^gamma       (error-dependent, smooth)
      fn_gated          weight applies only where p < tau    (error-dependent, hard)

  weight — how large the penalty is
      w=1 is the unweighted control, and it is the reference the custom loss has to
      beat before it can be called a contribution.

Every configuration is scored twice:

  1. the validation split against the heuristic keyword label — kept for continuity
     with the runs already collected, but saturated near ceiling, because a
     transformer reproducing a keyword rule is an easy problem. Differences here are
     within single-seed noise and cannot rank the variants.
  2. the fresh gold holdout against the independent LLM judgement (`gold_*` columns) —
     out-of-sample, far from ceiling, and therefore the only place where the loss
     formulation can actually be shown to matter. This is the criterion the summary
     sorts on; the validation metric is only the fallback when no gold set is present.

The deployed configuration (cfg.LOSS_VARIANT @ cfg.ASYMMETRIC_WEIGHT) is force-included
in every run unless --no-force-deployed is passed. Every headline number in the project
comes from that configuration, so a sweep that omits it compares everything except the
thing being reported.

On collapse: bare recall was abandoned as a selection metric because an all-positive
model maximises it by definition, and that degenerate solution was observed historically
under the label-keyed `pos_weight` formulation. That observation was never persisted to
an artifact, so it is not quoted here as a measurement. What the committed evidence does
show is the opposite for the current loss: the deployed `focal_asymmetric` model at
w=50 flags 20.7% of the test split, not everything. `pred_positive_rate` (validation) and
`gold_flag_rate` are recorded per run so collapse, if it appears, is visible in the table
rather than inferred.

Runtime note: the full grid is 3 variants x 7 weights = 21 fine-tunes. Use --variants
or --weights to cut it down; --quick runs a 2x4 subset that still contains the
comparison that matters. Completed (variant, weight) pairs are read back from the output
CSV and skipped, so an interrupted sweep resumes instead of re-paying for finished runs;
--force re-runs everything.
"""

import argparse
import gc
import os

import pandas as pd
import torch

import config.settings as cfg
from src.data_pipeline import load_and_split_data, load_gold_holdout
from src.sota_model import load_tokenizer, run_sota_training, tokenize_split
from analysis.evaluation_pipeline import score_variant, sota_probabilities

WEIGHT_CANDIDATES = [1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0]
VARIANTS = ["pos_weight", "focal_asymmetric", "fn_gated"]

QUICK_WEIGHTS = [1.0, 5.0, 15.0, 50.0]
QUICK_VARIANTS = ["pos_weight", "focal_asymmetric"]

# Held fixed so the loss configuration is the only thing varying.
EPOCHS = 3
LR = 1.8140198244240376e-05
BATCH_SIZE = 16

OUTPUT_CSV = os.path.join(cfg.RESULTS_DIR, "grid_search_loss_variants.csv")

# score_variant() returns display-friendly keys; flatten them into short prefixed
# column names so the gold block is obviously distinct from the validation block and
# the two can never be silently confused in a table.
GOLD_COLUMN_MAP = {
    "PR-AUC": "gold_pr_auc",
    "Recall (Safety Coverage)": "gold_recall",
    "Precision (Alert Validity)": "gold_precision",
    "F2 (recall-weighted)": "gold_f2",
    "Flag Rate": "gold_flag_rate",
    "False Negatives (Missed)": "gold_fn",
    "False Positives (Alarms)": "gold_fp",
    "Total Risk Cost": "gold_cost",
}

# A configuration that flags essentially everything has collapsed. Applied to both
# ground truths on the same rule so the two columns mean the same thing.
COLLAPSE_FLAG_RATE = 0.95

# Below this spread the validation metric cannot separate the configurations, and any
# ranking built on it is noise. Chosen an order of magnitude above nothing and well
# below a difference anyone would defend in a viva.
SATURATION_SPREAD = 0.01


def _free_device_memory():
    gc.collect()
    if cfg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif cfg.DEVICE == "mps":
        torch.mps.empty_cache()


def _pair_key(variant, weight):
    """Hashable identity for a grid cell. Weights arrive as floats from both the CLI
    and the CSV, so round before comparing rather than trusting float equality."""
    return (str(variant), round(float(weight), 6))


def _load_existing(path):
    """Previously completed runs, keyed by (variant, weight). Returns (df, keys)."""
    if not os.path.exists(path):
        return None, set()
    try:
        prior = pd.read_csv(path)
    except Exception as e:  # a truncated CSV should not cost a whole sweep
        print(f"[warn] could not read existing results at {path} ({e}); starting fresh.")
        return None, set()
    if prior.empty or "variant" not in prior.columns or "weight" not in prior.columns:
        return None, set()
    keys = {_pair_key(r.variant, r.weight) for r in prior.itertuples()}
    return prior, keys


def _score_gold(trainer, gold_tokenized, gold_y):
    """Gold-holdout metric block for one trained model, at the deployed threshold."""
    probs = sota_probabilities(trainer, gold_tokenized)
    metrics = score_variant(gold_y, probs, cfg.DECISION_THRESHOLD)
    row = {col: metrics[key] for key, col in GOLD_COLUMN_MAP.items()}
    row["gold_collapsed"] = bool(row["gold_flag_rate"] > COLLAPSE_FLAG_RATE)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="2 variants x 4 weights instead of the full 21-run grid")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", default=OUTPUT_CSV)
    parser.add_argument("--force", action="store_true",
                        help="re-run every cell even if it is already in the output CSV")
    parser.add_argument("--no-force-deployed", dest="force_deployed", action="store_false",
                        help="do not auto-add the deployed configuration to the run set")
    parser.set_defaults(force_deployed=True)
    args = parser.parse_args()

    variants = list(args.variants or (QUICK_VARIANTS if args.quick else VARIANTS))
    weights = list(args.weights or (QUICK_WEIGHTS if args.quick else WEIGHT_CANDIDATES))

    # The cartesian product is the default run set, but the deployed configuration is
    # appended as an explicit extra cell rather than by widening the product — adding
    # w=50 to `weights` would silently multiply the run count across every variant.
    planned = [(v, w) for v in variants for w in weights]
    deployed = (cfg.LOSS_VARIANT, float(cfg.ASYMMETRIC_WEIGHT))
    if args.force_deployed and _pair_key(*deployed) not in {_pair_key(*p) for p in planned}:
        planned.append(deployed)
        print(f"[force-deployed] added the reported configuration "
              f"{deployed[0]} @ w={deployed[1]} to the run set — every headline number "
              f"in the project comes from it, so the sweep must contain it. "
              f"Pass --no-force-deployed to suppress.")

    prior_df, done_keys = _load_existing(args.out)
    if args.force:
        if done_keys:
            print(f"[force] ignoring {len(done_keys)} completed run(s) in {args.out}; "
                  f"they will be overwritten.")
        prior_df, done_keys = None, set()

    todo = [p for p in planned if _pair_key(*p) not in done_keys]
    skipped = [p for p in planned if _pair_key(*p) in done_keys]

    print("========================================================")
    print("   GRID SEARCH: ASYMMETRIC SAFETY LOSS FORMULATIONS     ")
    print("========================================================")
    print(f"  variants: {variants}")
    print(f"  weights:  {weights}")
    print(f"  planned cells: {len(planned)}")
    if skipped:
        print(f"  resuming — {len(skipped)} already in {os.path.basename(args.out)}: "
              + ", ".join(f"{v}@{w:g}" for v, w in skipped))
    print(f"  to run now: {len(todo)}"
          + ("" if not todo else " — " + ", ".join(f"{v}@{w:g}" for v, w in todo)))
    print()

    # The grid compares loss configurations — a selection activity — so it scores
    # the validation split. The test split stays reserved for the final report.
    train_df, val_df, test_df = load_and_split_data(with_validation=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # The heuristic-label metrics above sit near ceiling, so the gold holdout is what
    # actually discriminates between variants. Tokenized once: the checkpoint is a
    # different fine-tune each iteration but the tokenizer is the same base model
    # every time, and re-tokenizing per cell would be pure waste.
    gold_df = load_gold_holdout(require=False)
    gold_tokenized = gold_y = None
    if gold_df is None:
        print("[warn] no gold holdout set found — the sweep will produce heuristic-label "
              "metrics only, which are saturated and cannot rank the variants. Build it "
              "with labeling/build_holdout_pool.py + labeling/create_gold_dataset.py and "
              "re-run to fill the gold_* columns.")
    elif todo:
        gold_tokenized = tokenize_split(gold_df, load_tokenizer())
        gold_y = gold_df[cfg.TARGET_COLUMN].values

    results = []
    for variant, w in todo:
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
        row["collapsed"] = bool(row["pred_positive_rate"] and row["pred_positive_rate"] > COLLAPSE_FLAG_RATE)

        print(f"    val  -> pr_auc={row['pr_auc']:.4f} f2={row['f2']:.4f} "
              f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
              f"flag_rate={row['pred_positive_rate']:.3f}"
              f"{'  [COLLAPSED]' if row['collapsed'] else ''}")

        if gold_tokenized is not None:
            row.update(_score_gold(trainer, gold_tokenized, gold_y))
            print(f"    gold -> pr_auc={row['gold_pr_auc']:.4f} f2={row['gold_f2']:.4f} "
                  f"precision={row['gold_precision']:.3f} recall={row['gold_recall']:.3f} "
                  f"flag_rate={row['gold_flag_rate']:.3f}"
                  f"{'  [COLLAPSED]' if row['gold_collapsed'] else ''}")

        results.append(row)

        # Write incrementally: a 21-run sweep is long enough that losing it to a
        # crash on the last run would be painful. Prior rows are merged back in on
        # every write so a resumed sweep never truncates the file it resumed from.
        _merge(prior_df, results).to_csv(args.out, index=False)

        del trainer
        _free_device_memory()

    df = _merge(prior_df, results)
    if df.empty:
        print("\nNothing to report: no runs completed and no prior results found.")
        return

    # Sort on the gold holdout when it is available: that is the metric with room to
    # separate the configurations. Validation PR-AUC is only the fallback.
    has_gold = "gold_pr_auc" in df.columns and df["gold_pr_auc"].notna().any()
    sort_col = "gold_pr_auc" if has_gold else "pr_auc"
    df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print("\n" + "-" * 78)
    print(f"           GRID SEARCH RESULTS (sorted by {sort_col})")
    print("-" * 78)
    print(df.to_string(index=False))

    print(f"\n--- Best per variant (by {sort_col}) ---")
    ranked = df[df[sort_col].notna()]
    if not ranked.empty:
        print(ranked.loc[ranked.groupby("variant")[sort_col].idxmax()].to_string(index=False))

    # Rows carried over from before gold scoring existed have NaN there. Say so out
    # loud — a blank cell in a results table is too easily read as a zero or as a
    # comparison that was made and lost.
    if has_gold:
        missing = df[df["gold_pr_auc"].isna()]
        if not missing.empty:
            print("\n[warn] no gold-holdout score for the following configurations — the "
                  "blank gold_* cells mean NOT MEASURED, not zero. Re-run them with "
                  "--force to fill in:")
            print("  " + ", ".join(f"{r.variant}@{r.weight:g}" for r in missing.itertuples()))

    # The saturation caveat is printed unconditionally when it applies, because the
    # obvious misreading of this table is to declare a winner on a 0.00x difference.
    val_spread = float(df["pr_auc"].max() - df["pr_auc"].min()) if df["pr_auc"].notna().any() else 0.0
    if len(df) > 1 and val_spread < SATURATION_SPREAD:
        print(f"\n[caveat] validation PR-AUC spread across configurations is only "
              f"{val_spread:.4f}. The heuristic keyword label is saturated at this "
              f"level — every configuration reproduces the rule almost perfectly, so "
              f"this metric cannot discriminate between loss variants and any ranking "
              f"on it is within single-seed noise. That is exactly what the gold_* "
              f"columns exist to resolve"
              + ("." if has_gold else ", and they are empty for this run."))

    if "collapsed" in df.columns:
        collapsed = df[df["collapsed"].fillna(False).astype(bool)]
        if not collapsed.empty:
            print(f"\n{len(collapsed)}/{len(df)} configurations collapsed to flagging >95% of "
                  f"reviews on the validation split:")
            print(collapsed[["variant", "weight", "recall", "precision"]].to_string(index=False))
            print("These score near-perfect recall while being useless — which is precisely")
            print("why bare recall was abandoned as the selection metric.")

    if "gold_collapsed" in df.columns:
        gold_collapsed = df[df["gold_collapsed"].fillna(False).astype(bool)]
        if not gold_collapsed.empty:
            print(f"\n{len(gold_collapsed)} configuration(s) collapsed on the gold holdout:")
            print(gold_collapsed[["variant", "weight", "gold_recall",
                                  "gold_precision"]].to_string(index=False))

    print(f"\nFull results: {args.out}")


def _merge(prior_df, results):
    """
    Prior rows plus this session's rows.

    Concatenated rather than joined: the prior file predates gold scoring, so its rows
    simply have no gold_* columns and pandas fills them with NaN. Existing validation
    columns are left untouched and unrenamed so old and new rows stay directly
    comparable. Any (variant, weight) re-run in this session wins over its prior row.
    """
    new_df = pd.DataFrame(results)
    if prior_df is None or prior_df.empty:
        return new_df
    if new_df.empty:
        return prior_df.copy()

    fresh = {_pair_key(r.variant, r.weight) for r in new_df.itertuples()}
    keep = prior_df[[_pair_key(r.variant, r.weight) not in fresh for r in prior_df.itertuples()]]
    return pd.concat([keep, new_df], ignore_index=True)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
