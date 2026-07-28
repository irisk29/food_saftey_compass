"""
Re-run the failure-mode bucketing over the *committed* error-detail CSVs, with no model.

Why this exists
---------------
`analysis/error_analysis.py` is normally driven from `analysis.py`, which fine-tunes
DeBERTa-v3 first and only then calls `analyze_errors`. That is the right entrypoint when
the predictions change, but it is the wrong one when only the *taxonomy* changes: editing
a regex should not cost a GPU run. The predictions we need are already persisted —
`results/error_analysis_*_detail.csv` records, for every error, which evaluated row it was
(`index`), whether it was an FP or an FN, and the model probability.

So this script reconstructs the exact evaluation state from files:

* the evaluated frame is re-loaded from its source (so the taxonomy sees the *full* review
  text, not the 400-character excerpt stored in the detail CSV);
* `y_true` is the ground-truth column that run was scored against;
* `y_pred` is `y_true` with exactly the recorded FP/FN indices flipped, which reproduces
  the original confusion matrix by construction;
* `prob` is copied back from the detail CSV.

It then calls the ordinary `analyze_errors`, so the regenerated detail/summary/markdown
artifacts are byte-comparable with the originals and no output path or format is special-cased.

Guard rail: with the taxonomy unchanged this script reproduces the committed
`primary_mode` / `all_modes` columns exactly on all three runs (verified before the
NEGATED_HAZARD widening). `--check` re-asserts that reproduction instead of writing, and
`--dry-run` prints the before/after bucket distribution without touching any file.

Usage
-----
    python -m analysis.rebucket_errors --dry-run   # show what would change
    python -m analysis.rebucket_errors             # rewrite detail/summary/markdown
"""

import argparse
import os
import sys

import pandas as pd

import config.settings as cfg
from analysis.error_analysis import analyze_errors


def _load_gold_holdout():
    """Fresh LLM-labelled holdout, enriched exactly as `src.data_pipeline` enriches it."""
    from src.features import enrich
    df = pd.read_csv(cfg.GOLD_HOLDOUT_PATH)
    df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")
    return enrich(df, text_column=cfg.TEXT_COLUMN), "llm_is_hazard"


def _load_test_split():
    """The 20% test split of the enriched dataset, scored against the heuristic label."""
    from src.data_pipeline import load_and_split_data
    _, _, test_df = load_and_split_data(with_validation=True)
    return test_df, cfg.TARGET_COLUMN


def _load_gold_inside():
    """The in-sample gold set: heuristic label judged against the LLM label."""
    return pd.read_csv(cfg.GOLD_INSIDE_PATH), "llm_is_hazard"


# label -> (loader, ground-truth column). The label is the artifact stem, i.e.
# results/error_analysis_{label}_{detail,summary}.csv and results/error_analysis_{label}.md.
RUNS = {
    "deberta_gold_llm_label_fresh_holdout": _load_gold_holdout,
    "deberta_heuristic_label_test_split": _load_test_split,
    "heuristic_label": _load_gold_inside,
}


def _detail_path(label, output_dir):
    return os.path.join(output_dir, f"error_analysis_{label}_detail.csv")


def reconstruct(label, output_dir):
    """
    Rebuild (df, y_true, y_pred, probs) for one run from its committed detail CSV.

    Raises if the recorded error indices are not present in the reloaded frame, which
    would mean the source data has drifted since the detail CSV was written.
    """
    detail = pd.read_csv(_detail_path(label, output_dir))
    df, truth_col = RUNS[label]()

    missing = set(detail["index"]) - set(df.index)
    if missing:
        raise ValueError(
            f"[{label}] {len(missing)} error indices are absent from the reloaded source "
            f"frame (e.g. {sorted(missing)[:5]}). The source data has changed; re-run the "
            f"full evaluation instead of re-bucketing."
        )

    # Sanity-check the mapping: the detail CSV stores the first 400 characters of the text.
    for _, r in detail.iterrows():
        if str(df.loc[r["index"], cfg.TEXT_COLUMN])[:200] != str(r["text"])[:200]:
            raise ValueError(f"[{label}] text mismatch at index {r['index']}; refusing to re-bucket.")

    y_true = df[truth_col].astype(int)
    y_pred = y_true.copy()
    y_pred.loc[detail.loc[detail.error_type == "FP", "index"]] = 1   # predicted 1, truth 0
    y_pred.loc[detail.loc[detail.error_type == "FN", "index"]] = 0   # predicted 0, truth 1

    probs = pd.Series(float("nan"), index=df.index)
    probs.loc[detail["index"]] = detail["prob"].values

    n_fp = int(((y_true == 0) & (y_pred == 1)).sum())
    n_fn = int(((y_true == 1) & (y_pred == 0)).sum())
    expected = detail.error_type.value_counts().to_dict()
    if n_fp != expected.get("FP", 0) or n_fn != expected.get("FN", 0):
        raise ValueError(
            f"[{label}] reconstructed {n_fp} FP / {n_fn} FN but the detail CSV records "
            f"{expected.get('FP', 0)} / {expected.get('FN', 0)}."
        )

    return detail, df, y_true, y_pred, probs


def _distribution(detail):
    """error_type x primary_mode counts, as a plain dict keyed by (type, mode)."""
    return detail.groupby(["error_type", "primary_mode"]).size().to_dict()


def _print_delta(label, before, after):
    keys = sorted(set(before) | set(after))
    width = max(len(f"{k[0]} {k[1]}") for k in keys)
    print(f"\n=== {label} ===")
    print(f"{'bucket'.ljust(width)}  before   after   delta")
    for k in keys:
        b, a = before.get(k, 0), after.get(k, 0)
        flag = "" if b == a else "   <--"
        print(f"{(k[0] + ' ' + k[1]).ljust(width)}  {b:6d}  {a:6d}  {a - b:+6d}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=cfg.RESULTS_DIR)
    ap.add_argument("--labels", nargs="*", default=list(RUNS),
                    help="subset of runs to re-bucket (default: all three)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the before/after distribution without writing artifacts")
    ap.add_argument("--check", action="store_true",
                    help="assert the current taxonomy reproduces the committed buckets exactly")
    args = ap.parse_args(argv)

    failures = []
    for label in args.labels:
        if label not in RUNS:
            raise SystemExit(f"unknown label {label!r}; known: {list(RUNS)}")

        detail, df, y_true, y_pred, probs = reconstruct(label, args.output_dir)
        before = _distribution(detail)

        out_dir = args.output_dir
        if args.dry_run or args.check:
            # Bucket into a throwaway directory so the committed artifacts are untouched.
            import tempfile
            out_dir = tempfile.mkdtemp(prefix=f"rebucket_{label}_")

        _, per_error = analyze_errors(
            df, y_true=y_true, y_pred=y_pred, probs=probs,
            label=label, output_dir=out_dir,
        )
        after = _distribution(per_error)
        _print_delta(label, before, after)

        if args.check:
            merged = detail.merge(per_error, on=["error_type", "index"], suffixes=("_old", "_new"))
            diff = merged[merged.all_modes_old != merged.all_modes_new]
            print(f"[{label}] {len(merged) - len(diff)}/{len(merged)} rows reproduce exactly")
            if len(diff):
                failures.append(label)

    if args.check and failures:
        print(f"\nFAILED reproduction: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
