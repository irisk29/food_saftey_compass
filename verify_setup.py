"""
Preflight check. Run this first on the training machine:

    python verify_setup.py

Verifies dependencies, data files, the integrity of the gold holdout set, and feature
parity, so failures surface in 30 seconds instead of two hours into a fine-tune.
Exits non-zero if anything blocking is wrong.
"""

import importlib
import os
import re
import sys

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"
problems, warnings_ = [], []


def check(label, fn, blocking=True):
    try:
        detail = fn()
        print(f"{OK} {label}" + (f" - {detail}" if detail else ""))
        return True
    except Exception as e:
        msg = f"{label}: {type(e).__name__}: {e}"
        if blocking:
            problems.append(msg)
            print(f"{FAIL} {label}\n         {type(e).__name__}: {e}")
        else:
            warnings_.append(msg)
            print(f"{WARN} {label}\n         {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 70)
    print("  FOOD SAFETY COMPASS - PREFLIGHT")
    print("=" * 70)

    # ---- 1. Dependencies -------------------------------------------------
    print("\n[1] Dependencies")

    def dep(name, import_name=None):
        def _f():
            m = importlib.import_module(import_name or name)
            return getattr(m, "__version__", "")
        return _f

    for pkg, imp in [("torch", None), ("transformers", None), ("datasets", None),
                     ("sklearn", None), ("xgboost", None), ("pandas", None),
                     ("numpy", None), ("matplotlib", None), ("seaborn", None),
                     ("nltk", None)]:
        check(pkg, dep(pkg, imp))

    # sentencepiece is required by the DeBERTa-v3 slow tokenizer (use_fast=False).
    # Missing it fails at tokenizer load, minutes into a run.
    check("sentencepiece (required for deberta-v3 tokenizer)", dep("sentencepiece"))

    for pkg in ("optuna", "wandb"):
        check(f"{pkg} (only needed for main.py)", dep(pkg), blocking=False)

    def vader():
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        SentimentIntensityAnalyzer()
        return "lexicon present"
    check("nltk vader_lexicon", vader)

    # ---- 2. Device -------------------------------------------------------
    print("\n[2] Compute device")

    def device():
        import torch
        import config.settings as cfg
        if cfg.DEVICE == "cpu":
            raise RuntimeError("no GPU/MPS detected - fine-tuning will take many hours")
        return f"{cfg.DEVICE} (torch {torch.__version__})"
    check("accelerator", device, blocking=False)

    # ---- 3. Data ---------------------------------------------------------
    print("\n[3] Data files")
    import pandas as pd
    import config.settings as cfg

    def enriched():
        df = pd.read_csv(cfg.INPUT_DATA_PATH)
        assert len(df) > 1000, f"only {len(df)} rows"
        return f"{len(df)} rows, {int(df[cfg.TARGET_COLUMN].sum())} hazards"
    check("enriched training set", enriched)

    def gold_inside():
        df = pd.read_csv(cfg.GOLD_INSIDE_PATH)
        return f"{len(df)} rows (label-quality analysis only - 89% is training data)"
    check("in-sample gold set", gold_inside)

    def gold_holdout():
        if not os.path.exists(cfg.GOLD_HOLDOUT_PATH):
            raise FileNotFoundError(
                "not built. Run:  python labeling/create_gold_dataset.py "
                "--source labeling/holdout_candidate_pool.csv --n 800")
        df = pd.read_csv(cfg.GOLD_HOLDOUT_PATH)
        if len(df) < 300:
            raise RuntimeError(f"only {len(df)} rows - resume the labelling run before evaluating")
        return f"{len(df)} rows, hazard base rate {df['llm_is_hazard'].mean():.1%}"
    check("gold holdout set", gold_holdout)

    # ---- 4. Contamination ------------------------------------------------
    print("\n[4] Holdout integrity (the whole point of the gold set)")

    def overlap():
        if not os.path.exists(cfg.GOLD_HOLDOUT_PATH):
            raise FileNotFoundError("gold holdout not built yet")

        def norm(t):
            return re.sub(r"\s+", " ", str(t).replace("\n", " ").replace("\t", " ")).strip()

        train = pd.read_csv(cfg.INPUT_DATA_PATH, usecols=[cfg.TEXT_COLUMN])
        seen = set(train[cfg.TEXT_COLUMN].fillna("").map(norm))
        gold = pd.read_csv(cfg.GOLD_HOLDOUT_PATH)
        n_overlap = int(gold[cfg.TEXT_COLUMN].map(norm).isin(seen).sum())
        if n_overlap:
            raise RuntimeError(f"{n_overlap} gold rows also appear in training data")
        return f"0 of {len(gold)} gold rows appear in training data"
    check("zero train/gold text overlap", overlap)

    # ---- 5. Feature parity ----------------------------------------------
    print("\n[5] Feature engineering parity")

    def parity():
        from src.features import verify_against_training_data
        rep = verify_against_training_data(cfg.INPUT_DATA_PATH, sample=500)
        bad = [k for k, v in rep.items() if not v]
        if bad:
            raise RuntimeError(f"recomputed features differ from training CSV: {bad}")
        return f"all {len(rep)} derived columns reproduce exactly"
    check("src/features.py matches the notebook", parity)

    # ---- 6. Config -------------------------------------------------------
    print("\n[6] Configuration")

    def selection_metrics():
        import config.settings as c
        if c.CHECKPOINT_METRIC == "recall" or c.HPO_METRIC == "recall":
            raise RuntimeError("selection metric is bare recall - an all-positive model wins")
        return f"checkpoint={c.CHECKPOINT_METRIC}, hpo={c.HPO_METRIC}, loss={c.LOSS_VARIANT}"
    check("selection metrics are not gameable", selection_metrics)

    # ---- 7. Unit tests ---------------------------------------------------
    print("\n[7] Unit tests")

    def tests():
        import subprocess
        r = subprocess.run([sys.executable, "tests/test_losses.py"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout.strip().splitlines()[-1] if r.stdout else "failed")
        return r.stdout.strip().splitlines()[-1]
    check("loss + metric tests", tests)

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    if problems:
        print(f"  {len(problems)} BLOCKING PROBLEM(S) - fix before running:")
        for p in problems:
            print(f"    - {p}")
    if warnings_:
        print(f"  {len(warnings_)} warning(s) (non-blocking):")
        for w in warnings_:
            print(f"    - {w}")
    if not problems:
        print("  READY. Next:  python analysis.py")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
