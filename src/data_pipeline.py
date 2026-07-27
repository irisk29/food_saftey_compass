import os

import pandas as pd
from sklearn.model_selection import train_test_split

import config.settings as cfg
from src.features import enrich


def load_and_split_data(with_validation=False):
    """
    Loads the enriched dataset and creates reproducible splits.

    Default: (train, test) — unchanged from the original two-way split.

    `with_validation=True`: (train, val, test). The validation split is carved out
    of the train split *after* the test split, so the test split is byte-identical
    in both modes (and identical to the historical one — everything documented about
    which gold rows fall in the test split still holds). Validation is what
    checkpoint selection and hyperparameter search are allowed to see; the test
    split is reserved for reporting only.
    """
    resolved_path = os.path.abspath(cfg.INPUT_DATA_PATH)
    print(f"Resolved absolute path: {resolved_path}")

    if not os.path.exists(cfg.INPUT_DATA_PATH):
        raise FileNotFoundError(f"Missing required dataset file: {cfg.INPUT_DATA_PATH}")

    df = pd.read_csv(cfg.INPUT_DATA_PATH)
    df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df[cfg.TARGET_COLUMN]
    )

    if not with_validation:
        print(f"Data split successfully. Train Size: {len(train_df)} | Test Size: {len(test_df)}")
        return train_df, test_df

    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.VAL_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=train_df[cfg.TARGET_COLUMN]
    )

    print(f"Data split successfully. Train Size: {len(train_df)} | "
          f"Val Size: {len(val_df)} | Test Size: {len(test_df)}")
    return train_df, val_df, test_df


def load_gold_holdout(path=None, require=True):
    """
    The out-of-sample evaluation set: reviews that never entered the pipeline,
    labelled independently by an LLM judge.

    The heuristic `is_hazard` column does not exist for these rows — they were never
    scored by the keyword+stars rule — so `llm_is_hazard` becomes the target. That is
    the whole point: metrics computed here measure hazard detection, not fidelity to
    a labelling heuristic.

    Features are recomputed with src.features, which is verified to reproduce the
    training enrichment exactly.
    """
    path = path or cfg.GOLD_HOLDOUT_PATH
    if not os.path.exists(path):
        if require:
            raise FileNotFoundError(
                f"Gold holdout set not found at {path}. Build it with:\n"
                f"  python labeling/build_holdout_pool.py --target 3000\n"
                f"  python labeling/create_gold_dataset.py --source labeling/holdout_candidate_pool.csv --n 800"
            )
        return None

    df = pd.read_csv(path)
    df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")
    df = enrich(df, text_column=cfg.TEXT_COLUMN)

    # The LLM judgement is the ground truth here.
    df[cfg.TARGET_COLUMN] = df["llm_is_hazard"].astype(int)

    base_rate = df[cfg.TARGET_COLUMN].mean()
    print(f"Gold holdout: {len(df)} rows | hazard base rate {base_rate:.1%} "
          f"(the 50/50 in-sample gold set cannot measure this)")
    return df


def load_gold_inside(path=None, restrict_to_test_split=True):
    """
    The in-sample gold set (sampled 50/50 from the enriched dataset).

    89% of its rows sit in the training split, so it is unusable for model metrics as
    a whole. Two things it IS valid for:
      - heuristic-vs-LLM label agreement, which is a property of the labelling rule
        and needs no holdout at all;
      - a small clean model check, if restricted to the test-split rows.

    `restrict_to_test_split=True` returns only the ~166 rows the model never saw.
    """
    path = path or cfg.GOLD_INSIDE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"In-sample gold set not found: {path}")

    gold = pd.read_csv(path)

    if not restrict_to_test_split:
        return gold

    _, test_df = load_and_split_data()
    clean = gold[gold["source_index"].isin(set(test_df.index))].copy()
    print(f"In-sample gold set: {len(gold)} rows, of which {len(clean)} are in the "
          f"test split ({len(gold) - len(clean)} were trained on and are excluded)")
    return clean


def heuristic_vs_llm_agreement(path=None):
    """
    Quantifies how trustworthy the keyword+stars label actually is.

    Valid on the full in-sample gold set despite the train overlap: this compares two
    *labels* to each other and never consults a model, so nothing leaks.
    """
    gold = pd.read_csv(path or cfg.GOLD_INSIDE_PATH)
    h, l = gold[cfg.TARGET_COLUMN].astype(int), gold["llm_is_hazard"].astype(int)

    tp = int(((h == 1) & (l == 1)).sum())
    fp = int(((h == 1) & (l == 0)).sum())   # heuristic over-flags
    fn = int(((h == 0) & (l == 1)).sum())   # heuristic misses
    tn = int(((h == 0) & (l == 0)).sum())

    return {
        "n": len(gold),
        "agreement": float((h == l).mean()),
        "heuristic_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "heuristic_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "heuristic_flagged_hazard_llm_says_benign": fp,
        "heuristic_flagged_benign_llm_says_hazard": fn,
        "both_hazard": tp,
        "both_benign": tn,
    }
