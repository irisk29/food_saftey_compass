import os
import pandas as pd
from sklearn.model_selection import train_test_split
import config.settings as cfg


def load_and_split_data():
    """
    Loads the enriched dataset and creates reproducible train/test splits
    for both tabular and textual structures.
    """
    resolved_path = os.path.abspath(cfg.INPUT_DATA_PATH)
    print(f"Resolved absolute path: {resolved_path}")
    print(f"Current working directory: {os.getcwd()}")

    if not os.path.exists(cfg.INPUT_DATA_PATH):
        raise FileNotFoundError(f"Missing required dataset file: {cfg.INPUT_DATA_PATH}")

    df = pd.read_csv(cfg.INPUT_DATA_PATH)

    # Fill any structural nan values safely
    df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df[cfg.TARGET_COLUMN]
    )

    print(f"Data split successfully. Train Size: {len(train_df)} | Test Size: {len(test_df)}")
    return train_df, test_df