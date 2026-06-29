import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import config.settings as cfg


def build_baseline_pipeline():
    """
    Constructs a multi-modal Scikit-Learn Pipeline combining
    TF-IDF text features and normalized tabular metadata.
    """
    # Text Processor configuration (enforces internal lowercasing for the baseline)
    text_processor = TfidfVectorizer(max_features=2500, stop_words='english', lowercase=True)

    # Numerical Preprocessor configuration
    numerical_processor = StandardScaler()

    # Consolidate column transformation mapping
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', text_processor, cfg.TEXT_COLUMN),
            ('num_scale', numerical_processor, cfg.TABULAR_FEATURES)
        ]
    )

    # Compute class weights to counter dataset imbalance
    # scale_pos_weight = total_negative_examples / total_positive_examples
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=cfg.RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05
        ))
    ])

    return pipeline


def train_and_evaluate_baseline(train_df, test_df):
    """Trains and executes complete valuation diagnostics for the baseline model."""
    print("\n--- Training Traditional XGBoost Baseline Pipeline ---")
    pipeline = build_baseline_pipeline()

    X_train = train_df[cfg.TABULAR_FEATURES + [cfg.TEXT_COLUMN]]
    y_train = train_df[cfg.TARGET_COLUMN]
    X_test = test_df[cfg.TABULAR_FEATURES + [cfg.TEXT_COLUMN]]
    y_test = test_df[cfg.TARGET_COLUMN]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("\n[Baseline Evaluation Report]")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return pipeline