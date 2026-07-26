import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import config.settings as cfg


def build_baseline_pipeline(scale_pos_weight=1.0):
    """
    Constructs a multi-modal Scikit-Learn Pipeline combining
    TF-IDF text features and normalized tabular metadata.

    `scale_pos_weight` matters for fairness of the headline comparison: the DeBERTa
    model trains with a large penalty on the hazard class, so an unweighted baseline
    is handicapped by construction at any shared threshold. Passing the empirical
    negative/positive ratio puts both models on comparable footing.
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

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=cfg.RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight
        ))
    ])

    return pipeline


def train_and_evaluate_baseline(train_df, test_df, balance_classes=True):
    """Trains and executes complete valuation diagnostics for the baseline model."""
    print("\n--- Training Traditional XGBoost Baseline Pipeline ---")

    y_train = train_df[cfg.TARGET_COLUMN]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = (n_neg / n_pos) if (balance_classes and n_pos) else 1.0
    print(f"    class balance: {n_neg} benign / {n_pos} hazard -> scale_pos_weight={spw:.2f}")

    pipeline = build_baseline_pipeline(scale_pos_weight=spw)

    X_train = train_df[cfg.TABULAR_FEATURES + [cfg.TEXT_COLUMN]]
    X_test = test_df[cfg.TABULAR_FEATURES + [cfg.TEXT_COLUMN]]
    y_test = test_df[cfg.TARGET_COLUMN]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("\n[Baseline Evaluation Report]")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return pipeline