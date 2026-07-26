import os
from dotenv import load_dotenv

load_dotenv()

# System Paths
# Anchor all paths dynamically to this configuration file's directory
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))  # points to food_saftey_compass/config
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)              # points to food_saftey_compass/

INPUT_DATA_PATH = os.path.join(PROJECT_ROOT, "postprocessing/enriched_allergy_hazard_dataset.csv")

MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_outputs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Gold (LLM-judged) evaluation sets.
#   GOLD_INSIDE_PATH  — sampled 50/50 from the enriched dataset. 1,334 of its 1,500
#                       rows fall in the TRAIN split, so it is NOT a valid evaluation
#                       set on its own; only the ~166 rows overlapping the test split
#                       are usable, and that is too few to report on. Kept because the
#                       heuristic-vs-LLM agreement rate measured on it (85.8%) is a
#                       finding about the *labelling rule*, which needs no holdout.
#   GOLD_HOLDOUT_PATH — sampled from reviews that never entered the pipeline at all
#                       (labeling/build_holdout_pool.py). This is the set to report
#                       model metrics on.
GOLD_INSIDE_PATH = os.path.join(PROJECT_ROOT, "labeling/gold_dataset.csv")
GOLD_HOLDOUT_PATH = os.path.join(PROJECT_ROOT, "labeling/gold_dataset_holdout.csv")
HOLDOUT_POOL_PATH = os.path.join(PROJECT_ROOT, "labeling/holdout_candidate_pool.csv")

# Feature Categorization
# NOTE: "stars" is excluded — it is one of the two conditions used to construct the
# is_hazard label itself (keyword match AND stars <= 3), so including it as a model
# input would be direct label leakage. "medical_lexicon_density" and
# "negation_window_flag" are excluded for the same reason: both are computed from
# overlap with the same keyword list used to build the label, so they leak the
# labeling rule rather than an independent signal. "vader_neg_intensity" is kept as
# a general sentiment signal not tied to the specific labeling keywords.
TABULAR_FEATURES = [
    "useful", "funny", "cool",
    "word_count", "char_count", "exclamation_count",
    "vader_neg_intensity"
]
TEXT_COLUMN = "text"
TARGET_COLUMN = "is_hazard"

# Operational Parameters
ASYMMETRIC_WEIGHT = 50.0
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Decision threshold. Lowered from 0.50 because a missed hazard costs ~100x a false
# alarm (see COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE in analysis/evaluation_pipeline.py),
# so we accept more false alarms to buy recall. Reported alongside th=0.50 everywhere
# so the trade-off is visible rather than baked in silently.
DECISION_THRESHOLD = 0.20

# Loss configuration — see src/losses.py for the derivation of each variant.
# 'pos_weight' is the original label-keyed formulation (equivalent to
# BCEWithLogitsLoss(pos_weight=w)); the other two make the penalty depend on the
# model's actual error, which is what "asymmetric safety loss" was always meant to say.
LOSS_VARIANT = "focal_asymmetric"   # 'pos_weight' | 'focal_asymmetric' | 'fn_gated'
FOCAL_GAMMA = 2.0                   # focal_asymmetric only; gamma=0 reduces to pos_weight
FN_GATE_TAU = DECISION_THRESHOLD    # fn_gated only; gate at the deployed threshold

# Model selection metrics. These deliberately differ:
#   - Checkpoint selection (within a run) uses F2, evaluated at DECISION_THRESHOLD,
#     because picking an epoch should reflect the operating point we actually deploy.
#   - Hyperparameter search (across runs) uses PR-AUC, which is threshold-free, so the
#     search is not entangled with our choice of 0.20.
# Both are computed and logged every eval, and main.py records which checkpoint each
# would have chosen so the disagreement can be reported.
# Neither can be gamed by an all-positive model, unlike the plain `recall` this
# replaced — that degenerate solution was actually observed (100% recall / 37.5%
# precision at weight=50; see grid_search_analysis.py).
CHECKPOINT_METRIC = "f2"        # HuggingFace metric_for_best_model (without 'eval_' prefix)
HPO_METRIC = "pr_auc"           # Optuna objective
FBETA = 2.0                     # beta for the F-beta metric; 2.0 weights recall 2x precision

# Deep Learning Hardware Gate
import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"  # Metal Performance Shaders on Apple Silicon
elif torch.cuda.is_available():
    DEVICE = "cuda"  # NVIDIA GPU
else:
    DEVICE = "cpu"  # Default to CPU

print(f"Using device: {DEVICE}")
