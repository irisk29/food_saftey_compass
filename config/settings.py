import os
from dotenv import load_dotenv

load_dotenv()

# System Paths
# Anchor all paths dynamically to this configuration file's directory
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))  # points to food_saftey_compass/config
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)              # points to food_saftey_compass/

INPUT_DATA_PATH = os.path.join(PROJECT_ROOT, "postprocessing/enriched_allergy_hazard_dataset.csv")

MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_outputs")

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

# Deep Learning Hardware Gate
import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"  # Metal Performance Shaders on Apple Silicon
elif torch.cuda.is_available():
    DEVICE = "cuda"  # NVIDIA GPU
else:
    DEVICE = "cpu"  # Default to CPU

print(f"Using device: {DEVICE}")
