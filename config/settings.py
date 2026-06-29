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
TABULAR_FEATURES = [
    "stars", "useful", "funny", "cool",
    "word_count", "char_count", "exclamation_count",
    "medical_lexicon_density", "vader_neg_intensity", "negation_window_flag"
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
