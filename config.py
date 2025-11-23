import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

FACEIT_API_KEY = os.getenv("FACEIT_API_KEY")
if not FACEIT_API_KEY:
    raise ValueError("FACEIT_API_KEY not found in environment variables. Please check your .env file.")

GAME_ID = "cs2"
BASE_URL = "https://open.faceit.com/data/v4"

HEADERS = {
    "Authorization": f"Bearer {FACEIT_API_KEY}",
    "Accept": "application/json"
}

# Data Directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Data File Paths
PRO_DATA_CSV = os.path.join(DATA_DIR, "pro_faceit_data.csv")
PRO_IDS_JSON = os.path.join(DATA_DIR, "pro_faceit_ids.json")
HIGH_LEVEL_IDS_JSON = os.path.join(DATA_DIR, "high_level_faceit_ids.json")
NORMAL_IDS_JSON = os.path.join(DATA_DIR, "normal_faceit_ids.json")
DATASET_CSV = os.path.join(DATA_DIR, "faceit_players_dataset.csv")
CLASS_STATISTICS_JSON = os.path.join(DATA_DIR, "class_statistics.json")

# Processed Data Paths
X_TRAIN = os.path.join(DATA_DIR, "X_train.npy")
X_VAL = os.path.join(DATA_DIR, "X_val.npy")
X_TEST = os.path.join(DATA_DIR, "X_test.npy")
Y_TRAIN = os.path.join(DATA_DIR, "y_train.npy")
Y_VAL = os.path.join(DATA_DIR, "y_val.npy")
Y_TEST = os.path.join(DATA_DIR, "y_test.npy")
FEATURE_NAMES_JSON = os.path.join(DATA_DIR, "feature_names.json")

# Models Directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_ENCODER = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER = os.path.join(MODELS_DIR, "scaler.pkl")
LOGISTIC_REGRESSION_MODEL = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, "random_forest.pkl")
XGBOOST_MODEL = os.path.join(MODELS_DIR, "xgboost.pkl")
NEURAL_NETWORK_MODEL = os.path.join(MODELS_DIR, "neural_network.h5")
BEST_MODEL_TXT = os.path.join(MODELS_DIR, "best_model.txt")

# Results Directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_COMPARISON_CSV = os.path.join(RESULTS_DIR, "model_comparison.csv")

# AI Coach Configuration
AI_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
