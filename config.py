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
