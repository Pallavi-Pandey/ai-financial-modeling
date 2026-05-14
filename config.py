import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
]
GEMINI_MODEL = "gemini-2.0-flash"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 6,
        "random_state": 42,
    },
}

DATA_PATH = "data/sample_data.csv"
TEST_SIZE = 0.2
FORECAST_PERIODS = 6
