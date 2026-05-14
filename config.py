import os
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) then fall back to env / .env."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

GEMINI_API_KEYS = [
    _get_secret("GEMINI_API_KEY_1"),
    _get_secret("GEMINI_API_KEY_2"),
]
GEMINI_MODEL = "gemini-2.0-flash"

ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

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
