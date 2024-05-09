import pandas as pd
import logging

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")


def fetch_data():
    try:
        data = pd.read_csv(DATA_URL)
        return data
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise RuntimeError("Data fetching failed") from e
