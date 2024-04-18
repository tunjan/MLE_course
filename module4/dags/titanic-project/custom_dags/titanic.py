import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")

default_args = {
    "owner": "airflow",
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="titanic_model_training",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    tags=["example"],
)
def titanic_training_dag():
    @task
    def fetch_data():
        try:
            data = pd.read_csv(DATA_URL)
            return data
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")
            raise RuntimeError("Data fetching failed") from e

    @task
    def preprocess_data(data: pd.DataFrame):
        """Preprocess the data by dropping non-essential columns and filling missing values."""
        required_drops = {"PassengerId", "Name", "Ticket", "Cabin"}

        drop_columns = [col for col in required_drops if col in data.columns]
        data.drop(drop_columns, axis=1, inplace=True)
        data["Age"].fillna(data["Age"].median(), inplace=True)
        data["Embarked"].fillna("S", inplace=True)
        return data

    @task
    def encode_data(data: pd.DataFrame):
        """Encodes categorical data to numerical data using specified mappings or one-hot encoding."""
        mappings = {
            "Sex": {"male": 1, "female": 0},
            "Embarked": {"S": 0, "C": 1, "Q": 2},
        }
        for col, mapping in mappings.items():
            if col in data.columns:
                data[col] = data[col].map(mapping)

        return data

    @task
    def split_and_scale_data(data: pd.DataFrame, random_state=None):
        X, y = data.drop("Survived", axis=1), data["Survived"]
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=random_state or 42
        )
        scaler = StandardScaler().fit(train_X)
        train_X_scaled = scaler.transform(train_X)
        test_X_scaled = scaler.transform(test_X)
        joblib.dump(scaler, SCALER_FILE)

        return train_X_scaled, test_X_scaled, train_y.values, test_y.values

    @task
    def train_and_evaluate(data):
        """Trains models and evaluates them returning the model with the best accuracy."""
        train_X, test_X, train_y, test_y = data
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": GaussianNB(),
        }
        best_accuracy = 0
        best_model_name = ""
        for name, model in models.items():
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            accuracy = accuracy_score(test_y, predictions)
            if accuracy > best_accuracy:
                best_model_name, best_accuracy = name, accuracy
                joblib.dump(model, BEST_MODEL_FILE)

        logging.info(
            f"Best model: {best_model_name} with accuracy: {best_accuracy:.2f}"
        )
        logging.info(
            "Model evaluation metrics:\n" + classification_report(test_y, predictions)
        )
        return f"Best model: {best_model_name} with accuracy: {best_accuracy:.2f}"

    data = fetch_data()
    preprocessed_data = preprocess_data(data)
    encoded_data = encode_data(preprocessed_data)
    dataset = split_and_scale_data(encoded_data)
    results = train_and_evaluate(dataset)

    return results


titanic_dag = titanic_training_dag()
