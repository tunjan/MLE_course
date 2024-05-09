import os
from datetime import timedelta
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
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
    catchup=False,  # Best Practice: Disabling catchup prevents backfilling when not needed.
    tags=["example"],
)
def titanic_training_dag():
    @task
    def fetch_data():
        data = pd.read_csv(DATA_URL)
        return data

    @task
    def preprocess_data(data: pd.DataFrame):
        required_drops = {"PassengerId", "Name", "Ticket", "Cabin"}
        data = data.drop(columns=[col for col in required_drops if col in data.columns])
        data["Age"].fillna(data["Age"].median(), inplace=True)
        data["Embarked"].fillna("S", inplace=True)
        return data

    @task
    def encode_data(data: pd.DataFrame):
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
    def train_model(train_X, train_y, model_name):
        """Trains a single model and returns the trained model."""
        model = models[model_name]
        model.fit(train_X, train_y)
        return model

    @task
    def evaluate_model(model, test_X, test_y):
        """Evaluates a trained model and returns the accuracy and classification report."""
        predictions = model.predict(test_X)
        accuracy = accuracy_score(test_y, predictions)
        report = classification_report(test_y, predictions)
        return accuracy, report

    @task
    def train_and_evaluate(data):
        """Trains multiple models, evaluates them, and returns the best model."""
        train_X, test_X, train_y, test_y = data
        best_accuracy = 0
        best_model_name = ""
        for name in models:
            model = train_model(train_X, train_y, name)
            accuracy, report = evaluate_model(model, test_X, test_y)
            if accuracy > best_accuracy:
                best_model_name, best_accuracy = name, accuracy
        joblib.dump(model, BEST_MODEL_FILE)
        logging.info(
            f"Best model: {best_model_name} with accuracy: {best_accuracy:.2f}"
        )
        logging.info("Model evaluation metrics:\n" + report)
        return f"Best model: {best_model_name} with accuracy: {best_accuracy:.2f}"
        data = fetch_data()
        preprocessed_data = preprocess_data(data)
        encoded_data = encode_data(preprocessed_data)
        dataset = split_and_scale_data(encoded_data)
        results = train_and_evaluate(dataset)

        return results


titanic_dag = titanic_training_dag()
