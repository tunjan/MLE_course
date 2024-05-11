from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from titanic_project.batch.batch_pipeline import run_batch_predictions
from titanic_project.data.data_utils import load_data, preprocess_data, encode_data
from titanic_project.models.model_utils import load_model, load_scaler


PROJECT_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DATA_DIR = PROJECT_DIR / "src" / "titanic_project" / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
TEST_DATA_PATH = DATA_DIR / "titanic.csv"

# Define default arguments for the DAG
default_args = {
    'owner': 'Alberto',
    'start_date': datetime(2023, 5, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Function to load model and scaler
def load_model_and_scaler():

    model_path = ARTIFACTS_DIR / "model.pkl"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Save model and scaler to temporary files
    temp_model_path = Path("/tmp/model.pkl")
    temp_scaler_path = Path("/tmp/scaler.pkl")

    joblib.dump(model, temp_model_path)
    joblib.dump(scaler, temp_scaler_path)

    return str(temp_model_path), str(temp_scaler_path)

# Function to run tests
def run_tests_on_code(**context):

    test_data = load_data(TEST_DATA_PATH)

    def test_load_data():
        data = load_data(TEST_DATA_PATH)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_preprocess_data():
        data = preprocess_data(test_data)
        required_drops = {"PassengerId", "Name", "Ticket", "Cabin"}
        dropped_cols = set(test_data.columns) - set(data.columns)
        assert dropped_cols == required_drops
        assert data["Age"].isna().sum() == 0
        assert data["Embarked"].isna().sum() == 0

    def test_encode_data():
        data = preprocess_data(test_data)
        encoded_data = encode_data(data)
        assert "Sex" in encoded_data.columns
        assert encoded_data["Sex"].dtype == int
        assert "Embarked" in encoded_data.columns
        assert encoded_data["Embarked"].dtype == int

    test_load_data()
    test_preprocess_data()
    test_encode_data()

# Function to run batch predictions
def dag_run_batch_predictions(**context):

    data_path = DATA_DIR / "titanic.csv"
    output_path = OUTPUT_DIR / "predictions.csv"

    # Get the model and scaler file paths from the XCom
    temp_model_path, temp_scaler_path = context["ti"].xcom_pull(task_ids="load_artifacts", key="return_value")

    # Load the model and scaler from the file paths
    model = joblib.load(temp_model_path)
    scaler = joblib.load(temp_scaler_path)

    predictions = run_batch_predictions(data_path, output_path, model, scaler)

    return predictions

with DAG('batch_prediction_pipeline', default_args=default_args, schedule_interval=None) as dag:

    # Task to load the model and scaler
    load_artifacts = PythonOperator(
        task_id='load_artifacts',
        python_callable=load_model_and_scaler
    )

    # Task to run tests on the code
    run_tests = PythonOperator(
        task_id='run_tests',
        python_callable=run_tests_on_code
    )

    # Task to run batch predictions
    batch_prediction = PythonOperator(
        task_id='batch_prediction',
        python_callable=dag_run_batch_predictions
    )

    # Task to build and run the Docker container

    build_image = BashOperator(
    	task_id='build_docker_image',
    	bash_command=f'docker build -t my-titanic-project {PROJECT_DIR}',
    	dag=dag,
    )
    
    run_image = BashOperator(
    	task_id='run_docker_image',
    	bash_command='docker run -d -p 5000:5000 my-titanic-project',
    	dag=dag,
    )


    # Set task dependencies
    load_artifacts >> run_tests >> batch_prediction >> build_image >> run_image
