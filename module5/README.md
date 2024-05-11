# Module 5 Project

This project consists of a Flask-based web application and Airflow pipeline that allows users to make predictions using a pre-trained machine learning model trained on the Titanic dataset. It supports both single predictions via a form and batch predictions via file uploads. The Airflow pipeline performs a batch prediction and then runs a Docker container that serves the Flask web application.

## Requirements
- docker
- python

## Features

- **Single Prediction**: Users can input values for various features (e.g., Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) and get a prediction from the pre-trained model on whether the passenger would have survived or not.
- **Batch Prediction**: Users can upload a CSV file containing multiple data points, and the application will generate predictions for the entire dataset.
- **Data Preprocessing**: Input data is automatically preprocessed and encoded before making predictions, ensuring consistency with the training data.
- **Model and Transformer Loading**: The application loads the pre-trained model and data transformer from serialized files stored in the `artifacts` directory.
- **Airflow Pipeline**: An Airflow DAG is included to automate the batch prediction process and serve the Flask web application using Docker.

## Installation

1. Clone the repository:

`git clone https://github.com/tunjan/MLE_course.git`


2. Navigate to the project directory:

`cd MLE_course/module5`

3. Create a python environment:
	
`python3 -m venv titanic_env`
`source titanic_env/bin/activate`

4. Install the required dependencies:

`pip install -e .[dev]`

## Usage

### Running the web app

To start the Flask server and access the web application, run the following command:

`start-online-server`

The server will start running on `http://127.0.0.1:5000`. Open your web browser and navigate to this address to access the application.

### Running the Airflow Pipeline

1. Start the airflow app.
`airflow db init`

2. Create a user.

```airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin```

3. Go to `~/airflow/airflow.cfg` and change the `dags_folder` path to the one of this repository `~path-to-this-repo/module5/src/titanic_project/dags`. Also, set `load_examples = False` in the `airflow.cfg` file.

4. Stop the airflow app and load it again with. `airflow scheduler & airflow webserver --port 8080` and run start the Titanic DAG.

5.	Run the Titanic Project DAG. It will automatically take the file in the `input/` directory and return the predictions file in the `output/` directory.
6. When the last step of the pipeline has been executed you can access the flask web app at `http://127.0.0.1:5000`

## Project Structure

my-project/
├── src/
│   └── titanic_project/
│       ├── batch/
│       │   └── batch_pipeline.py
│       ├── data/
│       │   └── data_utils.py
│       ├── models/
│       │   └── model_utils.py
│       └── online/
│           └── app.py
├── artifacts/
│   ├── model.pkl
│   └── scaler.pkl
├── input/
├── output/
├── setup.py
├── requirements.txt
└── README.md

- `src/titanic_project/batch/batch_pipeline.py`: Contains the code for running the batch prediction.
- `src/titanic_project/data/data_utils.py`: Contains utility functions for data preprocessing and encoding.
- `src/titanic_project/models/model_utils.py`: Contains utility functions for loading the pre-trained model and data transformer.
- `src/titanic_project/online/app.py`: Contains the Flask application code for the online prediction server.
- `src/titanic_project/dags/main_dag.py`: Contains the airflow pipeline that load the data, performs tests, and serves the app.
- `artifacts/`: Directory containing the serialized model and data transformer files.
- `input/`: Directory for storing input CSV files for batch predictions.
- `output/`: Directory where the batch prediction results will be saved.
- `setup.py`: Python setup script for installing and distributing the project.
- `requirements.txt`: File listing the project's Python dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
