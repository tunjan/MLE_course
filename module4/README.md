# Titanic Survival Prediction Model Training

<!--toc:start-->
- [Titanic Survival Prediction Model Training](#titanic-survival-prediction-model-training)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Files](#files)
  - [DAG Overview](#dag-overview)
  - [Model Persistence](#model-persistence)
  - [Contributing](#contributing)
  - [License](#license)
<!--toc:end-->

This project contains an Apache Airflow DAG (Directed Acyclic Graph) that trains machine learning models to predict survival on the Titanic disaster using the Titanic dataset. The DAG fetches the data, preprocesses it, encodes categorical features, splits the data into training and testing sets, trains and evaluates two models (Logistic Regression and Naive Bayes), and selects the model with the highest accuracy. The best model is then persisted for future use.

## Prerequisites

- Docker
- Docker Compose

## Setup

  1. Clone the repository:
```bash
    git clone https://github.com/tunjan/MLE_course.git
    cd MLE_couse/module4
    mkdir models
```
  2. Build the Docker image:
```docker
    docker-compose build
```

  3. Start the Airflow services:
```docker
    docker-compose up -d
```
  4. Access the Airflow UI at http://localhost:8080 and log in with the default credentials (username: airflow, password: airflow).
  5. Unpause the titanic_model_training DAG, and then manually trigger it.


## Files

  - `titanic.py`: Contains the Airflow DAG definition and the model training pipeline.
  - `docker-compose.yaml`: Docker Compose configuration file for setting up the Airflow environment.
  - `requirements.txt`: List of Python dependencies required by the project.
  - `Dockerfile`: Dockerfile for building a custom Airflow image with the required dependencies.

## DAG Overview

The titanic_model_training DAG performs the following tasks:

  1. Fetch Data: Fetches the Titanic dataset from a GitHub repository.
  2. Preprocess Data: Drops non-essential columns and fills missing values in the dataset.
  3. Encode Data: Encodes categorical features using predefined mappings or one-hot encoding.
  4. Split and Scale Data: Splits the data into training and testing sets, and scales the features using StandardScaler.
  5. Train and Evaluate: Trains Logistic Regression and Naive Bayes models on the training data, evaluates their performance 
  6. on the testing data, and selects the model with the highest accuracy. The best model is persisted to disk.

## Model Persistence

The trained model and the scaler used for feature scaling are persisted in the `/opt/airflow/models` directory inside the Docker container. To access these files from the host machine, mount a volume to this directory when starting the Airflow services.
## Contributing

  Contributions are welcome! Please open an issue or submit a pull request.
## License

  This project is licensed under the MIT License.
