# Titanic Survival Prediction Model

This repository contains a Machine Learning model for predicting survival on the Titanic, along with the necessary code and configurations for deploying the model as a web service using Amazon SageMaker.

## Requirements

- aws-cli
- boto3
- docker
- python
- pylint

## Files

- `Dockerfile`: Defines the Docker image used for deploying the model as a web service.
- `endpoint-test.py`: A Python script for testing the deployed model endpoint.
- `inference.py`: The main server script that loads the model and serves predictions via a Flask app.
- `requirements.txt`: Lists the Python package dependencies required for running the Flask app.

## Usage

1. Build the Docker image from the provided `Dockerfile`.
2. Deploy the Docker image as a model on Amazon SageMaker.
3. Test the deployed model endpoint using the `endpoint-test.py` script.

The `inference.py` script exposes two endpoints:

- `/ping`: Returns a simple "Healthy" status to check if the server is running.
- `/invocations`: Accepts a JSON payload containing the passenger data and returns the predicted survival probability.

## Testing

```bash
./setup.sh <PUBLIC KEY> <SECRET KEY>
```

## Quality check

```bash
pylint --rcfile=pylintrc inference.py
```

## Docker

```bash
docker build -t aws-script .
docker run --rm aws-script <access_key_id> <secret_access_key>
```
