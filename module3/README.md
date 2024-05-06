# California Housing MLOps Project

This repository contains code for a California housing price prediction project using MLflow for experiment tracking and model management. It includes a Docker Compose setup for running the MLflow server and a client container for training and evaluating different linear regression models on the California housing dataset.

# ToC
- [California Housing MLOps Project](#california-housing-mlops-project)
- [ToC](#toc)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Build and start the Docker containers](#build-and-start-the-docker-containers)
    - [Access MLFlow UI](#access-mlflow-ui)
    - [Save models](#save-models)
  - [Deployment](#deployment)
  - [Project Structure](#project-structure)



## Prerequisites

- Docker
- Docker Compose

## Setup

Clone the repository:

```bash
git clone https://github.com/tunjan/MLE_course
cd MLE_course/Module3
```


## Usage

### Build and start the Docker containers
```
make run
```

This command will build the MLflow server and client Docker images, create a Docker network, and start the containers. The training process will automatically start in the client container.

### Access MLFlow UI

After the containers are up and running, you can access the MLflow UI at the address listed in the `docker compose logs mlflow-server` command output.
You can view the experiment runs, logged metrics, and artifacts in the MLflow UI.

### Save models

``` bash
make save
```

This will copy the trained model artifacts and residual plots from the Docker containers to the local artifacts and figures directories, respectively.

## Deployment

To deploy the best-performing model as a Docker container, run the following command:

```
make deploy
```
This will build a Docker image for the best model and set the appropriate environment variable for using the local MLflow server for model serving.

## Project Structure

- **Dockerfile.server:** Dockerfile for the MLflow server container.
- **Dockerfile.client:** Dockerfile for the client container that runs the training code.
- **docker-compose.yml:** Docker Compose configuration file for setting uthe MLflow server and client containers.
- **client.py:** Python code for loading the California housindataset, preprocessing, training, and evaluating linear regressio   models.
- **Makefile:** Makefile with commands for running, saving artifacts, an deploying the best model.


