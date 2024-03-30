import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import uniform, randint

# Set the MLflow tracking URI to point to the MLflow server
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_data():
    # Load the dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print(f"Train score: {train_score:.3f}, Test score: {test_score:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")

    return model


def randomized_search(
    model,
    X_train,
    y_train,
    param_distributions,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
):
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print("Best score:", search.best_score_)

    return search.best_estimator_


def main():
    # Load the data
    X, y = load_data()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess the data
    X_train, X_test = preprocess_data(X_train, X_test)

    # Set up MLflow experiment
    mlflow.set_experiment("linear_models")

    # Linear models
    linear_models = [
        ("Linear Regression", LinearRegression(), {}),
        ("Ridge Regression", Ridge(), {"alpha": uniform(0.01, 10)}),
        ("Lasso Regression", Lasso(), {"alpha": uniform(0.01, 10)}),
        (
            "Elastic Net Regression",
            ElasticNet(),
            {"alpha": uniform(0.01, 10), "l1_ratio": uniform(0, 1)},
        ),
    ]

    for name, model, param_distributions in linear_models:
        with mlflow.start_run():
            print(f"\n{name}")
            if param_distributions:
                tuned_model = randomized_search(
                    model, X_train, y_train, param_distributions, n_iter=100, cv=5
                )
                trained_model = tuned_model
            else:
                trained_model = train_and_evaluate(
                    model, X_train, X_test, y_train, y_test
                )

            # Log model name, parameters, and metrics
            mlflow.log_param("model_name", name)
            params = trained_model.get_params()
            mlflow.log_params(params)

            train_score = trained_model.score(X_train, y_train)
            test_score = trained_model.score(X_test, y_test)
            train_rmse = np.sqrt(
                mean_squared_error(y_train, trained_model.predict(X_train))
            )
            test_rmse = np.sqrt(
                mean_squared_error(y_test, trained_model.predict(X_test))
            )

            mlflow.log_metric("train_score", train_score)
            mlflow.log_metric("test_score", test_score)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)

            # Log model artifact
            mlflow.sklearn.log_model(trained_model, f"{name}_model")


if __name__ == "__main__":
    main()
