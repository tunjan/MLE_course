import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import uniform

FIGURE_DIR = "figures"
RESIDUALS_FIGURE_SIZE = (10, 6)
RESIDUALS_DPI = 300
MLFLOW_SERVER_URI = "http://mlflow-server:5000"
MLFLOW_EXPERIMENT_NAME = "california_housing"
BEST_MODEL_ARTIFACT_PATH = "best_model"
N_ITER = 128
CV = 3


class DataLoader:
    def __init__(self, test_size=0.2, random_state=1):
        self.test_size = test_size
        self.random_state = random_state

    def load_and_split_data(self):
        housing = fetch_california_housing()
        X, y = housing.data, housing.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test


class Preprocessor:
    def __init__(self, k_features=5):
        self.k_features = k_features

    def standardize_and_select_features(self, X_train, X_test, y_train):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(f_regression, k=self.k_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        return X_train_selected, X_test_selected


class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.lowest_rmse = None
        self.best_name = None

    @staticmethod
    def calculate_residuals(y_true, predictions):
        return y_true - predictions

    @staticmethod
    def calculate_rmse(y_true, predictions):
        return np.sqrt(mean_squared_error(y_true, predictions))

    def log_and_compare(self, model, name, test_rmse):
        if self.lowest_rmse is None or test_rmse < self.lowest_rmse:
            self.best_model = model
            self.best_name = name
            self.lowest_rmse = test_rmse

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model, name, save_plots=True):
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_rmse = self.calculate_rmse(y_train, train_preds)
        test_rmse = self.calculate_rmse(y_test, test_preds)
        self.log_and_compare(model, name, test_rmse)

        if save_plots:
            plot_residuals(y_train, train_preds, y_test, test_preds, name)

        return train_rmse, test_rmse


def plot_residuals(y_train, train_preds, y_test, test_preds, name):
    train_residuals = y_train - train_preds
    test_residuals = y_test - test_preds

    fig, ax = plt.subplots(figsize=RESIDUALS_FIGURE_SIZE)
    sns.kdeplot(train_residuals, fill=True, label="Train", ax=ax)
    sns.kdeplot(test_residuals, fill=True, label="Test", ax=ax)

    ax.set_title(f"{name} Residuals")
    ax.legend()

    plt.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_path = os.path.join(FIGURE_DIR, f"{name}_residuals.png")
    fig.savefig(plot_path, dpi=RESIDUALS_DPI, bbox_inches="tight")
    plt.close(fig)


def run_experiment(X_train, X_test, y_train, y_test):
    models = [
        ("Linear_Regression", LinearRegression(), {}),
        ("Ridge_Regression", Ridge(), {"alpha": uniform(0.01, 10)}),
        ("Lasso_Regression", Lasso(), {"alpha": uniform(0.01, 10)}),
        ("Elastic_Net_Regression", ElasticNet(), {"alpha": uniform(0.01, 10), "l1_ratio": uniform(0, 1)}),
    ]

    trainer = ModelTrainer()

    for name, model, params in models:
        with mlflow.start_run(run_name=name):
            if params:
                search = RandomizedSearchCV(model, params, n_iter=N_ITER, cv=CV)
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                _, _ = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, best_model, name)
            else:
                _, _ = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, model, name)

    # Optional: Log the best model
    if trainer.best_model:
        mlflow.sklearn.log_model(trainer.best_model, artifact_path=BEST_MODEL_ARTIFACT_PATH)


def main():
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_and_split_data()

    preprocessor = Preprocessor()
    X_train_processed, X_test_processed = preprocessor.standardize_and_select_features(X_train, X_test, y_train)

    run_experiment(X_train_processed, X_test_processed, y_train, y_test)


if __name__ == "__main__":
    main()
