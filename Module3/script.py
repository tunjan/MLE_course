import os
import mlflow
import mlflow.sklearn
import numpy as np
from scipy.stats import uniform
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns


class HousingModel:
    """
    A class to handle the California housing dataset and model training.
    """

    def __init__(self, test_size=0.2, random_state=1, k_features=5):
        self.test_size = test_size
        self.random_state = random_state
        self.k_features = k_features
        self.best_model = None
        self.lowest_rmse = None

    def load_data(self):
        """Load the California housing dataset."""
        housing = fetch_california_housing()
        self.X, self.y = housing.data, housing.target

    def split_data(self):
        """Split the dataset into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def preprocess_data(self):
        """Preprocess the data by standardizing features."""
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select_features(self):
        """Select top K features using f_regression."""
        selector = SelectKBest(f_regression, k=self.k_features)
        selector.fit(self.X_train, self.y_train)
        self.X_train = selector.transform(self.X_train)
        self.X_test = selector.transform(self.X_test)

    def train_and_evaluate_model(self, model, name):
        """Train and evaluate a given model."""
        model.fit(self.X_train, self.y_train)
        train_rmse, test_rmse = self.calculate_rmse(model)
        train_residuals, test_residuals = self.calculate_residuals(model)
        self.log_metrics(model, name, train_rmse, test_rmse)
        self.plot_residuals(name, train_residuals, test_residuals)

    def plot_residuals(self, name, train_residuals, test_residuals):
        """Plot residuals for the given model."""
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.kdeplot(train_residuals, fill=True, label="Train", ax=ax)
        sns.kdeplot(test_residuals, fill=True, label="Test", ax=ax)

        ax.set_title(f"{name} Residuals")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()

        residuals_dir = "figures"
        os.makedirs(residuals_dir, exist_ok=True)

        plot_filename = f"{name}_residuals.png"
        plot_path = os.path.join(residuals_dir, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def calculate_residuals(self, model):
        """Calculate residuals for the given model."""
        train_residuals = self.y_train - model.predict(self.X_train)
        test_residuals = self.y_test - model.predict(self.X_test)
        return train_residuals, test_residuals

    def calculate_rmse(self, model):
        """Calculate RMSE for training and testing sets."""
        train_rmse = np.sqrt(
            mean_squared_error(self.y_train, model.predict(self.X_train))
        )
        test_rmse = np.sqrt(mean_squared_error(self.y_test, model.predict(self.X_test)))
        return train_rmse, test_rmse

    def log_metrics(self, model, name, train_rmse, test_rmse):
        """Log model metrics to MLflow if it has the lowest RMSE."""
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        if self.lowest_rmse is None or test_rmse < self.lowest_rmse:
            self.lowest_rmse = test_rmse
            self.best_model = model
            self.best_name = name

    def run_experiment(self):
        # Set up MLflow experiment
        mlflow.set_experiment("california_housing")

        # Linear models with hyperparameters
        models = [
            ("Linear_Regression", LinearRegression(), {}),
            ("Ridge_Regression", Ridge(), {"alpha": uniform(0.01, 10)}),
            ("Lasso_Regression", Lasso(), {"alpha": uniform(0.01, 10)}),
            (
                "Elastic_Net_Regression",
                ElasticNet(),
                {"alpha": uniform(0.01, 10), "l1_ratio": uniform(0, 1)},
            ),
        ]

        for name, model, params in models:
            with mlflow.start_run(run_name=name):
                mlflow.sklearn.autolog(disable=True)

                if params:
                    self.perform_hyperparameter_tuning(model, params)
                else:
                    self.train_and_evaluate_model(model, name)

        # Log the best model with the lowest RMSE
        if self.best_model is not None:
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                input_example=self.X_train,
                artifact_path=self.best_name,
                registered_model_name="best_model",
            )

    def perform_hyperparameter_tuning(self, model, param_distributions):
        """Perform randomized search for hyperparameter tuning."""
        search = RandomizedSearchCV(model, param_distributions, n_iter=128, cv=3)
        search.fit(self.X_train, self.y_train)
        best_model = search.best_estimator_
        self.train_and_evaluate_model(best_model, model.__class__.__name__)


def main():
    housing_model = HousingModel()
    housing_model.load_data()
    housing_model.split_data()
    housing_model.preprocess_data()
    housing_model.select_features()
    housing_model.run_experiment()


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    main()
