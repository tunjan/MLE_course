import numpy as np
from scipy.stats import uniform
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def fetch_and_split_data():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    test_size = 0.2
    random_state = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save intermediate data
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    return "Data fetched and split successfully."


def process_data():
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_features = 5
    selector = SelectKBest(f_regression, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    np.save("X_train_processed.npy", X_train_selected)
    np.save("X_test_processed.npy", X_test_selected)

    return "Data processed and features selected."


def run_experiment():
    X_train = np.load("X_train_processed.npy")
    y_train = np.load("y_train.npy")

    models = [
        (LinearRegression(), {}),
        (Ridge(), {"alpha": uniform(0.01, 10)}),
        (Lasso(), {"alpha": uniform(0.01, 10)}),
        (ElasticNet(), {"alpha": uniform(0.01, 10), "l1_ratio": uniform(0, 1)}),
    ]

    results = {}
    for model, params in models:
        search = RandomizedSearchCV(model, params, n_iter=10, cv=5) if params else model
        search.fit(X_train, y_train)
        results[model.__class__.__name__] = (
            search.best_score_ if params else model.score(X_train, y_train)
        )

    return results
