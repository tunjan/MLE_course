import pytest
import pandas as pd
from pathlib import Path
from src.titanic_project.data.data_utils import load_data, preprocess_data, encode_data

TEST_DATA_PATH = Path("/home/tunjan/Documents/github/MLE_course/module5/tests/data/test_titanic.csv")
TEST_DATA = pd.read_csv(TEST_DATA_PATH)

def test_load_data():
    data = load_data(TEST_DATA_PATH)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    data = preprocess_data(TEST_DATA)
    required_drops = {"PassengerId", "Name", "Ticket", "Cabin"}
    dropped_cols = set(TEST_DATA.columns) - set(data.columns)
    assert dropped_cols == required_drops
    assert data["Age"].isna().sum() == 0
    assert data["Embarked"].isna().sum() == 0

def test_encode_data():
    data = preprocess_data(TEST_DATA)
    encoded_data = encode_data(data)
    assert "Sex" in encoded_data.columns
    assert encoded_data["Sex"].dtype == int
    assert "Embarked" in encoded_data.columns
    assert encoded_data["Embarked"].dtype == int