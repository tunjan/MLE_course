import pandas as pd
from pathlib import Path

def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from a file or database."""
    with open(data_path, 'rb') as f:
        data = pd.read_csv(f)
    return data
    
    
    

CATEGORICAL_MAPPINGS = {
    "Sex": {"male": 1, "female": 0},
    "Embarked": {"S": 0, "C": 1, "Q": 2},
}

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data by dropping unnecessary columns and
    filling missing values with suitable replacements.

    Args:
        data (pd.DataFrame): The input data to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed data.
    """

    required_drops = {"PassengerId", "Name", "Ticket", "Cabin"}
    data = data.drop(columns=[col for col in required_drops if col in data.columns])
    data = data.copy()
    data.fillna({"Age": data["Age"].median(), "Embarked": "S"}, inplace=True)
    return data

def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns in the input data using the provided mappings.

    Args:
        data (pd.DataFrame): The input data to be encoded.

    Returns:
        pd.DataFrame: The data with categorical columns encoded.
    """

    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
    return data
