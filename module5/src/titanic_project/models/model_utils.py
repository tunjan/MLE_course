import joblib
from typing import Any, Callable
import pandas as pd
from pathlib import Path

def load_model(model_path: Path) -> Any:
    """Load the serialized model from a file."""
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def load_transformer(transformer_path: Path) -> Callable:
    """Load the serialized data transformer from a file."""
    with open(transformer_path, 'rb') as f:
        transformer = joblib.load(f)
    return transformer

def predict(model: Any, transformer: Callable, data: pd.DataFrame) -> pd.Series:
    """Make predictions using the loaded model and transformer."""
    transformed_data = transformer.transform(data)
    predictions = model.predict(transformed_data)
    return predictions