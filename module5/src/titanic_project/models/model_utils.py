import logging
from typing import Any, Callable
from pathlib import Path

import pandas as pd
import joblib

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    """
    Load the serialized model from a file.

    Args:
        model_path (Path): Path to the serialized model file.

    Returns:
        Any: The loaded model object.
    """
    try:
        with model_path.open('rb') as f:
            model = joblib.load(f)
    except (IOError, ValueError) as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
    else:
        logger.info(f"Model loaded from {model_path}")
        return model

def load_scaler(scaler_path: Path) -> Callable:
    """
    Load the serialized data scaler from a file.

    Args:
        scaler_path (Path): Path to the serialized scaler file.

    Returns:
        Callable: The loaded scaler object.
    """
    try:
        with scaler_path.open('rb') as f:
            scaler = joblib.load(f)
    except (IOError, ValueError) as e:
        logger.error(f"Error loading scaler from {scaler_path}: {e}")
        raise
    else:
        logger.info(f"scaler loaded from {scaler_path}")
        return scaler

def predict(model: Any, scaler: Callable, data: pd.DataFrame) -> pd.Series:
    """
    Make predictions using the loaded model and scaler.

    Args:
        model (Any): The loaded model object.
        scaler (Callable): The loaded scaler object.
        data (pd.DataFrame): The input data for prediction.

    Returns:
        pd.Series: The predicted values.
    """
    transformed_data = scaler.transform(data)
    predictions = model.predict(transformed_data)
    return pd.Series(predictions)
    
    
    
    
    







