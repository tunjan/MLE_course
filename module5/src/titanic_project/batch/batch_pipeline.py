import pandas as pd
from pathlib import Path
from ..data.data_utils import load_data, preprocess_data, encode_data
from ..models.model_utils import load_model, load_scaler, predict

def run_batch_predictions(data_path: Path, output_path: Path, model, scaler) -> pd.DataFrame:
    """
    Run batch predictions on the input data and save the results, and return the combined data and predictions.

    Args:
        data_path (Path): The path to the input data file.
        output_path (Path): The path to save the predictions.

    Returns:
        pd.DataFrame: DataFrame containing the original data with predictions appended.
    """
    data = load_data(data_path)
    preprocessed_data = preprocess_data(data)
    encoded_data = encode_data(preprocessed_data)

    predictions = predict(model, scaler, encoded_data)

    # Append predictions to the original data
    result = data.assign(predictions=predictions)
    result.to_csv(output_path, index=False)
    return result
