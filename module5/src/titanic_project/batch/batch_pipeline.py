import pandas as pd
from pathlib import Path
from ..data.data_utils import load_data, preprocess_data, encode_data
from ..models.model_utils import load_model, load_transformer, predict

def run_batch_predictions(data_path: Path, output_path: Path) -> None:
    """
    Run batch predictions on the input data and save the results.

    Args:
        data_path (Path): The path to the input data file.
        output_path (Path): The path to save the predictions.

    Returns:
        None
    """
    # Load and preprocess the input data
    data = load_data(data_path)
    preprocessed_data = preprocess_data(data)
    encoded_data = encode_data(preprocessed_data)

    # Load the trained model and transformer
    model = load_model('/home/tunjan/Documents/github/MLE_course/module5/artifacts/model.pkl')
    transformer = load_transformer('/home/tunjan/Documents/github/MLE_course/module5/artifacts/scaler.pkl')

    # Make predictions using the loaded model and transformer
    predictions = predict(model, transformer, encoded_data)

    # Save predictions to the output path
    pd.DataFrame({'predictions': predictions}).to_csv(output_path, index=False)