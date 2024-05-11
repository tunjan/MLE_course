import logging
from typing import Union

import pandas as pd

from ..data.data_utils import preprocess_data, encode_data

logger = logging.getLogger(__name__)

def preprocess_input(data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    """
    Preprocess the input data for prediction.

    Args:
        data (Union[pd.DataFrame, dict]): Input data to be preprocessed. It can be a Pandas DataFrame or a dictionary.

    Returns:
        pd.DataFrame: Preprocessed and encoded input data.

    Raises:
        ValueError: If the input data is not a Pandas DataFrame or a dictionary.
    """
    if not isinstance(data, (pd.DataFrame, dict)):
        logger.error("Input data must be a Pandas DataFrame or a dictionary.")
        raise ValueError("Input data must be a Pandas DataFrame or a dictionary.")

    if isinstance(data, dict):
        data = pd.DataFrame(data)

    try:
        preprocessed_data = preprocess_data(data)
        encoded_data = encode_data(preprocessed_data)
    except Exception as e:
        logger.error(f"Error during preprocessing or encoding: {e}")
        raise

    return encoded_data
