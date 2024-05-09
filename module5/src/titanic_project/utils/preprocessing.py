from ..data.data_utils import preprocess_data, encode_data

def preprocess_input(data):
    """Preprocess the input data for prediction."""
    preprocessed_data = preprocess_data(data)
    encoded_data = encode_data(preprocessed_data)
    return encoded_data
