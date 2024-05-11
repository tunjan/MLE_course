import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename

from ..data.data_utils import preprocess_data, encode_data
from ..models.model_utils import load_model, load_scaler, predict
from ..batch.batch_pipeline import run_batch_predictions

app = Flask(__name__)
app.secret_key = 'secret_key' 

PROJECT_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
INPUT_PATH = PROJECT_DIR / "input"
OUTPUT_PATH = PROJECT_DIR / "output/predictions.csv"

model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/', methods=['GET', 'POST'])
def predict_endpoint():
    if request.method == 'POST':
        action = request.form.get('action', '')

        if action == 'SINGLE PREDICTION':
            required_keys = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            if all(key in request.form for key in required_keys):
                user_data = {
                    'Pclass': [int(request.form['Pclass'])],
                    'Sex': [request.form['Sex']],
                    'Age': [float(request.form['Age'])],
                    'SibSp': [int(request.form['SibSp'])],
                    'Parch': [int(request.form['Parch'])],
                    'Fare': [float(request.form['Fare'])],
                    'Embarked': [request.form['Embarked']]
                }

                data = pd.DataFrame(user_data)
                preprocessed_data = preprocess_data(data)
                encoded_data = encode_data(preprocessed_data)
                predictions = predict(model, scaler, encoded_data)
                return render_template('index.html', predictions=predictions.tolist(), show_single_form=True)
            else:
                logger.warning(f"Missing required form data for single prediction: {', '.join(required_keys)}")
                return render_template('index.html', message='Missing required form data', show_single_form=True)

        elif action == 'TOGGLE SINGLE PREDICTION':
            session['show_form'] = not session.get('show_form', False)
            return render_template('index.html', show_single_form=session['show_form'])

        elif action == 'BATCH PREDICTION':
            if 'file' not in request.files:
                logger.warning("No file provided for batch prediction")
                return render_template('index.html', message="No file part")

            file = request.files['file']
            if file.filename == '':
                logger.warning("No file selected for batch prediction")
                return render_template('index.html', message="No selected file")

            if file:
                filename = secure_filename(file.filename)
                file_path = INPUT_PATH / filename
                file.save(file_path)

                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                batch_data = run_batch_predictions(file_path, OUTPUT_PATH, model, scaler)
                return render_template('index.html', batch_data=batch_data.to_html(classes='batch-table', index=False))

    return render_template('index.html', show_single_form=session.get('show_form', False))

def main():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
	main()
