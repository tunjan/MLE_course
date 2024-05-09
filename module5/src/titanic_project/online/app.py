from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename
from pandas import DataFrame
from pathlib import Path

from flask import Flask, request, render_template, session
from pandas import DataFrame
import numpy as np
from ..data.data_utils import preprocess_data, encode_data
from ..models.model_utils import load_model, load_transformer, predict
from ..batch.batch_pipeline import run_batch_predictions
from pathlib import Path


app = Flask(__name__)
app.secret_key = 'thekey'
app.config['UPLOAD_FOLDER'] = '/home/tunjan/Documents/github/MLE_course/module5/input/'  # Define a path to save uploaded files

model = load_model('/home/tunjan/Documents/github/MLE_course/module5/artifacts/model.pkl')
transformer = load_transformer('/home/tunjan/Documents/github/MLE_course/module5/artifacts/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict_endpoint():
    if request.method == 'POST':
        action = request.form.get('action', '')

        if action == 'SINGLE PREDICTION' and all(key in request.form for key in ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']):
            user_data = {
                'Pclass': [int(request.form['Pclass'])],
                'Sex': [request.form['Sex']],
                'Age': [float(request.form['Age'])],
                'SibSp': [int(request.form['SibSp'])],
                'Parch': [int(request.form['Parch'])],
                'Fare': [float(request.form['Fare'])],
                'Embarked': [request.form['Embarked']]
            }

            data = DataFrame(user_data)
            preprocessed_data = preprocess_data(data)
            encoded_data = encode_data(preprocessed_data)
            predictions = predict(model, transformer, encoded_data)
            return render_template('index.html', predictions=predictions.tolist(), show_single_form=session.get('show_form', False))

        if action == 'TOGGLE SINGLE PREDICTION':
            session['show_form'] = not session.get('show_form', False)
            return render_template('index.html', show_single_form=session['show_form'])

        elif action == 'BATCH PREDICTION':
            # Check if the post request has the file part
            if 'file' not in request.files:
                return render_template('index.html', message="No file part")

            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', message="No selected file")

            if file:
                filename = secure_filename(file.filename)
                file_path = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(file_path)

                output_path = Path('/home/tunjan/Documents/github/MLE_course/module5/output/predictions.csv')
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                run_batch_predictions(file_path, output_path)
                return render_template('index.html', message="Batch predictions saved successfully!")

    return render_template('index.html', show_single_form=session.get('show_form', False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
