import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "Healthy"}), 200


@app.route("/invocations", methods=["POST"])
def invoke():
    data = request.get_json()
    df = pd.DataFrame([data])
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
