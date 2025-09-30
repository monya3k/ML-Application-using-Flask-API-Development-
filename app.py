from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

MODEL_FILE = "btc_model.pkl"

# Save model
def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# Load model
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

@app.route("/train", methods=["POST"])
def train():
    """
    Train regression model to predict BTC closing price.
    Features: Open, High, Low, Volume
    Target: Close
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    save_model(model)

    y_pred = model.predict(X)

    return jsonify({
        "message": "Model trained successfully for BTC Close price prediction",
       
    })

@app.route("/test", methods=["POST"])
def test():
    """
    Test the trained model on BTC test dataset.
    """
    try:
        model = load_model()
    except:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return jsonify({
        "test_mse": mse,
        "test_r2": r2
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict BTC closing price from JSON input.
    Example:
    {
        "OHLV": [45000, 46000, 44000, 35000000000]
    }
    """
    try:
        model = load_model()
    except:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    data = request.get_json()
    if not data or "OHLV" not in data:
        return jsonify({"error": "No OHLV provided"}), 400

    OHLV = np.array([data["OHLV"]])  # must be 2D
    prediction = model.predict(OHLV)[0]

    return jsonify({"predicted_close_price": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)