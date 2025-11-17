from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../model")

# Load model
W1 = np.load(os.path.join(MODEL_DIR, "W1.npy"))
W2 = np.load(os.path.join(MODEL_DIR, "W2.npy"))
b1 = np.load(os.path.join(MODEL_DIR, "b1.npy"))
b2 = np.load(os.path.join(MODEL_DIR, "b2.npy"))

scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))

# Fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form['temperature'])
    ammonia = float(request.form['ammonia'])
    feed = float(request.form['feeding'])
    chickens = float(request.form['chickens'])

    # Gabungkan input
    user_input = np.array([[temp, ammonia, feed]])

    # Normalisasi
    X_scaled = scaler_X.transform(user_input)

    # Forward pass ANN
    hidden_input = np.dot(X_scaled, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_output = sigmoid(np.dot(hidden_output, W2) + b2)

    # Prediksi per ayam
    egg_per_chicken = scaler_y.inverse_transform(final_output)[0][0]

    # Total telur
    prediction = egg_per_chicken * chickens

    return render_template("index.html", result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
