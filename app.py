from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


# ==========================================================
# App
# ==========================================================

app = Flask(__name__)
CORS(app)


# ==========================================================
# Paths
# ==========================================================

MODEL_PATH = Path("models") / "best_pipeline.pkl"


# ==========================================================
# Load Model
# ==========================================================

model = joblib.load(MODEL_PATH)


# ==========================================================
# Required Features
# ==========================================================

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# ==========================================================
# Home
# ==========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "project": "Telco Churn Prediction API",
            "status": "running",
            "model": "Logistic Regression Pipeline",
        }
    )


# ==========================================================
# Health Check
# ==========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
        }
    )


# ==========================================================
# Prediction
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if data is None:
            return jsonify(
                {
                    "error": "No JSON data received."
                }
            ), 400

        # -----------------------------
        # Validate required fields
        # -----------------------------

        missing = [
            col
            for col in REQUIRED_COLUMNS
            if col not in data
        ]

        if missing:

            return jsonify(
                {
                    "error": "Missing required fields.",
                    "missing_columns": missing,
                }
            ), 400

        # -----------------------------
        # DataFrame
        # -----------------------------

        input_df = pd.DataFrame([data])

        # -----------------------------
        # Numeric conversion
        # -----------------------------

        numeric_cols = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]

        for col in numeric_cols:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce",
            )

        # -----------------------------
        # Prediction
        # -----------------------------

        probability = model.predict_proba(input_df)[0][1]

        prediction = int(
            model.predict(input_df)[0]
        )

        label = (
            "Churn"
            if prediction == 1
            else "No Churn"
        )

        # -----------------------------
        # Response
        # -----------------------------

        return jsonify(
            {
                "prediction": prediction,
                "label": label,
                "churn_probability": round(
                    float(probability),
                    4,
                ),
            }
        )

    except Exception as e:

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False,
    )