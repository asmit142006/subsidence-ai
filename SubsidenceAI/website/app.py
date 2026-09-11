import os
import pickle
import traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "subsidence_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "risk_encoder.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load encoder safely
encoder = None
if os.path.exists(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 0:
    try:
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load encoder ({e}).")

# Exact feature names expected by your trained XGBoost model
FEATURE_NAMES = [
    "mpu_acc_x_g", "mpu_acc_y_g", "mpu_acc_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "adxl_acc_x_g", "adxl_acc_y_g", "adxl_acc_z_g"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}

        row_data = {
            "mpu_acc_x_g": float(data.get("mpu_acc_x_g", 0.02)),
            "mpu_acc_y_g": float(data.get("mpu_acc_y_g", 0.01)),
            "mpu_acc_z_g": float(data.get("mpu_acc_z_g", 0.98)),
            "gyro_x_dps": float(data.get("gyro_x_dps", 0.1)),
            "gyro_y_dps": float(data.get("gyro_y_dps", 0.1)),
            "gyro_z_dps": float(data.get("gyro_z_dps", 0.05)),
            "adxl_acc_x_g": float(data.get("adxl_acc_x_g", 0.01)),
            "adxl_acc_y_g": float(data.get("adxl_acc_y_g", 0.02)),
            "adxl_acc_z_g": float(data.get("adxl_acc_z_g", 0.99)),
        }

        input_df = pd.DataFrame([row_data], columns=FEATURE_NAMES)
        pred_raw = model.predict(input_df)[0]

        # Handle numeric vs encoded labels safely
        if encoder:
            try:
                risk_label = encoder.inverse_transform([pred_raw])[0]
            except Exception:
                risk_label = str(pred_raw)
        else:
            label_map = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}
            risk_label = label_map.get(pred_raw, str(pred_raw))

        confidence = 85.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            confidence = round(float(np.max(probs)) * 100, 1)

        return jsonify({
            "risk": str(risk_label).upper(),
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        print("--- Prediction Error Details ---")
        traceback.print_exc()
        return jsonify({"risk": "ERROR", "confidence": "0%"}), 500

if __name__ == "__main__":
    app.run(debug=True)