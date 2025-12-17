from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "models/model.pkl"

# -----------------------
# Load model at startup
# -----------------------

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Model file not found. Train the model first by running train_model.py"
    )

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
CLASS_NAMES = artifact["class_names"]

# -----------------------
# Routes
# -----------------------

@app.route("/")
def home():
    return "Iris model inference service running", 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' field"}), 400

    features = data["features"]

    if len(features) != 4:
        return jsonify(
            {"error": "Expected 4 features: sepal length, sepal width, petal length, petal width"}
        ), 400

    X_input = np.array(features).reshape(1, -1)

    prediction = int(model.predict(X_input)[0])
    probabilities = model.predict_proba(X_input)[0]

    return jsonify(
        {
            "prediction": CLASS_NAMES[prediction],
            "class_index": prediction,
            "probabilities": {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            },
            "features": dict(zip(FEATURE_NAMES, features)),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
