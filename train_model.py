import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "model.pkl"

def train_and_save_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    model_artifact = {
        "model": model,
        "feature_names": iris.feature_names,
        "class_names": iris.target_names.tolist(),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
