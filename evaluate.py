import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    load_mat_data,
    extract_features_advanced,
    labels_to_3class,
    subject_wise_split,
    normalize_dataset
)

MODEL_DIR = "models"

def plot_confusion_matrix(y_true, y_pred, name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(name, dpi=300)
    plt.close()


def load_test_data():
    print("Loading data...")
    X_raw, y10, subjects, tasks = load_mat_data("data/Data/filtered_data")

    # Extract features for classical models
    X_feats = extract_features_advanced(X_raw, sf=256, nperseg=512)
    y = labels_to_3class(y10)

    # Subject-wise split
    X_train, X_val, X_test, y_train, y_val, y_test = subject_wise_split(
        X_feats, y, subjects, test_size=0.2, val_size=0.1
    )

    # Normalize using training scaler
    X_train, X_val, X_test, scaler = normalize_dataset(X_train, X_val, X_test)

    return X_test, y_test, scaler


def evaluate_keras(model_path):
    print(f"\n📌 Evaluating Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    X_test, y_test, _ = load_test_data()

    preds = np.argmax(model.predict(X_test), axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, digits=4))

    plot_confusion_matrix(y_test, preds, name=f"{os.path.basename(model_path)}_cm.png")


def evaluate_xgb(model_path):
    print(f"\n📌 Evaluating XGBoost model: {model_path}")

    import xgboost as xgb

    X_test, y_test, scaler = load_test_data()

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    preds = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, digits=4))

    plot_confusion_matrix(y_test, preds, name=f"{os.path.basename(model_path)}_cm.png")


if __name__ == "__main__":

    print("\n=============== EVALUATION MENU ===============")
    print("1. MLP")
    print("2. LSTM")
    print("3. CNN")
    print("4. BiLSTM")
    print("5. XGBoost")
    print("===============================================")

    choice = input("Enter model number to evaluate: ")

    if choice == "1":
        evaluate_keras("models/mlp_best.h5")
    elif choice == "2":
        evaluate_keras("models/lstm_best.h5")
    elif choice == "3":
        evaluate_keras("models/cnn_best.h5")
    elif choice == "4":
        evaluate_keras("models/bilstm_best.h5")
    elif choice == "5":
        evaluate_xgb("models/xgb_model.json")
    else:
        print("❌ Invalid choice!")
