# src/train_xgb.py
import os
import argparse
import numpy as np
import joblib
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

from utils import (
    load_mat_data,
    extract_features_advanced,
    labels_to_3class,
    subject_wise_split,
    normalize_dataset
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', type=str, default='data/Data/filtered_data')
    p.add_argument('--model_out', type=str, default=os.path.join(MODEL_DIR, 'xgb_model.json'))
    p.add_argument('--n_estimators', type=int, default=300)
    p.add_argument('--max_depth', type=int, default=6)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    print("Loading raw data...")
    X_raw, y10, subjects, tasks = load_mat_data(args.data_folder)
    print(f"Loaded {len(X_raw)} samples; example shape: {X_raw[0].shape}")

    print("Extracting advanced features (reduced)...")
    X_feats = extract_features_advanced(X_raw, sf=256, nperseg=512)
    print("Feature shape:", X_feats.shape)

    # convert 10-class labels → 3-class (low/medium/high)
    y = labels_to_3class(y10)

    # subject wise split
    X_train, X_val, X_test, y_train, y_val, y_test = subject_wise_split(
        X_feats, y, subjects,
        test_size=0.2,
        val_size=0.1,
        random_state=args.seed
    )

    print("\nApplying oversampling to handle minority classes...")
    ros = RandomOverSampler(random_state=args.seed)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print("New balanced class counts:", np.unique(y_train, return_counts=True))

    # Normalize
    X_train, X_val, X_test, scaler = normalize_dataset(X_train, X_val, X_test)

    # Build model (legacy-compatible)
    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        objective='multi:softprob',
        num_class=3,
        seed=args.seed
    )

    # ---- LEGACY FIT (no eval parameters allowed) ----
    print("\nTraining XGBoost (legacy mode: no eval, no early stopping)...")
    clf.fit(X_train, y_train)

    # Save model
    print("\nSaving model to", args.model_out)
    clf.save_model(args.model_out)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'xgb_scaler.pkl'))

    # Test
    print("\n=== TEST CLASSIFICATION REPORT ===")
    preds = clf.predict(X_test)

    print(classification_report(
        y_test, preds, digits=4,
        target_names=['low', 'medium', 'high']
    ))

    print("\nDone.")

if __name__ == '__main__':
    main()
