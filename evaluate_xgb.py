# src/evaluate_xgb.py
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

from utils import load_mat_data, extract_features_advanced, labels_to_3class, subject_wise_split, normalize_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', type=str, default='data/Data/filtered_data')
    p.add_argument('--model_path', type=str, default='models/xgb_model.json')
    p.add_argument('--scaler_path', type=str, default='models/xgb_scaler.pkl')
    return p.parse_args()

def plot_cm(y_true, y_pred, labels, out_path='confusion_xgb.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved", out_path)

def main():
    args = parse_args()
    X_raw, y10, subjects, tasks = load_mat_data(args.data_folder)
    X_feats = extract_features_advanced(X_raw, sf=256, nperseg=512)
    y = labels_to_3class(y10)
    _, _, X_test, _, _, y_test = subject_wise_split(X_feats, y, subjects, test_size=0.2, val_size=0.1, random_state=42)
    scaler = joblib.load(args.scaler_path)
    X_test_s = scaler.transform(X_test)
    clf = xgb.XGBClassifier()
    clf.load_model(args.model_path)
    preds = clf.predict(X_test_s)
    print("Classification report:\n")
    print(classification_report(y_test, preds, digits=4, target_names=['low','medium','high']))
    with open('models/xgb_eval_report.txt', 'w') as f:
        f.write(classification_report(y_test, preds, digits=4, target_names=['low','medium','high']))
    plot_cm(y_test, preds, labels=['low','medium','high'], out_path='models/xgb_confusion.png')

if __name__ == '__main__':
    main()
