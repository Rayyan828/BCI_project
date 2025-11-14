# metrics.py
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compile_model(model, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, labels, out_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def classification_report_to_string(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)
