# UNIVERSAL TRAIN SCRIPT
# src/train.py

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

from utils import (
    load_mat_data,
    extract_features_advanced,
    labels_to_3class,
    subject_wise_split,
    normalize_dataset
)
from models import (
    build_mlp,
    build_lstm,
    build_cnn1d,
    build_bilstm,
    build_transformer_encoder
)
from metrics import compile_model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='data/Data/filtered_data')
    parser.add_argument('--model', type=str,
                        choices=['mlp', 'lstm', 'cnn', 'bilstm', 'transformer'],
                        required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default=None)
    return parser.parse_args()


def build_selected_model(model_name, input_shape, n_classes=3):
    """Return the selected model."""
    if model_name == 'mlp':
        return build_mlp(input_shape, n_classes=n_classes, hidden=[128, 64], dropout=0.4)

    elif model_name == 'lstm':
        return build_lstm(input_shape, n_classes=n_classes)

    elif model_name == 'cnn':
        return build_cnn1d(input_shape, n_classes=n_classes)

    elif model_name == 'bilstm':
        return build_bilstm(input_shape, n_classes=n_classes)

    elif model_name == 'transformer':
        return build_transformer_encoder(input_shape, n_classes=n_classes)

    else:
        raise ValueError("Unknown model type")


def main():
    args = parse_args()

    print(f"\n🔹 Selected Model: {args.model}")
    print("🔹 Loading raw data...")
    X_raw, y10, subjects, tasks = load_mat_data(args.data_folder)

    # Extract features for MLP
    if args.model == 'mlp':
        print("\n📌 Using Advanced Feature Extraction for MLP...")
        X_feats = extract_features_advanced(X_raw, sf=256, nperseg=512)
        print("Feature shape:", X_feats.shape)

        y = labels_to_3class(y10)
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test
        ) = subject_wise_split(X_feats, y, subjects, test_size=0.2, val_size=0.1)

        X_train, X_val, X_test, scaler = normalize_dataset(X_train, X_val, X_test)
        input_shape = X_train.shape[1]

    else:
        # For deep models → raw EEG sequences
        print("\n📌 Preparing raw EEG for deep sequence models...")
        X = np.array([x for x in X_raw], dtype=object)
        max_len = max([arr.shape[1] for arr in X])
        X_pad = np.zeros((len(X), X_raw[0].shape[0], max_len))

        for i, arr in enumerate(X_raw):
            X_pad[i, :, :arr.shape[1]] = arr

        # shape → (samples, timesteps, channels)
        X_pad = np.transpose(X_pad, (0, 2, 1))

        y = labels_to_3class(y10)

        print("Raw EEG padded shape:", X_pad.shape)

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test
        ) = subject_wise_split(X_pad, y, subjects, test_size=0.2, val_size=0.1)

        # Normalize per-channel (not flatten)
        X_train = X_train.astype('float32')
        X_val   = X_val.astype('float32')
        X_test  = X_test.astype('float32')

        input_shape = X_train.shape[1:]

    print("\nBuilding model...")
    model = build_selected_model(args.model, input_shape, n_classes=3)
    model = compile_model(model, lr=args.lr)
    model.summary()

    print("\nComputing class weights...")
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}

    ckpt = args.save if args.save else os.path.join(MODEL_DIR, f"{args.model}_best.h5")

    callbacks = [
        ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print("\n🚀 Training started...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    np.save(f"{args.model}_history.npy", history.history)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n✅ Test result — loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    final_path = os.path.join(MODEL_DIR, f"{args.model}_final.h5")
    model.save(final_path)
    print("💾 Saved final model to", final_path)


if __name__ == "__main__":
    main()
