# src/train_ann.py
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_preproc import (
    load_and_prepare,
    train_val_test_split_by_dates,
    fit_scalers,
    transform_features,
    DEFAULT_FEATURES,
    TARGET_COL
)
from models_ann import build_dense_ann
from utils import compute_metrics
# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

def prepare_data():
    df = load_and_prepare()
    train, val, test = train_val_test_split_by_dates(df)
    x_scaler, y_scaler = fit_scalers(train)
    X_train, y_train = transform_features(train, x_scaler, y_scaler)
    X_val, y_val = transform_features(val, x_scaler, y_scaler)
    X_test, y_test = transform_features(test, x_scaler, y_scaler)
    return X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler, test

def train():
    (X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler, test_df) = prepare_data()
    model = build_dense_ann(input_dim=X_train.shape[1])
    ckpt_path = os.path.join(MODEL_DIR, "ann_best.h5")
    mc = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=200, batch_size=64, callbacks=[es, mc])
    model.save(os.path.join(MODEL_DIR, "ann_final.h5"))

    # evaluate
    y_pred = model.predict(X_test)
    # invert scaling
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.save"))
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    metrics = compute_metrics(y_test_inv, y_pred_inv)
    print("ANN Test metrics:", metrics)

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(test_df.index, y_test_inv, label="Actual")
    plt.plot(test_df.index, y_pred_inv, label="ANN Prediction")
    plt.legend()
    plt.title("ANN Actual vs Predicted (Test Set)")
    plt.show()

if __name__ == "__main__":
    train()
