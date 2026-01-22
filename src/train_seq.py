# src/train_seq.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_preproc import (
    load_and_prepare,
    train_val_test_split_by_dates,
    fit_scalers,
    create_sequence_windows,
    DEFAULT_FEATURES,
    TARGET_COL
)
from models_seq import build_lstm, build_gru
from utils import compute_metrics

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

SEQ_LEN = 48  # 1 day sequence window

def prepare_seq_data():
    df = load_and_prepare()
    train, val, test = train_val_test_split_by_dates(df)

    # Fit scalers using training data
    x_scaler, y_scaler = fit_scalers(train)

    # Scale entire dataframe so sliding windows are consistent
    full_features = x_scaler.transform(df[DEFAULT_FEATURES])
    scaled_target = y_scaler.transform(df[[TARGET_COL]])

    df_scaled = df.copy()
    df_scaled[DEFAULT_FEATURES] = full_features
    df_scaled[TARGET_COL] = scaled_target

    # Create sliding windows for each split
    X_train, y_train = create_sequence_windows(
        df_scaled.loc[train.index.min() : train.index.max()], SEQ_LEN
    )

    X_val, y_val = create_sequence_windows(
        df_scaled.loc[val.index.min() : val.index.max()], SEQ_LEN
    )

    X_test, y_test = create_sequence_windows(
        df_scaled.loc[test.index.min() : test.index.max()], SEQ_LEN
    )

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        x_scaler, y_scaler,
        test
    )


def train_models():
    (
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    x_scaler, y_scaler,
    test_df
    ) = prepare_seq_data()

    n_features = X_train.shape[2]

    # -------- LSTM --------
    lstm = build_lstm(seq_len=SEQ_LEN, n_features=n_features)
    lstm_ckpt = os.path.join(MODEL_DIR, "lstm_best.h5")

    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ModelCheckpoint(lstm_ckpt, monitor="val_loss", save_best_only=True)
        ]
    )

    # -------- GRU --------
    gru = build_gru(seq_len=SEQ_LEN, n_features=n_features)
    gru_ckpt = os.path.join(MODEL_DIR, "gru_best.h5")

    gru.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ModelCheckpoint(gru_ckpt, monitor="val_loss", save_best_only=True)
        ]
    )

    # Evaluate
    y_lstm = lstm.predict(X_test)
    y_gru = gru.predict(X_test)

    y_test_inv = y_scaler.inverse_transform(y_test)
    y_lstm_inv = y_scaler.inverse_transform(y_lstm)
    y_gru_inv = y_scaler.inverse_transform(y_gru)

    print("LSTM Metrics:", compute_metrics(y_test_inv, y_lstm_inv))
    print("GRU Metrics:", compute_metrics(y_test_inv, y_gru_inv))

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(test_df.index[SEQ_LEN:], y_test_inv, label="Actual")
    plt.plot(test_df.index[SEQ_LEN:], y_lstm_inv, label="LSTM")
    plt.plot(test_df.index[SEQ_LEN:], y_gru_inv, label="GRU")
    plt.legend()
    plt.title("Sequence Models (LSTM vs GRU) — Test Set")
    plt.show()

if __name__ == "__main__":
    train_models()