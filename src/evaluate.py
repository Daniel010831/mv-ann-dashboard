# src/evaluate.py
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from data_preproc import (
    load_and_prepare,
    train_val_test_split_by_dates,
    transform_features,
    create_sequence_windows,
    DEFAULT_FEATURES,
    TARGET_COL
)
from utils import compute_metrics

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
# ---------------------------------------------------------


def eval_ann():
    print("\n==============================")
    print(" Evaluating ANN Model")
    print("==============================")

    # Load dataframe
    df = load_and_prepare()
    train, val, test = train_val_test_split_by_dates(df)

    # Load scalers
    x_scaler = joblib.load(os.path.join(MODEL_DIR, "x_scaler.save"))
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.save"))

    # Transform test set
    X_test, y_test = transform_features(test, x_scaler, y_scaler)

    # Load model
    model = load_model(os.path.join(MODEL_DIR, "ann_best.h5"), compile=False)
    y_pred = model.predict(X_test)

    # Inverse transform
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    # Print metrics
    print("ANN metrics:", compute_metrics(y_test_inv, y_pred_inv))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(test.index, y_test_inv, label="Actual")
    plt.plot(test.index, y_pred_inv, label="ANN Prediction", alpha=0.7)
    plt.title("ANN — Actual vs Predicted")
    plt.legend()
    plt.show()

def eval_seq():
    print("\n==========================================")
    print(" Evaluating LSTM and GRU Sequence Models")
    print("==========================================")

    df = load_and_prepare()
    train, val, test = train_val_test_split_by_dates(df)

    # Load scalers
    x_scaler = joblib.load(os.path.join(MODEL_DIR, "x_scaler.save"))
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.save"))

    # Scale full DF
    full_features = x_scaler.transform(df[DEFAULT_FEATURES])
    scaled_target = y_scaler.transform(df[[TARGET_COL]])

    df_scaled = df.copy()
    df_scaled[DEFAULT_FEATURES] = full_features
    df_scaled[TARGET_COL] = scaled_target

    SEQ_LEN = 48

    # Create test windows
    X_test_seq, y_test_seq = create_sequence_windows(
        df_scaled.loc[test.index.min():test.index.max()],
        seq_len=SEQ_LEN
    )

    # Load models
    lstm = load_model(os.path.join(MODEL_DIR, "lstm_best.h5"), compile=False)
    gru = load_model(os.path.join(MODEL_DIR, "gru_best.h5"), compile=False)

    # Predict
    y_lstm = lstm.predict(X_test_seq)
    y_gru = gru.predict(X_test_seq)

    # Inverse transform
    y_test_inv = y_scaler.inverse_transform(y_test_seq)
    y_lstm_inv = y_scaler.inverse_transform(y_lstm)
    y_gru_inv = y_scaler.inverse_transform(y_gru)

    # Metrics
    print("LSTM metrics:", compute_metrics(y_test_inv, y_lstm_inv))
    print("GRU metrics:", compute_metrics(y_test_inv, y_gru_inv))

    # Plot comparison
    plt.figure(figsize=(12, 4))
    plt.plot(test.index[SEQ_LEN:], y_test_inv, label="Actual")
    plt.plot(test.index[SEQ_LEN:], y_lstm_inv, label="LSTM", alpha=0.7)
    plt.plot(test.index[SEQ_LEN:], y_gru_inv, label="GRU", alpha=0.7)
    plt.title("Sequence Models — LSTM vs GRU vs Actual")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Running eval_ann()...")
    eval_ann()

    print("Closing ANN plots...")
    plt.close('all')

    print("Running eval_seq()...")
    eval_seq()
