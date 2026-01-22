# src/data_preproc.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------------------------------------------
# PATH SETUP (works for Run Button & Terminal)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_FEATURES = [
    "Day",
    "Hour",
    "Time",
    "Lecture/Non-lecture",
    "Public Holiday",
    "Semester Break",
    "Semester : Lecture/Office",
    "Day Lagged Load",
    "Week Lagged Load"
]
TARGET_COL = "Load Consumption (kW)"

def load_and_prepare():
    path = os.path.join(DATA_DIR, "PenangBaselineData.csv")
    df = pd.read_csv(path)

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    df = df.set_index("DateTime")

    return df


def train_val_test_split_by_dates(df):
    train = df[: "2024-02-29 23:30:00"].copy()
    val = df["2024-03-01 00:00:00" : "2024-03-31 23:30:00"].copy()
    test = df["2024-04-01 00:00:00" : "2024-04-30 23:30:00"].copy()
    return train, val, test


def fit_scalers(train_df):
    X_train = train_df[DEFAULT_FEATURES].values.astype(float)
    y_train = train_df[[TARGET_COL]].values.astype(float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(X_train)
    y_scaler.fit(y_train)

    joblib.dump(x_scaler, os.path.join(MODEL_DIR, "x_scaler.save"))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, "y_scaler.save"))

    return x_scaler, y_scaler


def transform_features(df, x_scaler, y_scaler):
    X = df[DEFAULT_FEATURES].values.astype(float)
    y = df[[TARGET_COL]].values.astype(float)

    Xs = x_scaler.transform(X)
    ys = y_scaler.transform(y)

    return Xs, ys


def create_sequence_windows(df, seq_len=48):
    feature_cols = DEFAULT_FEATURES + [TARGET_COL]

    arr = df[feature_cols].values.astype(float)
    targets = df[TARGET_COL].values.astype(float)

    X, y = [], []

    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len : i, :])
        y.append(targets[i])

    return np.array(X), np.array(y).reshape(-1, 1)