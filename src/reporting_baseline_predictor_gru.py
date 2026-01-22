# src/reporting_baseline_predictor_gru.py
"""
GRU Adjusted Baseline Predictor (Reporting Period)
- RUN THIS: python src/reporting_baseline_predictor_gru.py --mode simulate --csv data/Reporting_GRU_Ready.csv
- Loads trained GRU model and saved scalers
- Uses Reporting_GRU_Ready.csv (already feature-engineered)
- Generates adjusted baseline power (kW)
- Saves results for M&V savings calculation
"""

import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_preproc import create_sequence_windows

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

INPUT_CSV = os.path.join(DATA_DIR, "Reporting_GRU_Ready.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_AdjustedBaseline_GRU.csv")

GRU_MODEL_PATH = os.path.join(MODEL_DIR, "gru_best.h5")
X_SCALER_PATH = os.path.join(MODEL_DIR, "x_scaler.save")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.save")

SEQ_LEN = 48  # must match training

# -------------------------------------------------
# FEATURE CONFIG (MUST MATCH TRAINING)
# -------------------------------------------------
FEATURE_COLS = [
    "Day",
    "Hour",
    "Lecture/Non-lecture",
    "Public Holiday",
    "Semester Break",
    "Semester : Lecture/Office",
    "Day Lagged Load",
    "Week Lagged Load",
    "Time"
]

TARGET_COL = "Load Consumption (kW)"

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading GRU model and scalers...")
    gru = load_model(GRU_MODEL_PATH, compile=False)
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    print("🔹 Loading GRU-ready reporting dataset...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    # -------------------------------------------------
    # Scale features & target (same as training)
    # -------------------------------------------------
    X_raw = df[FEATURE_COLS].values.astype(float)
    y_raw = df[[TARGET_COL]].values.astype(float)

    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw)

    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = X_scaled
    df_scaled[TARGET_COL] = y_scaled

    # -------------------------------------------------
    # Create GRU windows
    # -------------------------------------------------
    print("🔹 Creating GRU sequence windows...")
    X_seq, y_seq = create_sequence_windows(df_scaled, seq_len=SEQ_LEN)

    timestamps = df.index[SEQ_LEN:]

    # -------------------------------------------------
    # Predict adjusted baseline
    # -------------------------------------------------
    print("🔹 Predicting adjusted baseline (GRU)...")
    y_pred_scaled = gru.predict(X_seq, verbose=1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    y_actual = y_scaler.inverse_transform(y_seq.reshape(-1, 1))

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    results = pd.DataFrame(
        {
            "Actual Power (kW)": y_actual.flatten(),
            "Adjusted Baseline Power (kW)": y_pred.flatten(),
        },
        index=timestamps,
    )

    results.index.name = "DateTime"
    results.to_csv(OUTPUT_CSV)

    print("✅ Adjusted baseline saved to:", OUTPUT_CSV)
    print(results.head())
    print(results.tail())


if __name__ == "__main__":
    main()
