# src/reporting_gru_preprocessor.py
# PURPOSE:
# Build GRU-ready reporting dataset WITHOUT dropping any 30-min timestamps

import pandas as pd
import numpy as np
import os

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BASELINE_CSV = os.path.join(DATA_DIR, "PenangBaselineData.csv")
REPORTING_CLEAN_CSV = os.path.join(DATA_DIR, "Reporting_Clean_30min.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_GRU_Ready.csv")

SEQ_LEN = 48
WEEK_LAG = 48 * 7

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def add_time_features(df):
    df["Day"] = df.index.dayofweek
    df["Hour"] = df.index.hour
    df["Time"] = df.index.hour * 2 + (df.index.minute // 30)
    return df


def add_academic_flags(df):
    semester_break_ranges = [
        ("2024-02-24", "2024-03-23"),
        ("2024-08-11", "2024-09-28"),
        ("2025-02-24", "2025-03-23"),
        ("2025-08-11", "2025-09-28"),
    ]

    df["Semester Break"] = 0
    for start, end in semester_break_ranges:
        df.loc[start:end, "Semester Break"] = 1

    df["Lecture/Non-lecture"] = np.where(df["Semester Break"] == 1, 0, 1)
    df["Semester : Lecture/Office"] = np.where(df["Semester Break"] == 1, 0, 1)
    df["Public Holiday"] = 0

    return df


def add_lag_features(df):
    df["Day Lagged Load"] = df["Load Consumption (kW)"].shift(SEQ_LEN)
    df["Week Lagged Load"] = df["Load Consumption (kW)"].shift(WEEK_LAG)
    return df


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("🔹 Loading baseline data...")
    baseline = pd.read_csv(BASELINE_CSV, parse_dates=["DateTime"])
    baseline = baseline.sort_values("DateTime").set_index("DateTime")

    # Keep last 7 days only (for lag continuity)
    baseline_tail = baseline.tail(WEEK_LAG)

    print("🔹 Loading clean reporting-period data...")
    rpt = pd.read_csv(REPORTING_CLEAN_CSV, parse_dates=["DateTime"])
    rpt = rpt.sort_values("DateTime").set_index("DateTime")

    # Rename for model consistency
    rpt = rpt.rename(columns={"Power (kW)": "Load Consumption (kW)"})
    
    # -------------------------------------------------
    # ALIGN WITH CLEAN REPORTING DATA (NO DROPPING)
    # -------------------------------------------------

    # Ensure numeric stability
    rpt["Load Consumption (kW)"] = rpt["Load Consumption (kW)"].clip(lower=0)

    # Mark outage rows (keep timestamps!)
    if "Outage Flag" in rpt.columns:
        rpt.loc[rpt["Outage Flag"] == 1, "Load Consumption (kW)"] = np.nan

    # -------------------------------------------------
    # COMBINE (NO DROPPING)
    # -------------------------------------------------
    combined = pd.concat([baseline_tail, rpt], axis=0)
    combined = combined.sort_index()

    # -------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------
    combined = add_time_features(combined)
    combined = add_academic_flags(combined)
    combined = add_lag_features(combined)

    # -------------------------------------------------
    # IMPORTANT FIX:
    # Do NOT drop rows — only mask lag-invalid rows later
    # -------------------------------------------------

    # Keep only reporting-period timestamps
    reporting_ready = combined.loc[rpt.index.min():]

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    reporting_ready.to_csv(OUTPUT_CSV)

    print("✅ GRU-ready reporting dataset saved:")
    print(OUTPUT_CSV)
    print(reporting_ready.head(5))
    print(reporting_ready.tail(5))


if __name__ == "__main__":
    main()
