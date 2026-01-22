# src/savings_calculation.py

import os
import pandas as pd
import numpy as np

# -------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "Reporting_AdjustedBaseline_GRU.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_Savings_GRU.csv")

INTERVAL_HOURS = 0.5  # 30-minute data

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading adjusted baseline results...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    # -------------------------------------------------
    # Savings calculations
    # -------------------------------------------------
    print("🔹 Calculating savings...")

    # Instantaneous power savings (kW)
    df["Savings Power (kW)"] = (
        df["Adjusted Baseline Power (kW)"] - df["Actual Power (kW)"]
    )

    # Interval energy savings (kWh)
    df["Savings Energy (kWh)"] = df["Savings Power (kW)"] * INTERVAL_HOURS

    # Cumulative energy savings
    df["Cumulative Savings Energy (kWh)"] = df["Savings Energy (kWh)"].cumsum()

    # -------------------------------------------------
    # Optional: Flag negative savings
    # -------------------------------------------------
    df["Negative Savings Flag"] = (df["Savings Energy (kWh)"] < 0).astype(int)

    # -------------------------------------------------
    # Save output
    # -------------------------------------------------
    df.to_csv(OUTPUT_CSV)

    print(f"✅ Savings file saved to: {OUTPUT_CSV}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
