# src/cost_savings_tou.py
"""
Time-of-Use (TOU) cost savings calculation
based on TNB Commercial C1 MV ETOU (energy charge only).

Input:
- Reporting_Savings_GRU.csv

Output:
- Reporting_CostSavings_TOU.csv
"""

import os
import pandas as pd

# -------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "Reporting_Savings_GRU.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_CostSavings_TOU.csv")

# -------------------------------------------------
# TNB TOU ENERGY RATES (RM/kWh)
# -------------------------------------------------
TOU_RATES = {
    "peak": 0.584,
    "mid": 0.357,
    "offpeak": 0.281,
}

# -------------------------------------------------
# TOU PERIOD CLASSIFICATION
# -------------------------------------------------
def classify_tou_period(timestamp):
    hour = timestamp.hour

    if 10 <= hour < 14:
        return "peak"
    elif 8 <= hour < 10 or 14 <= hour < 22:
        return "mid"
    else:
        return "offpeak"

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading savings data...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    print("🔹 Classifying TOU periods...")
    df["TOU Period"] = df.index.map(classify_tou_period)

    print("🔹 Applying TOU energy rates...")
    df["TOU Rate (RM/kWh)"] = df["TOU Period"].map(TOU_RATES)

    # -------------------------------------------------
    # TOU cost savings calculation
    # -------------------------------------------------
    df["TOU Cost Savings (RM)"] = (
        df["Savings Energy (kWh)"] * df["TOU Rate (RM/kWh)"]
    )

    df["Cumulative TOU Cost Savings (RM)"] = (
        df["TOU Cost Savings (RM)"].cumsum()
    )

    # Optional: flag negative TOU savings
    df["Negative TOU Savings Flag"] = (
        df["TOU Cost Savings (RM)"] < 0
    ).astype(int)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    df.to_csv(OUTPUT_CSV)

    print(f"✅ TOU cost savings file saved to: {OUTPUT_CSV}")
    print("\n🔍 Preview (head):")
    print(df.head())
    print("\n🔍 Preview (tail):")
    print(df.tail())


if __name__ == "__main__":
    main()
