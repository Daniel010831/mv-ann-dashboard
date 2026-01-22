# src/cost_savings_calculation.py
"""
Cost savings calculation based on TNB Commercial Tariff C1 (Flat Rate).

Reference:
- TNB Commercial Tariff C1
- Energy charge: 36.50 sen/kWh = RM 0.365 / kWh

Input:
- Reporting_Savings_GRU.csv

Output:
- Reporting_CostSavings_C1.csv
"""

import os
import pandas as pd

# -------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "Reporting_Savings_GRU.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_CostSavings_C1.csv")

# -------------------------------------------------
# TNB TARIFF CONFIGURATION
# -------------------------------------------------
TNB_C1_RATE_RM_PER_KWH = 0.365  # RM/kWh

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading savings data...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    print("🔹 Applying TNB C1 tariff (RM 0.365 / kWh)...")

    # -------------------------------------------------
    # Cost savings calculation
    # -------------------------------------------------
    df["Cost Savings (RM)"] = df["Savings Energy (kWh)"] * TNB_C1_RATE_RM_PER_KWH

    # Cumulative cost savings
    df["Cumulative Cost Savings (RM)"] = df["Cost Savings (RM)"].cumsum()

    # Optional: flag negative cost savings
    df["Negative Cost Savings Flag"] = (df["Cost Savings (RM)"] < 0).astype(int)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    df.to_csv(OUTPUT_CSV)

    print(f"✅ Cost savings file saved to: {OUTPUT_CSV}")
    print("\n🔍 Preview (head):")
    print(df.head())
    print("\n🔍 Preview (tail):")
    print(df.tail())


if __name__ == "__main__":
    main()
