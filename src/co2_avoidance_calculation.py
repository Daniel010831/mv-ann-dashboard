# src/co2_avoidance_calculation.py
"""
CO₂ avoidance calculation based on energy savings.

Reference:
- Malaysia grid emission factor ≈ 0.584 kgCO2 / kWh

Input:
- Reporting_Savings_GRU.csv

Output:
- Reporting_CO2_Avoidance_GRU.csv
"""

import os
import pandas as pd

# -------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "Reporting_Savings_GRU.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_CO2_Avoidance_GRU.csv")

# -------------------------------------------------
# EMISSION FACTOR
# -------------------------------------------------
GRID_EMISSION_FACTOR = 0.774  # kgCO2 per kWh (Malaysia)

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading savings data...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    print("🔹 Calculating CO₂ avoidance...")

    # Interval CO2 avoided (kg)
    df["CO2 Avoided (kg)"] = (
        df["Savings Energy (kWh)"] * GRID_EMISSION_FACTOR
    )

    # Cumulative CO2 avoided (kg)
    df["Cumulative CO2 Avoided (kg)"] = (
        df["CO2 Avoided (kg)"].cumsum()
    )

    # Optional: convert to tonnes
    df["Cumulative CO2 Avoided (tonnes)"] = (
        df["Cumulative CO2 Avoided (kg)"] / 1000
    )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    df.to_csv(OUTPUT_CSV)

    print(f"✅ CO₂ avoidance file saved to: {OUTPUT_CSV}")
    print("\n🔍 Preview (head):")
    print(df.head())
    print("\n🔍 Preview (tail):")
    print(df.tail())


if __name__ == "__main__":
    main()
