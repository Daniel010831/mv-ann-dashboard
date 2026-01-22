# src/compare_c1_vs_tou.py
"""
Compare cumulative cost savings between:
- TNB C1 (flat tariff)
- TNB TOU (time-of-use tariff)

Outputs:
1) Comparison plot
2) CSV for dashboard:
   data/Reporting_Cost_C1_vs_TOU.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

C1_CSV = os.path.join(DATA_DIR, "Reporting_CostSavings_C1.csv")
TOU_CSV = os.path.join(DATA_DIR, "Reporting_CostSavings_TOU.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_Cost_C1_vs_TOU.csv")

# -------------------------------------------------
# HELPER
# -------------------------------------------------
def find_column(cols, keyword):
    for c in cols:
        if keyword.lower() in c.lower():
            return c
    raise KeyError(f"Column containing '{keyword}' not found")

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading C1 and TOU savings data...")

    df_c1 = pd.read_csv(C1_CSV, parse_dates=["DateTime"])
    df_tou = pd.read_csv(TOU_CSV, parse_dates=["DateTime"])

    df_c1 = df_c1.sort_values("DateTime").set_index("DateTime")
    df_tou = df_tou.sort_values("DateTime").set_index("DateTime")

    # Detect correct columns automatically
    c1_cum_col = find_column(df_c1.columns, "Cumulative Cost")
    tou_cum_col = find_column(df_tou.columns, "Cumulative TOU")

    # Align timestamps
    common_index = df_c1.index.intersection(df_tou.index)
    df_c1 = df_c1.loc[common_index]
    df_tou = df_tou.loc[common_index]

    # -------------------------------------------------
    # SAVE CSV FOR DASHBOARD
    # -------------------------------------------------
    df_compare = pd.DataFrame({
        "DateTime": common_index,
        "C1 Cumulative Cost Savings (RM)": df_c1[c1_cum_col].values,
        "TOU Cumulative Cost Savings (RM)": df_tou[tou_cum_col].values
    }).set_index("DateTime")

    df_compare.to_csv(OUTPUT_CSV)
    print(f"✅ Comparison CSV saved to: {OUTPUT_CSV}")

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(
        df_compare.index,
        df_compare["C1 Cumulative Cost Savings (RM)"],
        label="C1 Tariff (Flat)",
        linewidth=2
    )
    plt.plot(
        df_compare.index,
        df_compare["TOU Cumulative Cost Savings (RM)"],
        label="TOU Tariff",
        linewidth=2
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Cost Savings (RM)")
    plt.title("Cumulative Cost Savings Comparison: C1 vs TOU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("✅ Comparison plot displayed successfully.")

if __name__ == "__main__":
    main()
