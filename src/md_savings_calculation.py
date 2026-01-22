import pandas as pd
import os

# -------------------------------------------------
# PATH CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "Reporting_AdjustedBaseline_GRU.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_MD_Savings.csv")

# TNB MV charges (RM/kW/month)
MD_CAPACITY_CHARGE = 29.43
MD_NETWORK_CHARGE = 59.84
TOTAL_MD_CHARGE = MD_CAPACITY_CHARGE + MD_NETWORK_CHARGE  # 89.27 RM/kW

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    print("🔹 Loading adjusted baseline data...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").set_index("DateTime")

    # -------------------------------------------------
    # Peak-hour filter (08:00 – 22:00)
    # -------------------------------------------------
    df_peak = df.between_time("08:00", "22:00")

    # -------------------------------------------------
    # Monthly MD calculation
    # -------------------------------------------------
    df_peak["Month"] = df_peak.index.to_period("M")

    monthly_md = df_peak.groupby("Month").agg(
        Actual_MD_kW=("Actual Power (kW)", "max"),
        Baseline_MD_kW=("Adjusted Baseline Power (kW)", "max"),
    )

    # -------------------------------------------------
    # MD Savings
    # -------------------------------------------------
    monthly_md["MD Savings (kW)"] = (
        monthly_md["Baseline_MD_kW"] - monthly_md["Actual_MD_kW"]
    )

    monthly_md["MD Cost Savings (RM)"] = (
        monthly_md["MD Savings (kW)"] * TOTAL_MD_CHARGE
    )

    monthly_md = monthly_md.reset_index()
    monthly_md["Month"] = monthly_md["Month"].astype(str)

    # -------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------
    monthly_md.to_csv(OUTPUT_CSV, index=False)

    print("✅ Monthly MD savings saved to:", OUTPUT_CSV)
    print(monthly_md.head())
    print(monthly_md.tail())


if __name__ == "__main__":
    main()
