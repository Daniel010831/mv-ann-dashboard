# src/reporting_preprocessing.py
# PURPOSE:
# Clean reporting-period data from SRI and enforce
# physically correct 30-minute energy & power values.
# Handles short gaps via interpolation and long outages via exclusion.
# Output is safe for GRU prediction + dashboard visualization (IPMVP Option C).

import pandas as pd
import numpy as np
import os

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_REPORTING_CSV = os.path.join(DATA_DIR, "ReportingPeriodData.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "Reporting_Clean_30min.csv")

# -------------------------------------------------
# ENGINEERING CONSTANTS
# -------------------------------------------------
INTERVAL_HOURS = 0.5

MAX_KW_30MIN = 6000
MAX_KWH_30MIN = MAX_KW_30MIN * INTERVAL_HOURS

MAX_INTERPOLATE_INTERVALS = 4     # ≤ 2 hours
OUTAGE_THRESHOLD = 48             # ≥ 24 hours

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("🔹 Loading raw reporting-period data...")
    df = pd.read_csv(RAW_REPORTING_CSV)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")

    # -------------------------------------------------
    # STEP 1: Keep relevant columns
    # -------------------------------------------------
    df = df[["import_energy", "self_consume"]].copy()

    # -------------------------------------------------
    # STEP 2: Convert Wh → kWh (interval)
    # -------------------------------------------------
    df["Energy_raw"] = (df["import_energy"] + df["self_consume"]) / 1000.0

    # -------------------------------------------------
    # STEP 3: Enforce 30-min timeline
    # -------------------------------------------------
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="30min"
    )
    df = df.reindex(full_index)

    # -------------------------------------------------
    # STEP 4: Initial missing flag
    # -------------------------------------------------
    df["Missing Flag"] = df["Energy_raw"].isna().astype(int)

    # -------------------------------------------------
    # STEP 5: Physical validation (CRITICAL FIX)
    # -------------------------------------------------
    df["Energy_Rejected_Flag"] = (
        (df["Energy_raw"] < 0) |
        (df["Energy_raw"] > MAX_KWH_30MIN)
    ).astype(int)

    df.loc[df["Energy_Rejected_Flag"] == 1, "Energy_raw"] = np.nan

    # -------------------------------------------------
    # STEP 6: Identify continuous NaN runs (ROBUST)
    # -------------------------------------------------
    is_nan = df["Energy_raw"].isna()

    nan_groups = (
        is_nan.ne(is_nan.shift())
        .cumsum()
    )

    df["NaN_Run_Length"] = (
        is_nan.groupby(nan_groups).transform("sum")
    )

    df["Outage Flag"] = (
        (is_nan) &
        (df["NaN_Run_Length"] >= OUTAGE_THRESHOLD)
    )

    # -------------------------------------------------
    # STEP 7: Interpolate short gaps only
    # -------------------------------------------------
    df["Energy_interp"] = df["Energy_raw"]

    short_gap_mask = (~df["Outage Flag"]) & (df["Energy_raw"].isna())

    df.loc[short_gap_mask, "Energy_interp"] = (
        df["Energy_interp"]
        .interpolate(
            method="time",
            limit=MAX_INTERPOLATE_INTERVALS,
            limit_direction="both"
        )
    )

    # -------------------------------------------------
    # STEP 8: Final energy selection
    # -------------------------------------------------
    df["Energy (kWh)"] = df["Energy_interp"]

    df["Interpolated Flag"] = (
        df["Energy (kWh)"].notna() &
        df["Energy_raw"].isna()
    ).astype(int)

    # -------------------------------------------------
    # STEP 9: Power calculation
    # -------------------------------------------------
    df["Power (kW)"] = df["Energy (kWh)"] / INTERVAL_HOURS

    # -------------------------------------------------
    # STEP 10: Valid data KPI
    # -------------------------------------------------
    df["Valid Data Flag"] = (
        df["Energy (kWh)"].notna() &
        ~df["Outage Flag"]
    ).astype(int)

    # -------------------------------------------------
    # STEP 11: Final dataset
    # -------------------------------------------------
    df_final = df[[
        "import_energy",
        "self_consume",
        "Energy (kWh)",
        "Power (kW)",
        "Energy_Rejected_Flag",
        "Interpolated Flag",
        "Outage Flag",
        "Valid Data Flag"
    ]].reset_index().rename(columns={"index": "DateTime"})

    df_final.to_csv(OUTPUT_CSV, index=False)

    print("✅ Clean reporting dataset saved:")
    print(OUTPUT_CSV)

    availability = 100 * df_final["Valid Data Flag"].mean()
    print(f"📊 Data availability: {availability:.2f}%")

    print(df_final.head())
    print(df_final.tail())


if __name__ == "__main__":
    main()
