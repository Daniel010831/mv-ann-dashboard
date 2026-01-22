# dashboard/finalize_app.py
# RUN -> streamlit run dashboard/finalize_app.py
#git add dashboard/finalize_app.py
#git commit -m "Fix Full M&V timeline index handling and ECM separation"
#git push origin main

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Advanced M&V 2.0 Dashboard (GRU)",
    layout="wide"
)

st.title("Advanced Measurement & Verification (M&V 2.0)")
st.subheader("GRU-Based Adjusted Baseline | UiTM Facility")

st.markdown("""
**Project Owner & Developer (FYP II):**  
Ahmad Daniel bin Mohd Yusop  
Bachelor of Electrical Engineering (Hons)

**Supervisor:**  
Prof. Ir. Dr. Nofri Yenita Dahlan
""")

st.info(
    "Industry-aligned implementation of Advanced M&V (IPMVP Option C) "
    "using machine-learning–based adjusted baselines."
)

# ----------------------------------------------------
# SIDEBAR — METHODOLOGY & DATA QUALITY NOTES
# ----------------------------------------------------
with st.sidebar:
    st.header("📘 Methodology Overview")

    st.markdown("""
### 🔹 M&V Framework
- **IPMVP Option C (Whole Facility)**
- Advanced M&V / M&V 2.0
- Savings evaluated at whole-building level

### 🔹 Baseline & Modelling
- **Baseline Period:** Pre-ECM historical data
- **Model:** Gated Recurrent Unit (GRU)
- Captures temporal, operational, and behavioural patterns
- Adjusted baseline reflects post-ECM operating conditions

### 🔹 Reporting Period
- 30-minute interval resolution
- Measured load compared against GRU-adjusted baseline
- Savings may be positive or negative (IPMVP-compliant)

### 🔹 Carbon Accounting
- Grid emission factor (Peninsular Malaysia)
- **0.774 tCO₂/MWh** (Suruhanjaya Tenaga)

---
### 🔹 Data Quality Indicators (Important)
- **Interpolated Flag**
  - Short data gaps (≤ 2 hours)
  - Filled using time-based interpolation
  - Conservative and IPMVP-safe

- **Outage Flag**
  - Long data gaps (≥ 24 hours)
  - Excluded from savings evaluation
  - Indicates meter outage or data loss

- **Valid Data Flag**
  - Intervals used in savings calculations
  - Excludes outages automatically
""")


# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BASELINE_CSV = os.path.join(DATA_DIR, "PenangBaselineData.csv")
SAVINGS_CSV = os.path.join(DATA_DIR, "Reporting_Savings_GRU.csv")
COST_COMPARE_CSV = os.path.join(DATA_DIR, "Reporting_Cost_C1_vs_TOU.csv")
CO2_CSV = os.path.join(DATA_DIR, "Reporting_CO2_Avoidance_GRU.csv")
MD_CSV = os.path.join(DATA_DIR, "Reporting_MD_Savings.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "Reporting_Clean_30min.csv")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return (
        pd.read_csv(SAVINGS_CSV, parse_dates=["DateTime"]),
        pd.read_csv(COST_COMPARE_CSV, parse_dates=["DateTime"]),
        pd.read_csv(CO2_CSV, parse_dates=["DateTime"]),
        pd.read_csv(MD_CSV),
        pd.read_csv(BASELINE_CSV, parse_dates=["DateTime"]),
        pd.read_csv(CLEAN_CSV, parse_dates=["DateTime"])
    )

df_s, df_cmp, df_co, df_md, df_base, df_clean = load_data()

df_s = df_s.set_index("DateTime").sort_index()
df_cmp = df_cmp.set_index("DateTime").sort_index()
df_co = df_co.set_index("DateTime").sort_index()
df_base = df_base.set_index("DateTime").sort_index()
df_clean = df_clean.set_index("DateTime").sort_index()
df_md["Month"] = pd.to_datetime(df_md["Month"])

# ----------------------------------------------------
# DATE RANGE FILTER (FIXED)
# ----------------------------------------------------
min_date = df_s.index.min().date()
max_date = df_s.index.max().date()

date_range = st.date_input(
    "Select Reporting Period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

df_s_f = df_s.loc[start_dt:end_dt]
df_cmp_f = df_cmp.loc[start_dt:end_dt]
df_co_f = df_co.loc[start_dt:end_dt]
df_clean_f = df_clean.loc[start_dt:end_dt]

df_md_f = df_md[
    (df_md["Month"] >= start_dt) &
    (df_md["Month"] <= end_dt)
]

if df_s_f.empty:
    st.warning("No data available for selected period.")
    st.stop()

# ----------------------------------------------------
# KPI METRICS
# ----------------------------------------------------
st.markdown("### 🔹 Key Performance Indicators")

total_energy_savings = (
    df_s_f["Cumulative Savings Energy (kWh)"].iloc[-1]
    - df_s_f["Cumulative Savings Energy (kWh)"].iloc[0]
)

total_cost_c1 = (
    df_cmp_f["C1 Cumulative Cost Savings (RM)"].iloc[-1]
    - df_cmp_f["C1 Cumulative Cost Savings (RM)"].iloc[0]
)

total_cost_tou = (
    df_cmp_f["TOU Cumulative Cost Savings (RM)"].iloc[-1]
    - df_cmp_f["TOU Cumulative Cost Savings (RM)"].iloc[0]
)

total_co2 = (
    df_co_f["Cumulative CO2 Avoided (tonnes)"].iloc[-1]
    - df_co_f["Cumulative CO2 Avoided (tonnes)"].iloc[0]
)

negative_pct = 100 * df_s_f["Negative Savings Flag"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Energy Savings", f"{total_energy_savings:,.0f} kWh")
c2.metric("Cost Savings (C1)", f"RM {total_cost_c1:,.0f}")
c3.metric("Cost Savings (TOU)", f"RM {total_cost_tou:,.0f}")
c4.metric("CO₂ Avoided", f"{total_co2:,.2f} tCO₂")
c5.metric("Negative Savings", f"{negative_pct:.1f}%")

# ----------------------------------------------------
# DATA QUALITY INDICATORS (CORRECT SOURCE)
# ----------------------------------------------------
st.markdown("### 🔹 Data Quality Indicators")

interp_pct = 100 * df_clean_f["Interpolated Flag"].mean()
outage_pct = 100 * df_clean_f["Outage Flag"].mean()
valid_pct = 100 * df_clean_f["Valid Data Flag"].mean()

q1, q2, q3 = st.columns(3)
q1.metric("Interpolated Intervals", f"{interp_pct:.1f}%")
q2.metric("Outage Intervals", f"{outage_pct:.1f}%")
q3.metric("Valid Intervals", f"{valid_pct:.1f}%")
st.caption(
    "Data quality flags ensure transparency and traceability of savings results. "
    "Interpolated values are limited to short gaps only, while long outages are "
    "excluded in accordance with IPMVP Option C best practices."
)

# ----------------------------------------------------
# PRE-ECM BASELINE
# ----------------------------------------------------
st.markdown("### 🔹 Pre-ECM Baseline Load Profile")
st.line_chart(df_base["Load Consumption (kW)"])

# ----------------------------------------------------
# FULL M&V TIMELINE (BASELINE → ADJUSTED BASELINE → ACTUAL)
# ----------------------------------------------------
st.markdown("### 🔹 Full M&V Timeline")

ECM_DATE = pd.to_datetime("2024-05-01")

# 1️⃣ Build a TRUE full timeline index (baseline + reporting)
timeline_index = (
    df_base.index
    .union(df_s.index)
    .sort_values()
)

timeline_df = pd.DataFrame(index=timeline_index)

# 2️⃣ Assign series explicitly
timeline_df["Baseline Load (kW)"] = df_base["Load Consumption (kW)"]
timeline_df["Adjusted Baseline Power (kW)"] = df_s["Adjusted Baseline Power (kW)"]
timeline_df["Actual Power (kW)"] = df_s["Actual Power (kW)"]

# 3️⃣ Enforce IPMVP Option C logic
# Baseline ONLY before ECM
timeline_df.loc[timeline_df.index >= ECM_DATE, "Baseline Load (kW)"] = None

# Adjusted + Actual ONLY after ECM
timeline_df.loc[timeline_df.index < ECM_DATE, "Adjusted Baseline Power (kW)"] = None
timeline_df.loc[timeline_df.index < ECM_DATE, "Actual Power (kW)"] = None

# ----------------------------------------------------
# 4️⃣ Plot (Streamlit-native, stable, fast)
# ----------------------------------------------------
st.line_chart(
    timeline_df[
        [
            "Baseline Load (kW)",
            "Adjusted Baseline Power (kW)",
            "Actual Power (kW)",
        ]
    ]
)

st.caption(
    "Full Measurement & Verification (M&V) timeline in accordance with "
    "IPMVP Option C. Pre-ECM baseline represents historical operating behaviour. "
    "Post-ECM period shows GRU-adjusted baseline and measured consumption."
)
# ----------------------------------------------------
# MONTHLY AGGREGATED M&V (REPORTING-LEVEL VIEW)
# ----------------------------------------------------
st.markdown("### 🔹 Monthly Aggregated M&V Performance")

# 1️⃣ Create a copy to avoid side effects
monthly_df = timeline_df.copy()

# 2️⃣ Convert index to monthly period
monthly_df["Month"] = monthly_df.index.to_period("M").to_timestamp()

# 3️⃣ Aggregate using MEAN (power-based, Option C appropriate)
monthly_agg = (
    monthly_df
    .groupby("Month")
    .mean(numeric_only=True)
)

# 4️⃣ Plot
st.line_chart(
    monthly_agg[
        [
            "Baseline Load (kW)",
            "Adjusted Baseline Power (kW)",
            "Actual Power (kW)",
        ]
    ]
)

st.caption(
    "Monthly aggregated M&V view showing average baseline load (pre-ECM), "
    "GRU-adjusted baseline (post-ECM), and measured load. "
    "This representation aligns with IPMVP Option C reporting practices "
    "and improves interpretability by reducing short-term variability."
)

# ----------------------------------------------------
# ADJUSTED BASELINE VS ACTUAL
# ----------------------------------------------------
st.markdown("### 🔹 Adjusted Baseline vs Actual Load")
st.line_chart(df_s_f[["Actual Power (kW)", "Adjusted Baseline Power (kW)"]])

# ----------------------------------------------------
# CUMULATIVE ENERGY SAVINGS
# ----------------------------------------------------
st.markdown("### 🔹 Cumulative Energy Savings")
st.line_chart(df_s_f["Cumulative Savings Energy (kWh)"])

# ----------------------------------------------------
# COST SAVINGS
# ----------------------------------------------------
st.markdown("### 🔹 Cumulative Cost Savings")
st.line_chart(
    df_cmp_f[
        ["C1 Cumulative Cost Savings (RM)", "TOU Cumulative Cost Savings (RM)"]
    ]
)

# ----------------------------------------------------
# MD SAVINGS
# ----------------------------------------------------
st.markdown("### 🔹 Monthly Maximum Demand (MD) Savings")

if not df_md_f.empty:
    if "MD Savings (kW)" in df_md_f:
        st.bar_chart(df_md_f.set_index("Month")["MD Savings (kW)"])
    if "MD Cost Savings (RM)" in df_md_f:
        st.bar_chart(df_md_f.set_index("Month")["MD Cost Savings (RM)"])

# ----------------------------------------------------
# CO₂ AVOIDANCE
# ----------------------------------------------------
st.markdown("### 🔹 CO₂ Avoidance")
st.line_chart(df_co_f["Cumulative CO2 Avoided (tonnes)"])

# ----------------------------------------------------
# DOWNLOADABLE DATASETS
# ----------------------------------------------------
st.markdown("### 🔹 Download Analysis Datasets")

d1, d2, d3 = st.columns(3)
d4, d5, d6 = st.columns(3)

with d1:
    st.download_button(
        "⬇ Baseline Data",
        df_base.to_csv().encode(),
        file_name="Baseline_Data.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        "⬇ Clean Reporting Data (30-min)",
        df_clean_f.to_csv().encode(),
        file_name="Reporting_Clean_30min.csv",
        mime="text/csv"
    )

with d3:
    st.download_button(
        "⬇ Energy Savings (GRU)",
        df_s_f.to_csv().encode(),
        file_name="Reporting_Savings_GRU.csv",
        mime="text/csv"
    )

with d4:
    st.download_button(
        "⬇ Cost Savings (C1 vs TOU)",
        df_cmp_f.to_csv().encode(),
        file_name="Reporting_Cost_Savings.csv",
        mime="text/csv"
    )

with d5:
    st.download_button(
        "⬇ CO₂ Avoidance",
        df_co_f.to_csv().encode(),
        file_name="Reporting_CO2_Avoidance.csv",
        mime="text/csv"
    )

with d6:
    st.download_button(
        "⬇ MD Savings",
        df_md_f.to_csv().encode(),
        file_name="Reporting_MD_Savings.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.caption( "Advanced M&V 2.0 Dashboard | GRU-Based Adjusted Baseline |""Developed by Ahmad Daniel bin Mohd Yusop (FYP II) | " "Supervised by Prof. Ir. Dr. Nofri Yenita Dahlan | UiTM" )
