# dashboard/app.py
# COPY this code to run -> streamlit run dashboard/app.py
import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Advanced M&V 2.0 Dashboard (GRU)",
    layout="wide"
)

st.title("Advanced Measurement & Verification (M&V 2.0)")
st.subheader("GRU-based Adjusted Baseline | UiTM Facility")

st.markdown("""
**Model:** Gated Recurrent Unit (GRU)  
**Methodology:** IPMVP Option C / Advanced M&V (M&V 2.0)  
**Data Resolution:** 30-minute intervals  
""")

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_data
def load_data(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename), parse_dates=["DateTime"])

df_baseline = load_data("Reporting_AdjustedBaseline_GRU.csv")
df_savings = load_data("Reporting_Savings_GRU.csv")
df_c1 = load_data("Reporting_CostSavings_C1.csv")
df_tou = load_data("Reporting_CostSavings_TOU.csv")
df_co2 = load_data("Reporting_CO2_Avoidance_GRU.csv")

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

start_date = df_baseline["DateTime"].min()
end_date = df_baseline["DateTime"].max()

date_range = st.sidebar.date_input(
    "Select date range",
    [start_date, end_date]
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0]
    end_date = date_range[0]

mask = (
    (df_baseline["DateTime"] >= pd.to_datetime(start_date)) &
    (df_baseline["DateTime"] <= pd.to_datetime(end_date))
)

df_baseline_f = df_baseline.loc[mask]
df_savings_f = df_savings.loc[mask]
df_c1_f = df_c1.loc[mask]
df_tou_f = df_tou.loc[mask]
df_co2_f = df_co2.loc[mask]

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.header("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    total_energy = df_savings_f["Savings Energy (kWh)"].sum()
    st.metric("Total Energy Savings (kWh)", f"{total_energy:,.0f}")

with col2:
    total_cost = df_c1_f["Cost Savings (RM)"].sum()
    st.metric("Total Cost Savings (C1) (RM)", f"{total_cost:,.0f}")

with col3:
    total_co2 = df_co2_f["Cumulative CO2 Avoided (tonnes)"].iloc[-1]
    st.metric("Total CO₂ Avoided (tonnes)", f"{total_co2:,.2f}")

st.markdown("---")

# -------------------------------------------------
# PLACEHOLDERS FOR PLOTS (NEXT PHASE)
# -------------------------------------------------
st.header("Adjusted Baseline vs Actual (Preview)")
st.line_chart(
    df_baseline_f.set_index("DateTime")[[
        "Actual Power (kW)",
        "Adjusted Baseline Power (kW)"
    ]]
)

st.header("Cumulative Cost Savings (C1 vs TOU)")
st.line_chart(
    pd.DataFrame({
        "C1": df_c1_f["Cumulative Cost Savings (RM)"].values,
        "TOU": df_tou_f["Cumulative TOU Cost Savings (RM)"].values
    }, index=df_c1_f["DateTime"])
)

st.header("Cumulative CO₂ Avoidance")
st.line_chart(
    df_co2_f.set_index("DateTime")["Cumulative CO2 Avoided (tonnes)"]
)

st.markdown("""
---
**Note:**  
This dashboard presents post-ECM adjusted baseline analysis using a GRU model trained on pre-ECM data, following IPMVP Option C methodology.
""")
