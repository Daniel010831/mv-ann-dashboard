# src/postprocess_baseline.py
import pandas as pd
import os

in_path = os.path.join(os.getcwd(), "baseline_simulation_results.csv")
out_path = os.path.join(os.getcwd(), "baseline_simulation_results_with_savings.csv")

df = pd.read_csv(in_path, parse_dates=["DateTime"]).set_index("DateTime")

# Instantaneous savings (kW)
df["Savings_kW"] = df["Baseline"] - df["Actual"]

# Convert to energy for 30-min interval: kWh = kW * 0.5
df["Savings_kWh"] = df["Savings_kW"] * 0.5

# Cumulative savings
df["Cumulative_Savings_kWh"] = df["Savings_kWh"].cumsum()

# Save
df.to_csv(out_path)
print("Saved:", out_path)
print(df.tail(10))

#Aggregate daily & monthly savings (and print)
# daily aggregation
daily = df.resample("D").agg({
    "Actual": "sum",        # sums of kW readings over day (but if you want kWh use converting)
    "Baseline": "sum",
    "Savings_kWh": "sum"
})
print("Daily savings (last 7 days):")
print(daily.tail(7))

# monthly aggregation
monthly = df.resample("M").agg({"Savings_kWh": "sum"})
print("Monthly savings:")
print(monthly)

import matplotlib.pyplot as plt

# Actual vs Baseline (sample period)
plt.figure(figsize=(12,5))
plt.plot(df.index, df["Actual"], label="Actual")
plt.plot(df.index, df["Baseline"], label="Baseline (GRU)", alpha=0.8)
plt.legend()
plt.title("Actual vs Baseline (GRU)")
plt.tight_layout()
plt.show()

# Cumulative savings
plt.figure(figsize=(10,4))
plt.plot(df.index, df["Cumulative_Savings_kWh"], label="Cumulative Savings (kWh)")
plt.title("Cumulative Energy Savings (kWh)")
plt.tight_layout()
plt.show()

