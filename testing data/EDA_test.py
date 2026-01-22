import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("data/PenangBaselineData.csv")

print("Shape:", df.shape) #Shape(rows,columns)
print(df.info()) #columns and data types

#print first and last rows
print(df.head()) 
print(df.tail())

#convert DataTime column to datetime & set index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')
df = df.set_index('DateTime')
print(df.index) #verify

#check for missing values
print(df.isnull().sum())
print("\nPercentage missing:\n", df.isnull().mean() * 100)

#check for missing timestamps
df_continuous = df.asfreq('30T')
missing_timestamps = df_continuous[df_continuous.isnull().any(axis=1)]
print("Missing timestamps:", missing_timestamps.shape[0])
missing_timestamps.head()

#statistical summary
print(df.describe())

#quick plot of load consumption
plt.figure(figsize=(12,4))
plt.plot(df['Load Consumption (kW)'])
plt.title("Load Consumption Over Time (Raw Data)")
plt.xlabel("Time")
plt.ylabel("kW")
plt.show()

#check categorical variables (unique values)
for col in ['Lecture/Non-lecture', 'Public Holiday', 'Semester Break', 'Semester : Lecture/Office']:
    print(col, df[col].unique())
