import pandas as pd

try:
    df = pd.read_csv("data/PenangBaselineData.csv")
    print("CSV loaded successfully.")
    #print(df.head())
except Exception as e:
    print("Error loading CSV:", e)
#print(df.shape)
#print(df.info())
#print(df.tail())

df = df.sort_values('DateTime')
df = df.set_index('DateTime')

# Check if there are missing timestamps
missing = df.asfreq('30T')
print(missing.isnull().sum())
