import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\chris\Downloads\delaware_river_trenton_nj last_five_years clean 04-01-23 0614pm.xlsx")

df['turbidity'] = pd.to_numeric(df['turbidity'],errors='coerce')
df['chlorophyll_mg_per_liter'] = pd.to_numeric(df['chlorophyll_mg_per_liter'],errors='coerce')
df['nitrate_mg_per_liter'] = pd.to_numeric(df['nitrate_mg_per_liter'],errors='coerce')
df['dissolved_oxygen_mg_per_liter'] = pd.to_numeric(df['dissolved_oxygen_mg_per_liter'],errors='coerce')
df['dissolved_oxygen_percent'] = pd.to_numeric(df['dissolved_oxygen_percent'],errors='coerce')
df['specific_conductance'] = pd.to_numeric(df['specific_conductance'],errors='coerce')
df['ph'] = pd.to_numeric(df['ph'],errors='coerce')

df.drop(columns=['discharge_per_second'],inplace=True)
# One observation per day
# Take mean of numerical fields
daily_df = df.resample('D', on='datetime').mean()
monthly_df = df.resample('M', on='datetime').mean()

summary_df = daily_df.describe()


chlorophyll_df = daily_df['chlorophyll_mg_per_liter'].copy()
chlorophyll_df = chlorophyll_df['chlorophyll_mg_per_liter'].dropna()

chlorophyll_df = chlorophyll_df.reset_index()
chlorophyll_df.plot()

import seaborn as sns

sns.lineplot(x='datetime',y='chlorophyll_mg_per_liter',data=chlorophyll_df)

plt.boxplot(chlorophyll_df['chlorophyll_mg_per_liter'])
plt.show()


# Chlorophyll, while it seems to have outliers, doesn't appear to have a normal distribution.
# Let's go with nitrate, which does.

plt.hist(df['nitrate_mg_per_liter'])