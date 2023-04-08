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

precovid_df = df[df['datetime']<'2021']

algo_df = precovid_df[['datetime','temp_of_water_celsius','nitrate_mg_per_liter']]
algo_df.dropna(inplace=True)
algo_df.set_index('datetime', inplace=True)

### Multivariate: water against nitrate
# This follow concepts from the Introduction to Anomaly Detection tutorial in R
# Some of the concepts had a one to one correspondence in Python, and others didn't

# Let's build a KNN score for each point by taking the average distance from all its 20 neighbors

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale

scaled_df = pd.DataFrame(scale(algo_df))

neigh = NearestNeighbors(n_neighbors=20)
neigh.fit(scaled_df)

kneighbors = neigh.kneighbors()
knn_distances = kneighbors[0]
knn_indices = kneighbors[1]

knn_distances = pd.DataFrame(knn_distances)
knn_distances['knn_score'] = knn_distances.mean(axis=1)
knn_distances = pd.DataFrame(knn_distances['knn_score'])

scaled_df = scaled_df.join(knn_distances)

algo_df.reset_index(drop=False,inplace=True)
scaled_df = scaled_df['knn_score']

final_algo_df_with_score = algo_df.join(scaled_df)

final_algo_df_with_score['knn_score_rank'] = final_algo_df_with_score['knn_score'].rank(pct=True) * 100

### Let's take local outlier factor (LOF) of each observation
from pyod.pyod.models.lof import LOF

algo_df.set_index('datetime', inplace=True)
algo_scaled_df = pd.DataFrame(scale(algo_df))

# Fit
lof = LOF(n_neighbors=12, metric="manhattan")
lof.fit(algo_scaled_df)

# Returns a dataset, first column is probability the observation is normal,
# second column is probability observation is outlier
probs = lof.predict_proba(algo_scaled_df)
probs = pd.DataFrame(probs)

final_algo_df_with_score['lof_probs'] = probs[1]
final_algo_df_with_score['lof_score_rank'] = final_algo_df_with_score['lof_probs'].rank(pct=True) * 100

final_algo_df_with_score.to_excel(r"C:\Users\chris\OneDrive\Documents\python_rivers 04-08-23 0624pm.xlsx")

### Let's use isolation forest, which is tree based instead of distance and recursion based
from sklearn.ensemble import IsolationForest

train_df = algo_df.reset_index(drop=True).copy()

model_IF = IsolationForest(contamination=float(0.1),random_state=42)
model_IF.fit(train_df)

final_algo_df_with_score['isofor_scores'] = model_IF.decision_function(train_df)
# inlier: 1, outlier: -1
final_algo_df_with_score['isofor_anomalies'] = model_IF.predict(train_df)

# TODO: rewrite LOF using sklearn library