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

algo_df = precovid_df[['temp_of_water_celsius','nitrate_mg_per_liter']]
algo_df.dropna(inplace=True)
algo_df.reset_index(drop=True,inplace=True)

### Multivariate: water against nitrate
# The original tutorial is a little different from what I am doing now

from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=20)
neigh.fit(algo_df)

kneighbors = neigh.kneighbors()
knn_distances = kneighbors[0]
knn_indices = kneighbors[1]

# Let's build a KNN score for each point by taking the average distance from all its 20 neighbors
knn_distances = pd.DataFrame(knn_distances)
knn_distances['knn_score'] = knn_distances.mean(axis=1)
knn_distances = pd.DataFrame(knn_distances['knn_score'])

algo_df = algo_df.join(knn_distances)

knn_anomalies = algo_df.nlargest(20, 'knn_score')

from sklearn.preprocessing import scale

scale(algo_df)
# KNNs

from pyod.pyod.models.knn import KNN # Strange import issue, so I had to go down two subdirectories
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np



# Tune the number of neighbors and contamination
knn = KNN(n_neighbors=20, contamination=0.01, n_jobs=-1)
# Let's find outliers
knn.fit(algo_df)
probs = knn.predict_proba(algo_df)
# Use 55% threshold for filtering
is_outlier = probs[:, 1] > 0.55
# Isolate the outliers
outliers = algo_df[is_outlier]
len(outliers)



# Quantile transformer
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal')
# What is the feature (independent variable) and what is the response (dependent variable?)
# The feature or the cause is the temperature. The response or the effect is the nitrate levels.
X = algo_df.drop('nitrate_mg_per_liter', axis=1)
y = algo_df['nitrate_mg_per_liter']

X.loc[:,:] = qt.fit_transform(X)

# Fit a knn model on the scaled data
knn_scaled = KNN(n_neighbors=20, contamination=0.01, n_jobs=-1)
knn_scaled.fit(X, y)


# load the data, split off the target
cancer = load_breast_cancer(as_frame=True)
cancer_df = cancer.frame
cancer_features = cancer_df.drop(columns='target')
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(cancer_features, cancer_df.target, random_state=23)

# For comparison, run KNN again without scaling:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("accuracy without scaling: ", knn.score(X_test, y_test))

# Scale the training data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Fit a knn model on the scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

# Scale the test data
X_test_scaled = scaler.transform(X_test)
print("accuracy with scaling: ", knn_scaled.score(X_test_scaled, y_test))


# Tune the number of neighbors and contamination
knn = KNN(n_neighbors=20, contamination=0.01, n_jobs=-1)
# Let's find outliers in summer of 2020
knn.fit(X,y)
probs = knn.predict_proba(algo_df)
# Use 55% threshold for filtering
is_outlier = probs[:, 1] > 0.55
# Isolate the outliers
outliers = algo_df_summer_2020_for_fit[is_outlier]
len(outliers)




# Tune the number of neighbors and contamination
knn = KNN(n_neighbors=20, contamination=0.01, n_jobs=-1)
# Let's find outliers in summer of 2020
knn.fit(X)
probs = knn.predict_proba(X)
# Use 55% threshold for filtering
is_outlier = probs[:, 1] > 0.55
# Isolate the outliers
outliers = X[is_outlier]
len(outliers)

# Split the data into features (X) and target (y)
X = algo_df_summer_2020_for_fit.drop('nitrate_mg_per_liter', axis=1)
y = algo_df_summer_2020_for_fit['nitrate_mg_per_liter']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)






