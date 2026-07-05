import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

df = pd.read_csv("city_day.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows Before Removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate Rows After Removal:", df.duplicated().sum())

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.day_name()

print("\nAvailable Cities:")
print(df["City"].unique())

df_ncr = df[df["City"] == "Delhi"].copy()

print("\nSelected Dataset Shape:", df_ncr.shape)
print("Selected City:", df_ncr["City"].unique())

features = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3"]

air_data = df_ncr[["City", "Date"] + features].copy()

missing_percentage = air_data[features].isnull().mean() * 100

print("\nMissing Percentage:")
print(missing_percentage)

valid_features = [
    column for column in features
    if missing_percentage[column] < 50
]

print("\nSelected Features:")
print(valid_features)

air_data = air_data[["City", "Date"] + valid_features]

for column in valid_features:
    air_data[column] = air_data[column].fillna(air_data[column].median())

for column in valid_features:
    air_data = air_data[air_data[column] >= 0]

print("\nFinal Dataset Shape After Preprocessing:", air_data.shape)
print("\nMissing Values After Preprocessing:")
print(air_data.isnull().sum())

air_data[valid_features].hist(figsize=(15, 10), bins=30)
plt.suptitle("Distribution of Air Pollutants", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 7))
sns.boxplot(data=air_data[valid_features])
plt.title("Boxplot of Pollutant Values")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(
    air_data[valid_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Air Pollutants")
plt.show()

air_data["Month"] = air_data["Date"].dt.month

monthly_pm25 = air_data.groupby("Month")["PM2.5"].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly_pm25.index, monthly_pm25.values, marker="o")
plt.title("Average Monthly PM2.5 Level")
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.xticks(range(1, 13))
plt.show()

X = air_data[valid_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(k_range)
plt.show()

silhouette_results = pd.DataFrame({
    "K": list(k_range),
    "Silhouette Score": silhouette_scores
})

print("\nSilhouette Scores:")
print(silhouette_results)

plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker="o")
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(k_range)
plt.show()

optimal_k = silhouette_results.loc[
    silhouette_results["Silhouette Score"].idxmax(),
    "K"
]

print("\nSelected Optimal K:", optimal_k)

final_kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

air_data["Cluster"] = final_kmeans.fit_predict(X_scaled)

final_silhouette = silhouette_score(X_scaled, air_data["Cluster"])
final_db_score = davies_bouldin_score(X_scaled, air_data["Cluster"])

print("\nFinal Silhouette Score:", round(final_silhouette, 4))
print("Davies-Bouldin Score:", round(final_db_score, 4))

print("\nCluster Counts:")
print(air_data["Cluster"].value_counts().sort_index())

cluster_profile = air_data.groupby("Cluster")[valid_features].mean()

print("\nAverage Pollutant Values in Each Cluster:")
print(cluster_profile.round(2))

cluster_pm25 = cluster_profile["PM2.5"].sort_values()

pollution_names = [
    "Low Pollution",
    "Moderate Pollution",
    "High Pollution",
    "Severe Pollution"
]

if optimal_k != 4:
    pollution_names = [f"Pollution Group {i + 1}" for i in range(optimal_k)]

cluster_name_map = {
    cluster: name
    for cluster, name in zip(cluster_pm25.index, pollution_names)
}

air_data["Pollution_Category"] = air_data["Cluster"].map(cluster_name_map)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

air_data["PCA1"] = X_pca[:, 0]
air_data["PCA2"] = X_pca[:, 1]

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Explained Variance:", round(sum(pca.explained_variance_ratio_), 4))

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=air_data,
    x="PCA1",
    y="PCA2",
    hue="Pollution_Category",
    palette="Set2",
    alpha=0.7
)
plt.title("Air Quality Clusters Visualized Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Pollution Cluster")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=air_data,
    x="Pollution_Category",
    y="PM2.5"
)
plt.title("PM2.5 Distribution Across Pollution Clusters")
plt.xlabel("Pollution Category")
plt.ylabel("PM2.5 Concentration")
plt.xticks(rotation=20)
plt.show()

final_output = air_data[
    ["City", "Date"] + valid_features +
    ["Cluster", "Pollution_Category", "PCA1", "PCA2"]
]

final_output.to_csv("air_quality_kmeans_results.csv", index=False)

print("\nProject output saved successfully as air_quality_kmeans_results.csv")