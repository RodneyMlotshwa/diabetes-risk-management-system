import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# load and prepare
df = pd.read_csv('cleaned_dataset.csv')
print(f"Dataset shape: {df.shape}")

cluster_features = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c',
    'diabetes_risk_score'
]

X = df[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Using {X.shape[1]} features for clustering")

# find optimal k using Elbow and Silhouette methods 
inertia = []
sil_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
    sample_idx = np.random.choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
    sil = silhouette_score(X_scaled[sample_idx], kmeans.labels_[sample_idx])
    sil_scores.append(sil)
    print(f"k={k} → Inertia={kmeans.inertia_:.0f} | Silhouette={sil:.3f}")

# Plot to confirm choice
fig, ax1 = plt.subplots(1, 2, figsize=(12, 5))
ax1[0].plot(K_range, inertia, 'bo-')
ax1[0].set_title('Elbow Method')
ax1[0].set_xlabel('k')
ax1[0].set_ylabel('Inertia')
ax1[1].plot(K_range, sil_scores, 'go-')
ax1[1].set_title('Silhouette Score')
ax1[1].set_xlabel('k')
ax1[1].set_ylabel('Score')
plt.tight_layout()
plt.show()

# run KMeans with optimal k
optimal_k = 3   

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# interpret clusters 
print("\n=== CLUSTER PROFILES (average values) ===")
profile = df.groupby('Cluster')[cluster_features + ['diagnosed_diabetes']].mean().round(2)
print(profile)

print("\n=== % DIABETES IN EACH CLUSTER ===")
print((df.groupby('Cluster')['diagnosed_diabetes'].mean() * 100).round(2))

# Save results
profile.to_csv('cluster_profiles_k3.csv', index=True)

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['glucose_fasting'], y=df['bmi'], hue=df['Cluster'], palette='viridis', alpha=0.6)
plt.title('K=3 Clusters: Fasting Glucose vs BMI')
plt.show()

print("\n Clustering with k=3 is complete!")
print("Check the tables above and the file: cluster_profiles_k3.csv")

print("\n=== RANDOM FOREST + SHAP ANALYSIS ===")

# Define target and features
y = df['diagnosed_diabetes']
X_rf = df[cluster_features]

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_rf, y)

print("Random Forest model trained.")

# Feature importance (quick win)
importances = pd.Series(rf.feature_importances_, index=cluster_features).sort_values(ascending=False)

print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
print(importances.head(10))

# Plot feature importance
plt.figure(figsize=(8,6))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

# ---------------- SHAP ----------------
print("\nCalculating SHAP values (this may take a bit)...")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_rf)

# SHAP summary plot
shap.summary_plot(shap_values[1], X_rf)