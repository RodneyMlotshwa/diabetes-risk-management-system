import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("data/raw/Diabetes_and_LifeStyle_Dataset_.csv")

print(dataset.info())
print(dataset.describe())
print("Number of rows:", len(dataset))

dataset = dataset.drop_duplicates() #Removes all duplicates from raw data


for col in dataset.select_dtypes(include=["float64","int64"]).columns: #Missing Value correction with median
    dataset[col] = dataset[col].fillna(dataset[col].median())

for col in dataset.select_dtypes(include=["object","string"]).columns: #Missing categorical correction with mode
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

for col in dataset.select_dtypes(include=["object","string"]).columns: #Standardize capitalization
    dataset[col] = dataset[col].str.strip().str.lower()

# Map categorical values to consistent capitalized labels
dataset["smoking_status"] = dataset["smoking_status"].map({
    "never": "Never",
    "former": "Former",
    "current": "Current"
})
dataset["employment_status"] = dataset["employment_status"].map({
    "employed": "Employed",
    "unemployed": "Unemployed",
    "student": "Student",
    "retired": "Retired"
})
dataset["income_level"] = dataset["income_level"].map({
    "high": "High",
    "upper-middle": "Upper-Middle",
    "middle": "Middle",
    "lower-middle": "Lower-Middle",
    "low": "Low"
})
dataset["diabetes_stage"] = dataset["diabetes_stage"].map({
    "gestational": "Gestational",
    "no-diabetes": "No-Diabetes",
    "pre-diabetes": "Pre-Diabetes",
    "type 1": "Type 1",
    "type 2": "Type 2"
})

# Filters based on inconsistent data
dataset = dataset[(dataset["cardiovascular_history"] >= 0) & (dataset["cardiovascular_history"] <= 1)]
dataset = dataset[(dataset["hypertension_history"] >= 0) & (dataset["hypertension_history"] <= 1)]
dataset = dataset[(dataset["family_history_diabetes"] >= 0) & (dataset["family_history_diabetes"] <= 1)]
dataset = dataset[(dataset["diagnosed_diabetes"] >= 0) & (dataset["diagnosed_diabetes"] <= 1)]
dataset = dataset[(dataset["bmi"] >= 10) & (dataset["bmi"] <= 70)]
dataset = dataset[(dataset["glucose_fasting"] > 0) & (dataset["glucose_fasting"] <= 600)]

# Saves the clean procesed data to this file path
dataset.to_csv("data/processed/cleaned_dataset.csv", index=False)
