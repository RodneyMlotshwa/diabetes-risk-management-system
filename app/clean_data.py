import pandas as panda

dataset = panda.read_csv("data/raw/Diabetes_and_LifeStyle_Dataset_.csv")

print(dataset.info())
print(dataset.describe())
print("Number of rows:", len(dataset))