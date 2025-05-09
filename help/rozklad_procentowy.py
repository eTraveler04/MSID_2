import pandas as pd

# Wczytaj dane
# df = pd.read_csv("dataset.csv")
df = pd.read_csv("data/formatted_dataset.csv")

# Lista kolumn kategorycznych
from utils import *

# Rozkład procentowy dla każdej kolumny
for col in categorical_features:
    print(f"\n🔹 {col}")
    print(df[col].value_counts(normalize=True, dropna=True).round(4) * 100)
