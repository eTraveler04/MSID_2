import pandas as pd
from utils import *

# Dla dataset.csv -> 236 
# Dla formatted_dataset.csv -> 90
# Róznica to 236 - 90 = 146 ( Pozbylismy sie 146 kolumn!)
# Plus po zrobieniu drop first zostanie 90 - 17 = 73 ( To są kategorialne. Nie mozna zapomniec o 17 kolumnach liczbowych.)

# Wczytaj plik CSV
try:
    df = pd.read_csv('data/dataset.csv')
    # df = pd.read_csv("formatted_dataset.csv")
except FileNotFoundError:
    print("❌ Plik 'dataset.csv' nie został znaleziony.")
    exit()


# Lista kolumn kategorycznych (użyj własnej, jeśli inna)


# Zlicz liczbę unikalnych wartości w każdej kolumnie
category_counts = {col: df[col].nunique(dropna=True) for col in categorical_features}

# Podsumowanie
total_categories = sum(category_counts.values())

# Wynik
print("Liczba podkategorii w każdej kolumnie:")
for col, count in category_counts.items():
    print(f"- {col}: {count}")

print(f"\n🔢 Łączna liczba podkategorii: {total_categories}")

