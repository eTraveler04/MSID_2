import pandas as pd
from utils import high_cardinality_columns

def group_rare_categories(df, column, threshold=0.02):
    value_counts = df[column].value_counts(normalize=True)
    rare_values = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_values, -1)  # zamieniamy 'Other' na -1
    df[column] = df[column].astype(int)  # upewniamy się, że wszystko to liczby
    return df

def main():
    df = pd.read_csv("dataset.csv")

    # Uproszczenie rzadkich kategorii
    for col in high_cardinality_columns:
        df = group_rare_categories(df, col)
        print(f"{col}: {df[col].nunique()} unikalnych wartości po uproszczeniu")

    # Zakodowanie kolumny Target
    target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df['Target'] = df['Target'].map(target_map)

    # Pozostałe kolumny typu object (np. 'Gender', 'Course', itp.)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    df.to_csv("formatted_dataset.csv", index=False)
    print("\n✅ Zapisano do formatted_dataset.csv")

if __name__ == "__main__":
    main()
