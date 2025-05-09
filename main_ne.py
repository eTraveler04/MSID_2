
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib
import matplotlib.pyplot as plt

# Wczytanie danych
df = pd.read_csv('dane_bez1.csv')
X = df.drop(columns=['Target'])
y = df['Target']

# Podział na train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Pipeline preprocessingowy
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Modele
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'SVC': SVC()
}

# Ewaluacja
def evaluate(model, X, y, dataset_name):
    y_pred = model.predict(X)
    return {
        "Dataset": dataset_name,
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision (macro)": round(precision_score(y, y_pred, average='macro'), 4),
        "Recall (macro)": round(recall_score(y, y_pred, average='macro'), 4),
        "F1-score (macro)": round(f1_score(y, y_pred, average='macro'), 4)
    }

# Folder na wyniki
os.makedirs('results_no_enrolled', exist_ok=True)

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)

    for X_data, y_data, label in [
        (X_train, y_train, "train"),
        (X_val, y_val, "val"),
        (X_test, y_test, "test")
    ]:
        metrics = evaluate(pipeline, X_data, y_data, label)
        metrics["Model"] = name
        results.append(metrics)

    if name == "SVC":
        joblib.dump(pipeline, "results_no_enrolled/final_model.pkl")

# Zapis metryk
df_results = pd.DataFrame(results)
df_results.to_csv("results_no_enrolled/dataset_split_results.csv", index=False)

# Wykresy
for metric in ["Accuracy", "F1-score (macro)"]:
    plt.figure(figsize=(8, 5))
    for model_name in df_results["Model"].unique():
        subset = df_results[df_results["Model"] == model_name]
    plt.plot(subset["Dataset"], subset[metric], label=model_name, marker='o')
    plt.title(f'{metric} dla modeli na zbiorach')
    plt.ylabel(metric)
    plt.xlabel('Zbiór danych')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results_no_enrolled/{metric.lower().replace(' ', '_')}_by_dataset.png")
    plt.close()
