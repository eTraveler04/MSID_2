import pandas as pd
from utils import *


df = pd.read_csv('data/formatted_dataset.csv')
# df = pd.read_csv('dataset.csv')

# Podział na cechy i etykietę
X = df.drop(columns=['Target']) # Wszystko poza Target ( dane wejsciowe )
y = df['Target'] # Dane wyjściowe ( dane do przewidzenia ) 

# Podział na zbiór treningowy i testowy 
from sklearn.model_selection import train_test_split

# 20% pójdzie na dane testowe, 80$ na trening
# random_state - by wynik był powtarzalny przy kadym uruchomieniu. chat: Powtarzalność: za każdym uruchomieniem kodu podział danych będzie identyczny.
# stratify=y - Zapewnia, ze proporcje klas ( dropout, graduate .. ) zostaną zachowane w obu zbiorach 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Kolumny numeryczne 
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),       # brakujące wartości → średnia        --> Dostępne strategie: mean, median, most_frequent, constant 
    ('scaler', StandardScaler())                        # standaryzacja (średnia=0, std=1)
])

# Kolumny kategoryczne 
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # brakujące → najczęstsza wartość
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first')) # dodanie drop first     # zamiana na wektory binarne
])

# Połączenie wszystkiego 
# Zastosowanie rónych przekształeń do rónych kolumn w tabeli 
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'SVC': SVC()
}

# for name, model in models.items():
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])
    
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
    
#     print(f"\n=== {name} ===")
#     print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
#     print(classification_report(y_test, y_pred))

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import joblib

# 🔧 Upewnij się, że folder istnieje
os.makedirs('results', exist_ok=True)

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap='Blues')
    cm_path = f'results/{name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.title(f'Confusion Matrix: {name}')
    plt.savefig(cm_path)
    plt.close()

    # Cross-validation
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()

    # Zapis metryk
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'CV Mean Accuracy': round(cv_score, 4),
        'Precision (macro avg)': round(report['macro avg']['precision'], 4),
        'Recall (macro avg)': round(report['macro avg']['recall'], 4),
        'F1-score (macro avg)': round(report['macro avg']['f1-score'], 4)
    })

    # Zapis najlepszego modelu (np. SVC)
    if name == 'SVC':
        joblib.dump(pipeline, 'results/final_model.pkl')

# Eksport metryk do CSV
df = pd.DataFrame(results)
df.to_csv('results/model_results.csv', index=False)

# 📊 Wykresy
metrics = ['Accuracy', 'F1-score (macro avg)']
colors = ['#4c72b0', '#55a868']

for metric, color in zip(metrics, colors):
    plt.figure(figsize=(8, 5))
    plt.bar(df['Model'], df[metric], color=color)
    plt.ylim(0, 1)
    plt.title(f'{metric} dla modeli')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_name = metric.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "") + '_plot.png'
    plt.savefig(f'results/{plot_name}')
    plt.close()
