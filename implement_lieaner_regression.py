#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def closed_form_regression(X, y):
    """
    Oblicza wagi w regresji liniowej metodą closed-form:
        w = (X^T X)^{-1} X^T y
    X powinno zawierać już kolumnę bias (jedynki).
    """
    XtX     = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty     = X.T @ y
    return XtX_inv @ Xty

def main():
    results_dir = "reg_results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Wczytanie danych (wszystkie kolumny numeryczne)
    df = pd.read_csv("formatted_dataset.csv")

    counts = df["Target"].value_counts().sort_index()  # liczba próbek w każdej klasie
    props  = df["Target"].value_counts(normalize=True).sort_index() * 100  # procentowo

    print("Rozkład klas w Target (liczba próbek):")
    print(counts)
    print("\nRozkład klas w Target (procentowo):")
    print(props.round(2).astype(str) + " %")

    # 2. Przygotowanie X i y — tutaj y = "Age at enrollment"
    target = "Age at enrollment"
    y      = df[target].to_numpy()
    X      = df.drop(columns=[target]).to_numpy()

    # 3. Podział na trening/test (80/20)
    n_samples       = X.shape[0]
    split           = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Standaryzacja cech (na podstawie X_train)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    # 5. Dodanie kolumny bias (jedynki)
    X_train_bias = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    X_test_bias  = np.hstack([np.ones((X_test.shape[0],1)),  X_test])

    # 6. Obliczenie wag (closed-form)
    w = closed_form_regression(X_train_bias, y_train)

    # 7. Predykcja
    y_pred = X_test_bias @ w

    # 8a. Mean Squared Error
    mse = np.mean((y_pred - y_test)**2)
    print(f"MSE (closed-form, standaryzowane) na kolumnie {target}: {mse:.4f}")

    # 8b. Dodatkowe metryki dla regresji
    mae  = np.mean(np.abs(y_pred - y_test))          # Mean Absolute Error
    rmse = np.sqrt(mse)                              # Root Mean Squared Error
    ss_res = np.sum((y_test - y_pred)**2)            # suma kwadratów reszt
    ss_tot = np.sum((y_test - np.mean(y_test))**2)   # całkowita suma wariancji
    r2   = 1 - ss_res/ss_tot                         # współczynnik determinacji

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")


    # 9. Zapis wyników do CSV
    preds_path   = os.path.join(results_dir, "predictions_closed_form.csv")
    weights_path = os.path.join(results_dir, "weights_closed_form.csv")

    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    }).to_csv(preds_path, index=False)

    pd.Series(w, name="weight").to_csv(weights_path, index=False)

    print("🔖 Zapisano pliki:")
    print(f" - {preds_path}")
    print(f" - {weights_path}")

    # ----------------------------
    # 10. Generowanie wykresów
    # ----------------------------
    # Obliczenie residuów
    residuals = y_test - y_pred

    # 10.1 Scatter: rzeczywiste vs predykowane
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    # linia y=x dla odniesienia
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Rzeczywiste y")
    plt.ylabel("Predykowane y")
    plt.title("Scatter: Rzeczywiste vs Predykowane")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "true_vs_pred.png"))
    plt.close()

    # 10.2 Histogram residuów
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel("Residua (y_true - y_pred)")
    plt.ylabel("Liczba próbek")
    plt.title("Histogram residuów")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "residuals_hist.png"))
    plt.close()

    # 10.3 Residua vs wartości predykowane
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predykowane y")
    plt.ylabel("Residua")
    plt.title("Residua vs Predykowane y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "residuals_vs_pred.png"))
    plt.close()

    print("📊 Wykresy zapisane w:", results_dir)

if __name__ == "__main__":
    main()
