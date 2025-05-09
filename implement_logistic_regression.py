#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
Zamienia surowy wynik (tzw. logit) w wartość z przedziału (0, 1)
Jest interpretowana jako prawdopodobieństwo przynależności do klasy 1
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

"""
Liczy średnią funkcję kosztu (binary cross-entropy)
Jest to miara jak bardzo model się myli
Im mniejszy wynik, tym lepiej
"""
def compute_loss(P, Y):
    return -np.mean(Y * np.log(P + 1e-15) + (1 - Y) * np.log(1 - P + 1e-15))

def train_logistic_gd_with_history(X, y, lr=0.05, epochs=200, batch_size=128):
    N, D = X.shape
    """
    W – wektor wag, który będzie się uczył
    Każdy element W[j] to waga dla jednej cechy
    """
    W = np.zeros(D)
    history_loss = []

    # Pętla ucząca ( 200 epok ) - Każda epoka = jedno pełne przejście przez zbiór treningowy
    for epoch in range(1, epochs+1):
        perm = np.random.permutation(N)
        X_shuff, y_shuff = X[perm], y[perm]

        """
        Przetwarzasz dane w kawałkach, np. po 128 rekordów
        To poprawia stabilność i przyspiesza trening
        """
        for start in range(0, N, batch_size):
            Xb = X_shuff[start:start+batch_size]
            yb = y_shuff[start:start+batch_size]
            # Obliczanie prognoz i gradientu:
            Z = Xb @ W # logity (surowy wynik)
            P = sigmoid(Z) # prawdopodobieństwa
            grad = Xb.T @ (P - yb) / Xb.shape[0] # gradient funkcji kosztu względem wag
            W -= lr * grad # aktualizacja wag. Przesuwasz wagi w kierunku zmniejszającym stratę ( lr = learning rate (krok uczenia) )

        P_full = sigmoid(X @ W)
        loss = compute_loss(P_full, y)
        history_loss.append(loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} — loss: {loss:.4f}")

    return W, history_loss

"""
Rysuje wykres straty (loss) w zależności od epoki
Pomaga obserwować, czy model się uczy (spadek loss)
"""
def plot_loss_curve(loss_history, out_dir):
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title('Binary Cross-Entropy Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

def main():
    out_dir = "logreg_results"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv("data/df_ne.csv")
    y = df["Target"].to_numpy()
    X = df.drop(columns=["Target"]).to_numpy()

    # podział na trening/test
    N = X.shape[0]
    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    
    """
    Standaryzacja
    Dzięki temu wszystkie cechy mają podobną skalę (0-1)
    Ułatwia uczenie modelu
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # dodanie biasu. Bias (wyraz wolny w₀) pozwala funkcji decyzyjnej przesunąć się poza punkt (0,0,...)
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # trening
    W, loss_history = train_logistic_gd_with_history(X_train, y_train)
    """
    Zwraca:
    W: nauczony wektor wag
    loss_history: lista strat dla każdego przebiegu
    """

    # predykcja
    P_train = sigmoid(X_train @ W)
    P_test = sigmoid(X_test @ W)
    y_pred_train = (P_train >= 0.5).astype(int)
    y_pred_test = (P_test >= 0.5).astype(int)
    # Jeśli P_test ≥ 0.5 → klasa 1 (graduate), inaczej 0 (dropout)

    # metryki
    results = []
    for y_true, y_pred, dataset in [
        (y_train, y_pred_train, "train"),
        (y_test, y_pred_test, "test")
    ]:
        results.append({
            "Dataset": dataset,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision (macro)": round(precision_score(y_true, y_pred, average="macro"), 4),
            "Recall (macro)": round(recall_score(y_true, y_pred, average="macro"), 4),
            "F1-score (macro)": round(f1_score(y_true, y_pred, average="macro"), 4),
            "Model": "Logistic Regression"
        })

    # zapis wyników
    pd.DataFrame(results).to_csv(f"{out_dir}/dataset_split_results.csv", index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}).to_csv(f"{out_dir}/predictions_logreg.csv", index=False)
    pd.DataFrame(P_test, columns=["prob_graduate"]).to_csv(f"{out_dir}/probs_logreg.csv", index=False)

    # wykres strat
    plot_loss_curve(loss_history, out_dir)

    print("✅ Zapisano metryki, predykcje i wykresy w:", out_dir)

if __name__ == "__main__":
    main()
