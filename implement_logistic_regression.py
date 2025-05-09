#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Mapowanie etykiet numerycznych na czytelne nazwy klas
label_map = {0: 'dropout', 1: 'graduate', 2: 'enrolled'}

# ----------------------------
# FUNKCJE POMOCNICZE
# ----------------------------

def softmax(Z): # zamiast sigmoidu
    """
    Softmax: zamienia surowe logity Z (NxK) na prawdopodobieństwa (sumują się do 1 w każdym wierszu).
    Stabilizujemy obliczenia przez odjęcie maksymalnej wartości w każdym wierszu.
    """
    Z_exp = np.exp(Z - Z.max(axis=1, keepdims=True))
    return Z_exp / Z_exp.sum(axis=1, keepdims=True)

def one_hot(y, K):
    """
    Zamienia wektor etykiet y (N,) na macierz one-hot (NxK).
    Dzięki temu możemy użyć cross-entropii jako funkcji kosztu.
    """
    N = y.shape[0]
    Y = np.zeros((N, K))
    # dla każdej próbki ustawiamy 1 w kolumnie odpowiadającej jej etykiecie
    Y[np.arange(N), y] = 1
    return Y

def compute_loss(P, Y):
    """
    Oblicza stratę cross-entropy:
      -sum(Y * log(P)) / N
    Dodajemy malutką eps (1e-15), by uniknąć log(0).
    """
    return -np.mean(np.sum(Y * np.log(P + 1e-15), axis=1))

def train_logistic_gd_with_history(X, y, K, lr=0.05, epochs=200, batch_size=128):
    """
    Trenuje regresję logistyczną metodą mini-batch gradient descent.
    Zwraca finalne wagi W oraz historię strat (po każdej epoce).
    """
    N, D = X.shape
    # inicjalizacja wag: D cech × K klas
    W = np.zeros((D, K))
    # przygotowanie one-hot dla etykiet
    Y = one_hot(y, K)

    history_loss = []
    # history_W możesz odkomentować, jeśli chcesz śledzić same wagi
    # history_W = []

    for epoch in range(1, epochs+1):
        # losowo permutujemy próbki, by batche były różne co epokę
        perm = np.random.permutation(N)
        X_shuff, Y_shuff = X[perm], Y[perm]

        # dzielimy na mini-batche
        for start in range(0, N, batch_size):
            Xb = X_shuff[start:start+batch_size]
            Yb = Y_shuff[start:start+batch_size]

            # krok forward: logity i softmax
            Z  = Xb @ W
            P  = softmax(Z)

            # gradient cross-entropy: X^T (P - Y) / batch_size
            grad = Xb.T @ (P - Yb) / Xb.shape[0]

            # update wag
            W -= lr * grad

        # po każdej epoce liczymy stratę na całym zbiorze treningowym
        P_full = softmax(X @ W)
        loss = compute_loss(P_full, Y)
        history_loss.append(loss)
        # history_W.append(W.copy())

        # co 10 epok (i za pierwszym razem) drukujemy postęp
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} — loss: {loss:.4f}")

    return W, history_loss

def plot_loss_curve(loss_history, out_dir):
    """
    Rysuje i zapisuje krzywą strat (loss vs epochs).
    """
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title('Cross-Entropy Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

def plot_probs(probs, out_dir):
    """
    Dla każdej klasy rysuje:
     - wykres liniowy prawdopodobieństwa w kolejnych próbkach
     - histogram rozkładu tych prawdopodobieństw
    """
    # nazwy kolumn według label_map
    cols = [label_map[i] for i in range(probs.shape[1])]
    dfp = pd.DataFrame(probs, columns=cols)

    # 1) wykresy liniowe
    for cls in cols:
        plt.figure()
        plt.plot(dfp.index, dfp[cls], lw=1)
        plt.title(f"Prawdopodobieństwo klasy “{cls}”")
        plt.xlabel("Indeks próbki")
        plt.ylabel("Prawdopodobieństwo")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cls}_line.png"))
        plt.close()

    # 2) histogramy
    for cls in cols:
        plt.figure()
        plt.hist(dfp[cls], bins=20)
        plt.title(f"Rozkład prawdopodobieństwa klasy “{cls}”")
        plt.xlabel("Prawdopodobieństwo")
        plt.ylabel("Liczba próbek")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cls}_hist.png"))
        plt.close()

# ----------------------------
# GŁÓWNY BLOK
# ----------------------------

def main():
    # katalog na wszystkie wyniki i wykresy
    out_dir = "logreg_results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Wczytanie danych z CSV
    #    - Pandas do I/O, później tylko NumPy na macierzach
    # df = pd.read_csv("data/formatted_dataset.csv")
    df = pd.read_csv("data/dane_bez1.csv")


    counts = df["Target"].value_counts().sort_index()  # liczba próbek w każdej klasie
    props  = df["Target"].value_counts(normalize=True).sort_index() * 100  # procentowo

    print("Rozkład klas w Target (liczba próbek):")
    print(counts)
    print("\nRozkład klas w Target (procentowo):")
    print(props.round(2).astype(str) + " %")

    # fig = px.histogram(df, x="Target", nbins=30, marginal="box")
    # fig.show(renderer="browser")

    y = df["Target"].to_numpy()
    X = df.drop(columns=["Target"]).to_numpy()

    # 2. Podział na zbiór treningowy i testowy (80/20)
    N = X.shape[0] # to liczba wierszy (próbek) w macierzy
    split = int(0.8 * N)
    # Pierwsze split próbek to trening, pozostałe to test
    X_train = X[:split]    # wiersze od 0 do split-1
    X_test  = X[split:]    # wiersze od split do N-1

    y_train = y[:split]    # odpowiadające etykiety
    y_test  = y[split:]

    # 3. Standaryzacja cech: (x - mean) / std na podstawie X_train
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    # 4. Dodanie bias (kolumna jedynek) – pozwala modelowi mieć wyraz wolny w równaniu
    X_train = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    X_test  = np.hstack([np.ones((X_test.shape[0],1)),  X_test])
    # Pozwala modelowi przesunąć funkcję decyzyjną lub linię w górę/dół,
    # Bez biasu funkcja zawsze przechodziłaby przez początek układu, co ogranicza jej możliwości dopasowania.

    # 5. Trening modelu
    K = len(label_map)  # liczbę klas bierzemy z długości mapy
    W, loss_history = train_logistic_gd_with_history(
        X_train,    # macierz cech treningowych (N_train × (D+1)), zawiera bias
        y_train,    # wektor etykiet treningowych (N_train,)
        K,          # liczba klas
        lr=0.05,    # współczynnik uczenia (learning rate)
        epochs=200, # liczba pełnych przejść przez dane (epok)
        batch_size=128  # rozmiar mini-batch’y
    )
    
    # 6. Ewaluacja na zbiorze testowym
    P_test = softmax(X_test @ W)          # przewidywane prawdopodobieństwa
    y_pred = np.argmax(P_test, axis=1)    # wybór klasy o najwyższym P
    acc = np.mean(y_pred == y_test)
    print(f"Accuracy na zbiorze testowym: {acc:.4f}")

    # 7. Zapis wag i wyników do plików CSV
    pd.DataFrame(W, columns=[label_map[i] for i in range(K)]) \
      .to_csv(f"{out_dir}/weights_logreg.csv", index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}) \
      .to_csv(f"{out_dir}/predictions_logreg.csv", index=False)
    pd.DataFrame(P_test, columns=[label_map[i] for i in range(K)]) \
      .to_csv(f"{out_dir}/probs_logreg.csv", index=False)

    # 8. Generowanie wykresów
    print("🔍 Generuję krzywą strat...")
    plot_loss_curve(loss_history, out_dir)

    print("🔍 Generuję wykresy prawdopodobieństw...")
    plot_probs(P_test, out_dir)

if __name__ == "__main__":
    main()
