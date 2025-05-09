## 📘 Tabela wyników – interpretacja metryk


| Kolumna             | Znaczenie                                                                              |
| ------------------- | -------------------------------------------------------------------------------------- |
| `Dataset`           | Zbiór danych, na którym liczono metryki: `train` (treningowy), `test`                  |
| `Accuracy`          | Procent wszystkich próbek, które zostały poprawnie sklasyfikowane                      |
| `Precision (macro)` | Średnia precyzja dla każdej klasy – ile z przewidzianych „tak” to było prawdziwe „tak” |
| `Recall (macro)`    | Średnia czułość – ile prawdziwych „tak” zostało poprawnie wykrytych                    |
| `F1-score (macro)`  | Średnia harmoniczna precyzji i czułości – równoważy oba wskaźniki                      |
| `Model`             | Nazwa modelu – tutaj: `Logistic Regression` (sklearn vs własna wersja)                 |


## Tabela wyników dla własnej implementacji sklearn ( zadanie 3.0 )
![alt text](other/image.png)

## Tabela wyników dla własnej implementacji regresji logistycznej ( zadanie 4.0 )
![alt text](other/image-1.png)

## 📊 Porównanie wyników regresji logistycznych


| Dataset   | Metryka           | Sklearn (`zadanie 3.0`) | Własna (`zadanie 4.0`) |
| --------- | ----------------- | ----------------------- | ---------------------- |
| **train** | Accuracy          | 0.92                    | 0.92                   |
|           | Precision (macro) | 0.93                    | 0.92                   |
|           | Recall (macro)    | 0.91                    | 0.90                   |
|           | F1-score (macro)  | 0.92                    | 0.91                   |
| **test**  | Accuracy          | 0.91                    | 0.91                   |
|           | Precision (macro) | 0.92                    | 0.91                   |
|           | Recall (macro)    | 0.90                    | 0.90                   |
|           | F1-score (macro)  | 0.91                    | 0.90                   |


## Wnioski 

Obie wersje modelu dają prawie identyczne wyniki, co oznacza, że  własna implementacja działa bardzo dobrze.

Sklearn jest minimalnie lepszy, co prawdopodobnie wynika z zastosowania:

lepszego optymalizatora 

wbudowanej regularyzacji L2

precyzyjnych tolerancji numerycznych

Różnice ≤ 0.01

## 4.0 1a Metryki regresji:
🔹 MSE – Mean Squared Error: 23.2541
Średnia z kwadratów błędów predykcji
Im niższy, tym lepiej

Wartość 23.25 oznacza, że średni błąd podniesiony do kwadratu wynosi 23.25 lat²

🔹 MAE – Mean Absolute Error: 3.3868
Średnia z wartości bezwzględnych błędów
Oznacza, że przeciętnie model myli się o 3.39 roku przy przewidywaniu wieku zapisu

Jednostka: lata — bez podnoszenia do kwadratu

🔹 RMSE – Root Mean Squared Error: 4.8223
Pierwiastek z MSE

Też ma jednostkę „lata”
Jest bardziej „wrażliwy na duże błędy” niż MAE

Mówi: przeciętnie błąd to ~4.82 roku

🔹 R² – współczynnik determinacji: 0.4800
Zakres: od 0 do 1 (czasem też <0, jeśli model kompletnie zawodzi)

0.48 oznacza, że Twój model wyjaśnia 48% wariancji wieku zapisu

Pozostałe 52% to „szum” lub coś, czego model nie jest w stanie przewidzieć. 

Więc model jest średni. 

## 4.0 1b Przeanalizuj ograniczenia zastosowania zamkniętej formuły. Dlaczego nie jest w praktyce wykorzysytwana?

- Koszt obliczeniowy i pamięciowy
Aby obliczyć,Mnożenie macierzy, operacji, a odwrócenie tej macierzy to 
Gdy masz np. 10 000 cech, odwracanie macierzy wymaga setek miliardów operacji i ogromnej ilości pamięci RAM.

- Brak skalowalności do ogromnych danych
Dla zbiorów o milionach próbek lub dziesiątkach tysięcy cech pojęcie „wszystko w pamięci” się rozpada.

- Prawdziwe aplikacje uczą modele na strumieniach danych lub w mini‐batchach, co zamknięta formuła uniemożliwia.

-  Tylko regresja liniowa
Zamknięta formuła zadziała jedynie dla regresji z MSE (błędem kwadratowym) i modelu liniowego.

W closed-form nie masz kontroli nad uczeniem
Nie możesz użyć early stopping, ani śledzić postępu jak w gradient descent

Nie zadziała dobrze, jeśli kolumny są skorelowane
Jeśli w danych są kolumny, które są kombinacją innych (np. „suma” dwóch kolumn), macierz może być nieodwracalna.
Wtedy np.linalg.inv(...) może rzucić błąd lub dać niestabilne wyniki

- nie ma pętli, nie robisz tego w każdej epoce jak w gradient descent.

