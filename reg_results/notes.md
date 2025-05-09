# Implement linear regression 
2b. Dlaczego w praktyce nie jest wykorzystywana?

1. Koszt obliczeniowy i pamięciowy
Aby obliczyć,Mnożenie macierzy, operacji, a odwrócenie tej macierzy to 
Gdy masz np. 10 000 cech, odwracanie macierzy wymaga setek miliardów operacji i ogromnej ilości pamięci RAM.

4. Brak skalowalności do ogromnych danych
Dla zbiorów o milionach próbek lub dziesiątkach tysięcy cech pojęcie „wszystko w pamięci” się rozpada.

- Prawdziwe aplikacje uczą modele na strumieniach danych lub w mini‐batchach, co zamknięta formuła uniemożliwia.

5. Tylko regresja liniowa
Zamknięta formuła zadziała jedynie dla regresji z MSE (błędem kwadratowym) i modelu liniowego.

- Nie zastosujesz jej do:
- regresji logistycznej (cross‐entropy)
- sieci neuronowych
- innych funkcji kosztu, które nie dają się sprowadzić do odwrócenia jednej macierzy.
