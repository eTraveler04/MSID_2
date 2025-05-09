# Dla Historiogram klasy dropout 

Wyraźne piki przy 0 i przy 1

Duża liczba próbek ma prawdopodobieństwo bliskie 0 → model jest niemal pewny, że te próbki nie są dropoutami.

Drugi pik przy prawdopodobieństwie bliskim 1 → model jest niemal pewny, że to są dropouty.

Stosunkowo mało przypadków w środku rozkładu

Niewiele próbek trafia w środek (np. 0.4–0.6), co oznacza, że model rzadko jest „niepewny” — większość predykcji jest wyraźna.

Drobny ogon rozkładu

Pojedyncze próbki mogą mieć umiarkowane prawdopodobieństwo (0.1–0.4 lub 0.6–0.9), co sugeruje, że dla nielicznych przypadków model miał wątpliwości.


# Na histogramie dla klasy „graduate” widać inny kształt niż dla „dropout”:

Silne skupienie przy 0
– Najwięcej próbek ma bardzo niskie P(graduate) (<0.1), czyli model jest przekonany, że te próbki nie należą do klasy „graduate” (to zapewne dropout lub enrolled).

Długi „ogon” ku wyższym prawdopodobieństwom
– Wartości P(graduate) rozciągają się aż do ~0.8, ale im wyżej, tym próbek jest coraz mniej.
– Oznacza to, że dla niewielu próbek model ma dużą pewność, że to absolwenci (graduate).

Mało wartości bliskich 1
– W przeciwieństwie do klasy „dropout”, gdzie było dużo pików blisko 1, tutaj praktycznie nie ma próbek z P≈1.
– Model nigdy nie jest w stu procentach pewny „graduate” – najwyższe prawdopodobieństwa sięgają ~0.8.



Co to oznacza w praktyce?
- Model zwraca wektor prawdopodobieństw [𝑃(Graduate),𝑃(Enrolled),𝑃(Dropout)].
- Klasyfikacja: wybierasz klasę o najwyższym prawdopodobieństwie (argmax).
- Używasz funkcji kosztu cross‐entropy zamiast MSE.