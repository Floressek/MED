### Interpretacja metryk oceny modelu:

1. **R² (R-squared / Współczynnik determinacji)**
- Zakres: 0 do 1 (może być ujemny dla bardzo złych modeli)
- Interpretacja:
  - R² = 1: idealny model (100% wyjaśnionej wariancji)
  - R² > 0.9: bardzo dobry model
  - R² > 0.7: dobry model
  - R² > 0.5: umiarkowany model
  - R² < 0.3: słaby model
  - R² ≤ 0: model gorszy niż średnia
- Przykład: R² = 0.85 oznacza, że model wyjaśnia 85% zmienności w danych

2. **MSE (Mean Squared Error / Błąd średniokwadratowy)**
- Zakres: 0 do ∞ (im niższy, tym lepiej)
- Jednostka: kwadrat jednostki oryginalnej (np. dla milionów będzie w bilionach)
- Interpretacja:
  - Duże wartości są mocno "karane" ze względu na kwadrat
  - Trudny w bezpośredniej interpretacji przez jednostkę kwadratową
  - Użyteczny głównie do porównywania modeli
- Przykład: MSE = 1000000 dla danych w milionach oznacza średni błąd kwadratowy rzędu biliona

3. **RMSE (Root Mean Squared Error / Pierwiastek błędu średniokwadratowego)**
- Zakres: 0 do ∞ (im niższy, tym lepiej)
- Jednostka: ta sama co dane oryginalne
- Interpretacja:
  - Pokazuje "typowy" błąd predykcji w oryginalnych jednostkach
  - Łatwiejszy w interpretacji niż MSE
  - Wartości powinny być analizowane w kontekście skali danych
- Przykład: RMSE = 1000 dla danych w milionach oznacza, że typowy błąd predykcji to około 1 miliard

4. **MAE (Mean Absolute Error / Średni błąd bezwzględny)**
- Zakres: 0 do ∞ (im niższy, tym lepiej)
- Jednostka: ta sama co dane oryginalne
- Interpretacja:
  - Średnia wartość błędu bez względu na kierunek
  - Zwykle mniejszy niż RMSE (nie karze tak mocno dużych błędów)
  - Łatwy w interpretacji
- Przykład: MAE = 800 dla danych w milionach oznacza, że średnio predykcje różnią się o 800 milionów

5. **MAPE (Mean Absolute Percentage Error / Średni bezwzględny błąd procentowy)**
- Zakres: 0% do ∞% (im niższy, tym lepiej)
- Jednostka: procenty
- Interpretacja:
  - MAPE < 10%: bardzo dobra dokładność
  - MAPE < 20%: dobra dokładność
  - MAPE < 30%: umiarkowana dokładność
  - MAPE > 30%: słaba dokładność
- Przykład: MAPE = 15% oznacza, że średnio predykcje różnią się o 15% od rzeczywistych wartości

### Wskazówki do analizy:

1. **Porównanie train vs test:**
- Jeśli metryki na zbiorze testowym są znacznie gorsze niż na treningowym -> przeuczenie modelu
- Jeśli metryki są podobne -> model dobrze generalizuje

2. **Co patrzeć najpierw:**
1. MAPE - daje szybki obraz procentowej dokładności
2. R² - pokazuje ogólną jakość dopasowania
3. RMSE - pokazuje skalę błędów w oryginalnych jednostkach

3. **Kiedy model jest akceptowalny:**
- Zależy od dziedziny i wymagań
- Dla danych finansowych/biznesowych:
  - MAPE < 20% często uznawane za akceptowalne
  - R² > 0.7 uznawane za dobre dopasowanie
  - RMSE powinien być analizowany w kontekście skali biznesowej

4. **Czerwone flagi:**
- MAPE > 50%
- R² < 0.5
- Duże różnice między metrykami train i test
- RMSE większy niż średnia wartość prognozowanej zmiennej
