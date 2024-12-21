# Metody Eksploracji Danych (MED)

Autorzy: Izabela Skowron i Szymon Florek

## Opis Projektu

Niniejsze repozytorium zawiera materiały i rozwiązania z laboratoriów realizowanych w ramach przedmiotu **Metody Eksploracji Danych**. Celem zajęć jest zapoznanie się z różnorodnymi technikami analizy danych, w tym:

- **Regresja liniowa**: modelowanie zależności między zmiennymi ciągłymi.
- **Regresja logistyczna**: analiza zależności między zmiennymi, gdzie zmienna zależna jest dychotomiczna.
- **Klasyfikatory**: implementacja i ocena różnych algorytmów klasyfikacyjnych, takich jak drzewa decyzyjne, k-NN, SVM i inne.

## Struktura Repozytorium

Repozytorium podzielone jest na katalogi odpowiadające poszczególnym laboratoriom:

- `MED_LAB_1/` - materiały z pierwszego laboratorium dotyczącego analizy regresji.
- `MED_LAB_2/` - materiały z drugiego laboratorium poświęconego analizie skupień.
- `MED_LAB_3/` - materiały z trzeciego laboratorium obejmującego analizę płatków śniadaniowych. Obecnie w produkcji!

## Wymagania

Aby zapewnić kompatybilność z Pythonem 3.12, zaleca się użycie następujących wersji pakietów:

- **NumPy**: `1.26.3`
- **pandas**: `2.2.3`
- **Matplotlib**: `3.7.1`
- **scikit-learn**: `1.5.2`
- **statsmodels**: `0.14.4`

Aby zainstalować te pakiety, użyj następujących poleceń:

```bash
pip install numpy==1.26.3
pip install pandas==2.2.3
pip install matplotlib==3.7.1
pip install scikit-learn==1.5.2
pip install statsmodels==0.14.4
```

Pamiętaj, że niektóre z tych pakietów mogą wymagać dodatkowych zależności. Zaleca się instalację wirtualnego środowiska, aby uniknąć konfliktów z innymi projektami.

Jeśli napotkasz problemy z instalacją lub działaniem tych pakietów, sprawdź dokumentację każdego z nich pod kątem kompatybilności z Pythonem 3.12. 

## Uruchomienie

1. Sklonuj repozytorium:

   ```bash
   git clone https://github.com/Floressek/MED.git
   ```

2. Przejdź do wybranego katalogu laboratorium:

   ```bash
   cd MED/MED_LAB_X  # Zamień X na numer laboratorium
   ```

3. Uruchom skrypt:

   ```bash
   python main.py
   ```

## Licencja

Projekt jest licencjonowany na podstawie licencji MIT.
