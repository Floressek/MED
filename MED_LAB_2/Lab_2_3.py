import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def growth_curve(t, k, a, b):
    """
    Funkcja logistyczna modelująca skumulowany wzrost
    t: zmienna czasowa (numer miesiąca)
    k: maksymalna wartość (asymptota)
    a: współczynnik wzrostu
    b: parametr przesunięcia
    """
    return k / (1 + b * np.exp(-a * t))


# Tworzymy przykładowe dane
data = pd.DataFrame({
    'miesiac': range(1, 94),
    'liczba_bledow': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 0, 0, 3, 1, 1, 0, 1, 0, 2,
                      10, 0, 16, 0, 2, 2, 1, 1, 1, 0, 3, 2, 1, 6, 3, 0, 1, 1, 0, 0, 14, 1, 4, 1, 1, 7, 14, 6, 0,
                      1, 2, 19, 7, 2, 5, 1, 12, 4, 6, 4, 7, 2, 2, 3, 8, 4, 6, 3, 3, 5, 8, 0, 6, 5, 10, 0, 5, 1, 2, 1, 2,
                      1, 0]
})

# Obliczamy skumulowaną sumę błędów
data['suma_bledow'] = data['liczba_bledow'].cumsum()

# Podział na zbiór treningowy (80%) i testowy (30%)
train_size = int(1.0 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Dopasowujemy model do danych treningowych
popt, pcov = curve_fit(growth_curve,
                       train_data['miesiac'],
                       train_data['suma_bledow'],
                       p0=[300, 0.1, 90],
                       bounds=([0, 0, 0], [1000, 1, 1000]))

# Wyświetlamy znalezione parametry
k, a, b = popt
print(f"Znalezione parametry na zbiorze treningowym:")
print(f"k (maksymalna liczba błędów): {k:.2f}")
print(f"a (współczynnik wzrostu): {a:.6f}")
print(f"b (parametr przesunięcia): {b:.2f}")

# Tworzymy wykres
plt.figure(figsize=(12, 6))

# Dane treningowe
plt.scatter(train_data['miesiac'], train_data['suma_bledow'],
            label='Dane treningowe', color='blue', alpha=0.5)

# Dane testowe
plt.scatter(test_data['miesiac'], test_data['suma_bledow'],
            label='Dane testowe', color='green', alpha=0.5)

# Model
t = np.linspace(1, 93, 1000)
plt.plot(t, growth_curve(t, k, a, b), 'r-', label='Model', linewidth=2)

plt.xlabel('Numer miesiąca')
plt.ylabel('Sumaryczna liczba błędów')
plt.title('Modelowanie wzrostu liczby błędów w systemie (podział train-test)')
plt.legend()
plt.grid(True)
plt.show()

# Obliczamy błąd średniokwadratowy (MSE) dla zbioru treningowego
predicted_train = growth_curve(train_data['miesiac'], k, a, b)
mse_train = np.mean((train_data['suma_bledow'] - predicted_train) ** 2)
print(f"\nBłąd średniokwadratowy (MSE) dla zbioru treningowego: {mse_train:.2f}")

# Obliczamy błąd średniokwadratowy (MSE) dla zbioru testowego
predicted_test = growth_curve(test_data['miesiac'], k, a, b)
mse_test = np.mean((test_data['suma_bledow'] - predicted_test) ** 2)
print(f"\nBłąd średniokwadratowy (MSE) dla zbioru testowego: {mse_test:.2f}")


# Obliczamy R² dla obu zbiorów
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


r2_train = r2_score(train_data['suma_bledow'], predicted_train)
r2_test = r2_score(test_data['suma_bledow'], predicted_test)

print(f"\nWspółczynnik determinacji R² dla zbioru treningowego: {r2_train:.4f}")
print(f"Współczynnik determinacji R² dla zbioru testowego: {r2_test:.4f}")
