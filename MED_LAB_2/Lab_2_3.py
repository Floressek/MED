import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Definiujemy funkcję logistyczną (krzywą wzrostu)
def growth_curve(t, k, a, b):
    """
    Funkcja logistyczna modelująca skumulowany wzrost
    t: zmienna czasowa (numer miesiąca)
    k: maksymalna wartość (asymptota)
    a: współczynnik wzrostu
    b: parametr przesunięcia
    """
    return k / (1 + b * np.exp(-a * t))


# Tworzymy przykładowe dane (możesz zastąpić swoimi)
data = pd.DataFrame({
    'miesiac': range(1, 94),  # 93 miesiące
    'liczba_bledow': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 0, 0, 3, 1, 1, 0, 1, 0, 2,
                      10, 0, 16, 0, 2, 2, 1, 1, 1, 0, 3, 2, 1, 6, 3, 0, 1, 1, 0, 0, 14, 1, 4, 1, 1, 7, 14, 6, 0,
                      1, 2, 19, 7, 2, 5, 1, 12, 4, 6, 4, 7, 2, 2, 3, 8, 4, 6, 3, 3, 5, 8, 0, 6, 5, 10, 0, 5, 1, 2, 1, 2,
                      1, 0]
})

# Obliczamy skumulowaną sumę błędów
data['suma_bledow'] = data['liczba_bledow'].cumsum()

# # Dopasowujemy model do danych
# popt, pcov = curve_fit(growth_curve,
#                        data['miesiac'],
#                        data['suma_bledow'],
#                        p0=[300, 0.1, 90],  # początkowe przybliżenia parametrów
#                        bounds=([0, 0, 0], [1000, 1, 1000]))  # ograniczenia parametrów
#

# Testujemy różne wartości początkowe b
initial_b_values = [5, 90, 999]
for b_init in initial_b_values:
    popt, _ = curve_fit(growth_curve,
                        data['miesiac'],
                        data['suma_bledow'],
                        p0=[300, 0.1, b_init],
                        bounds=([0, 0, 0], [1000, 1, 1000]))

    k, a, b = popt
    print(f"\nDla początkowego b = {b_init}:")
    print(f"Końcowe parametry:")
    print(f"k = {k:.2f}")
    print(f"a = {a:.6f}")
    print(f"b = {b:.2f}")

# # Wyświetlamy znalezione parametry
# k, a, b = popt
# print(f"Znalezione parametry:")
# print(f"k (maksymalna liczba błędów): {k:.2f}")
# print(f"a (współczynnik wzrostu): {a:.6f}")
# print(f"b (parametr przesunięcia): {b:.2f}")

# Tworzymy wykres
plt.figure(figsize=(12, 6))
plt.scatter(data['miesiac'], data['suma_bledow'], label='Rzeczywiste dane', color='blue', alpha=0.5)
t = np.linspace(1, 93, 1000)
plt.plot(t, growth_curve(t, k, a, b), 'r-', label='Model', linewidth=2)
plt.xlabel('Numer miesiąca')
plt.ylabel('Sumaryczna liczba błędów')
plt.title('Modelowanie wzrostu liczby błędów w systemie')
plt.legend()
plt.grid(True)
plt.show()

# Obliczamy błąd średniokwadratowy (MSE)
predicted = growth_curve(data['miesiac'], k, a, b)
mse = np.mean((data['suma_bledow'] - predicted) ** 2)
print(f"\nBłąd średniokwadratowy (MSE): {mse:.2f}")
