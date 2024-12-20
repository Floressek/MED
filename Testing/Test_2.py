import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Ustawienia dla lepszej czytelności wykresów

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Wczytanie danych
df = pd.read_csv('Płatki-sniadaniowe-cereals.txt', sep='\t')

print("=== Analiza danych wejściowych ===")
print(f"\nLiczba produktów: {len(df)}")
print("\nRozkład produktów na półkach:")
print(df[['polka_1', 'polka_2', 'polka_3']].sum())
print("\nStatystyki dla głównych składników:")
print(df[['kalorie', 'cukry', 'weglowodany', 'proteiny']].describe())

# Model 1: Klasyfikator dla środkowej półki
print("\n=== Model 1: Klasyfikator środkowej półki (cukier i kalorie) ===")
X_simple = df[['cukry', 'kalorie']]
y_middle = (df['polka_2'] == 1).astype(int)

# Standaryzacja i trenowanie modelu
scaler_simple = StandardScaler()
scaler_simple.fit(X_simple)
X_simple_scaled = scaler_simple.transform(X_simple)

rf_middle = RandomForestClassifier(n_estimators=200, random_state=42)
scores_middle = cross_val_score(rf_middle, X_simple_scaled, y_middle, cv=5)
print(f"\nDokładność cross-validation: {scores_middle.mean():.2f} (+/- {scores_middle.std() * 2:.2f})")

# Trenowanie na całym zbiorze
rf_middle.fit(X_simple_scaled, y_middle)
y_pred_middle = rf_middle.predict(X_simple_scaled)

print("\nRaport klasyfikacji dla środkowej półki:")
print(classification_report(y_middle, y_pred_middle))

# # Wizualizacja macierzy pomyłek dla środkowej półki
# plt.figure(figsize=(8, 6))
# conf_matrix = confusion_matrix(y_middle, y_pred_middle)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Macierz pomyłek dla środkowej półki')
# plt.xlabel('Przewidziana wartość')
# plt.ylabel('Prawdziwa wartość')
# plt.show()


# Macierz pomyłek z lepszymi opisami
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_middle, y_pred_middle)

# Tworzymy DataFrame z czytelnymi etykietami
df_cm = pd.DataFrame(
    conf_matrix,
    index=['Nie na środkowej', 'Na środkowej'],
    columns=['Przewidziano: Nie', 'Przewidziano: Tak']
)

# Tworzenie mapy ciepła
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')

# Dodajemy dodatkowe opisy do każdego pola
for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            plt.text(j+0.5, i+0.2, 'Prawdziwe Negatywne (TN)',
                    ha='center', va='center', fontsize=8)
        elif i == 0 and j == 1:
            plt.text(j+0.5, i+0.2, 'Fałszywe Pozytywne (FP)',
                    ha='center', va='center', fontsize=8)
        elif i == 1 and j == 0:
            plt.text(j+0.5, i+0.2, 'Fałszywe Negatywne (FN)',
                    ha='center', va='center', fontsize=8)
        elif i == 1 and j == 1:
            plt.text(j+0.5, i+0.2, 'Prawdziwe Pozytywne (TP)',
                    ha='center', va='center', fontsize=8)

plt.title('Macierz pomyłek dla środkowej półki')
plt.ylabel('Prawdziwa wartość')
plt.xlabel('Przewidziana wartość')

# Dodanie legendy z objaśnieniami
plt.figtext(1.1, 0.8, 'Objaśnienia:', fontsize=10, fontweight='bold')
plt.figtext(1.1, 0.7, 'TN: Model poprawnie przewidział, że produkt NIE jest na środkowej półce', fontsize=8)
plt.figtext(1.1, 0.6, 'FP: Model błędnie przewidział, że produkt JEST na środkowej półce', fontsize=8)
plt.figtext(1.1, 0.5, 'FN: Model błędnie przewidział, że produkt NIE jest na środkowej półce', fontsize=8)
plt.figtext(1.1, 0.4, 'TP: Model poprawnie przewidział, że produkt JEST na środkowej półce', fontsize=8)

plt.tight_layout()
plt.show()

# Dodatkowo wyświetlamy metryki w konsoli
print("\nMetryki klasyfikacji:")
print(f"Dokładność (Accuracy) = {(conf_matrix[0,0] + conf_matrix[1,1])/conf_matrix.sum():.2f}")
print(f"Precyzja (Precision) = {conf_matrix[1,1]/(conf_matrix[0,1] + conf_matrix[1,1]):.2f}")
print(f"Czułość (Recall) = {conf_matrix[1,1]/(conf_matrix[1,0] + conf_matrix[1,1]):.2f}")

# Wizualizacja klasyfikatora środkowej półki
plt.figure(figsize=(15, 10))

# Granice decyzyjne
x_min, x_max = X_simple['cukry'].min() - 1, X_simple['cukry'].max() + 1
y_min, y_max = X_simple['kalorie'].min() - 10, X_simple['kalorie'].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 2))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler_simple.transform(grid_points)
Z = rf_middle.predict_proba(grid_points_scaled)[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), alpha=0.6, cmap='RdYlBu')
plt.colorbar(label='Prawdopodobieństwo środkowej półki')

# Rysowanie punktów danych
plt.scatter(X_simple['cukry'][y_middle == 0], X_simple['kalorie'][y_middle == 0],
            c='red', label='Inna półka', alpha=0.8)
plt.scatter(X_simple['cukry'][y_middle == 1], X_simple['kalorie'][y_middle == 1],
            c='blue', label='Środkowa półka', alpha=0.8)

plt.xlabel('Zawartość cukru (g)')
plt.ylabel('Kalorie')
plt.title('Klasyfikacja produktów na środkowej półce')

# Dodanie etykiet dla wybranych produktów
mask_high_prob = Z.ravel() > 0.7
for i, row in df.iterrows():
    point_scaled = scaler_simple.transform([[row['cukry'], row['kalorie']]])
    prob = rf_middle.predict_proba(point_scaled)[0][1]
    if prob > 0.7 or prob < 0.3:
        plt.annotate(row['nazwa'],
                     (row['cukry'], row['kalorie']),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.legend()
plt.tight_layout()
plt.show()

# Model 2: Automatyczny dobór cech dla każdej półki
print("\n=== Model 2: Klasyfikatory dla poszczególnych półek ===")
features = ['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'blonnik', 'sod', 'potas']
X_full = df[features]
scaler_full = StandardScaler()
scaler_full.fit(X_full)
X_full_scaled = scaler_full.transform(X_full)

# Inicjalizacja słowników
shelf_models = {}
shelf_masks = {}

# Analiza każdej półki
for shelf_num, shelf in enumerate(['polka_1', 'polka_2', 'polka_3'], 1):
    print(f"\n--- Analiza półki {shelf_num} ---")
    y_shelf = (df[shelf] == 1).astype(int)

    # Wybór cech
    selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42))
    selector.fit(X_full_scaled, y_shelf)
    selected_mask = selector.get_support()
    shelf_masks[shelf] = selected_mask

    # Wyświetlenie wybranych cech
    selected_features = [feat for feat, selected in zip(features, selected_mask) if selected]
    print(f"Wybrane cechy: {selected_features}")

    # Trenowanie modelu
    X_selected = X_full_scaled[:, selected_mask]
    rf_shelf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Cross-validation
    scores = cross_val_score(rf_shelf, X_selected, y_shelf, cv=5)
    print(f"Dokładność cross-validation: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # Trenowanie na całym zbiorze
    rf_shelf.fit(X_selected, y_shelf)
    shelf_models[shelf] = rf_shelf
    y_pred = rf_shelf.predict(X_selected)

    print("\nRaport klasyfikacji:")
    print(classification_report(y_shelf, y_pred))

    # Wizualizacja ważności cech
    plt.figure(figsize=(10, 5))
    feature_importance = pd.DataFrame({
        'cecha': selected_features,
        'waga': rf_shelf.feature_importances_
    })
    sns.barplot(data=feature_importance, x='cecha', y='waga')
    plt.title(f'Ważność cech dla półki {shelf_num}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Macierz pomyłek
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_shelf, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Macierz pomyłek dla półki {shelf_num}')
    plt.xlabel('Przewidziana wartość')
    plt.ylabel('Prawdziwa wartość')
    plt.show()


def predict_shelf(nutrition_values):
    """
    Przewiduje półkę dla nowego produktu.
    """
    # Predykcja dla środkowej półki
    values_simple = np.array([[
        nutrition_values['cukry'],
        nutrition_values['kalorie']
    ]])
    values_simple_scaled = scaler_simple.transform(values_simple)
    middle_shelf_prob = rf_middle.predict_proba(values_simple_scaled)[0][1]

    # Predykcje dla wszystkich półek
    values_full = np.array([[
        nutrition_values.get('kalorie', 0),
        nutrition_values.get('cukry', 0),
        nutrition_values.get('weglowodany', 0),
        nutrition_values.get('proteiny', 0),
        nutrition_values.get('tluszcz', 0),
        nutrition_values.get('blonnik', 0),
        nutrition_values.get('sod', 0),
        nutrition_values.get('potas', 0)
    ]])
    values_full_scaled = scaler_full.transform(values_full)

    shelf_probabilities = {}
    for shelf in shelf_models.keys():
        X_selected = values_full_scaled[:, shelf_masks[shelf]]
        shelf_probabilities[shelf] = float(shelf_models[shelf].predict_proba(X_selected)[0][1])

    return {
        'prawdopodobienstwo_srodkowa_polka': float(middle_shelf_prob),
        'prawdopodobienstwa_wszystkich_polek': shelf_probabilities,
        'sugerowana_polka': max(shelf_probabilities.items(), key=lambda x: x[1])[0]
    }


# Przykłady użycia
print("\n=== Testowanie klasyfikatorów ===")

# Przykład 1: Płatki podobne do Cocoa Puffs
przyklad1 = {
    'kalorie': 110,
    'cukry': 13,
    'weglowodany': 12,
    'proteiny': 1,
    'tluszcz': 1,
    'blonnik': 0,
    'sod': 180,
    'potas': 55
}

print("\nPrzykład 1 (płatki podobne do Cocoa Puffs):")
print("Wartości odżywcze:", przyklad1)
wynik1 = predict_shelf(przyklad1)
print("Wyniki predykcji:")
print(f"- Prawdopodobieństwo środkowej półki: {wynik1['prawdopodobienstwo_srodkowa_polka']:.2f}")
print("- Prawdopodobieństwa dla wszystkich półek:")
for polka, prob in wynik1['prawdopodobienstwa_wszystkich_polek'].items():
    print(f"  * {polka}: {prob:.2f}")
print(f"- Sugerowana półka: {wynik1['sugerowana_polka']}")

# Przykład 2: Płatki podobne do Cheerios
przyklad2 = {
    'kalorie': 110,
    'cukry': 1,
    'weglowodany': 17,
    'proteiny': 6,
    'tluszcz': 2,
    'blonnik': 2,
    'sod': 290,
    'potas': 105
}

print("\nPrzykład 2 (płatki podobne do Cheerios):")
print("Wartości odżywcze:", przyklad2)
wynik2 = predict_shelf(przyklad2)
print("Wyniki predykcji:")
print(f"- Prawdopodobieństwo środkowej półki: {wynik2['prawdopodobienstwo_srodkowa_polka']:.2f}")
print("- Prawdopodobieństwa dla wszystkich półek:")
for polka, prob in wynik2['prawdopodobienstwa_wszystkich_polek'].items():
    print(f"  * {polka}: {prob:.2f}")
print(f"- Sugerowana półka: {wynik2['sugerowana_polka']}")