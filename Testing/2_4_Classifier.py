import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
df = pd.read_csv('Płatki-sniadaniowe-cereals.txt', sep='\t')

# Model 1: Klasyfikator dla środkowej półki bazujący na cukrze i kaloriach
print("Model 1: Klasyfikator środkowej półki (cukier i kalorie)")
X_simple = df[['cukry', 'kalorie']]
y_middle = (df['polka_2'] == 1).astype(int)

# Standaryzacja danych dla prostego modelu
scaler_simple = StandardScaler()
X_simple_scaled = scaler_simple.fit_transform(X_simple)

# Prosty RandomForest dla środkowej półki
rf_middle = RandomForestClassifier(n_estimators=200, random_state=42)
scores_middle = cross_val_score(rf_middle, X_simple_scaled, y_middle, cv=5)
print(f"\nDokładność cross-validation (środkowa półka): {scores_middle.mean():.2f} (+/- {scores_middle.std() * 2:.2f})")

# Trenowanie na całym zbiorze
rf_middle.fit(X_simple_scaled, y_middle)

# Wizualizacja klasyfikatora środkowej półki
plt.figure(figsize=(12, 8))

# Tworzenie siatki punktów do wizualizacji granic decyzyjnych
x_min, x_max = X_simple['cukry'].min() - 1, X_simple['cukry'].max() + 1
y_min, y_max = X_simple['kalorie'].min() - 10, X_simple['kalorie'].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 1))

# Przekształcenie punktów siatki
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler_simple.transform(grid_points)
Z = rf_middle.predict_proba(grid_points_scaled)[:, 1]
Z = Z.reshape(xx.shape)

# Rysowanie granic decyzyjnych
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
plt.colorbar(label='Prawdopodobieństwo środkowej półki')

# Rysowanie punktów danych
colors = ['red' if y == 0 else 'blue' for y in y_middle]
plt.scatter(X_simple['cukry'], X_simple['kalorie'], c=colors, alpha=0.8)

plt.xlabel('Zawartość cukru (g)')
plt.ylabel('Kalorie')
plt.title('Klasyfikacja produktów na środkowej półce')
plt.legend(['Inna półka', 'Środkowa półka'])

# Dodanie etykiet dla wybranych produktów
for i, row in df.iterrows():
    if i % 3 == 0:  # Pokazujemy co trzecią etykietę dla czytelności
        plt.annotate(row['nazwa'], (row['cukry'], row['kalorie']))

plt.show()

# Wizualizacja ważności cech
importance_middle = pd.DataFrame({
    'cecha': ['cukry', 'kalorie'],
    'waga': rf_middle.feature_importances_
})
plt.figure(figsize=(8, 4))
sns.barplot(data=importance_middle, x='cecha', y='waga')
plt.title('Ważność cech dla klasyfikacji środkowej półki')
plt.show()

# Model 2: Automatyczny dobór cech dla każdej półki
print("\nModel 2: Automatyczny dobór cech dla poszczególnych półek")

# Wszystkie cechy numeryczne
features = ['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'blonnik', 'sod', 'potas']
X_full = df[features]
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

# Słownik do przechowywania modeli i masek dla każdej półki
shelf_models = {}
shelf_masks = {}
feature_importance_per_shelf = {}

# Analiza każdej półki osobno
for shelf in ['polka_1', 'polka_2', 'polka_3']:
    y_shelf = (df[shelf] == 1).astype(int)

    # Wybór cech
    selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42))
    selector.fit(X_full_scaled, y_shelf)
    selected_mask = selector.get_support()
    shelf_masks[shelf] = selected_mask

    # Wyświetlenie wybranych cech
    selected_features = [feat for feat, selected in zip(features, selected_mask) if selected]
    print(f"\nWybrane cechy dla {shelf}:")
    print(selected_features)

    # Trenowanie modelu na wybranych cechach
    X_selected = X_full_scaled[:, selected_mask]
    rf_shelf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_shelf.fit(X_selected, y_shelf)
    shelf_models[shelf] = rf_shelf

    # Zbieranie ważności cech
    selected_features = [feat for feat, selected in zip(features, selected_mask) if selected]
    feature_importance = pd.DataFrame({
        'cecha': selected_features,
        'waga': rf_shelf.feature_importances_
    })
    feature_importance_per_shelf[shelf] = feature_importance

    # Wizualizacja ważności cech dla każdej półki
    plt.figure(figsize=(10, 4))
    sns.barplot(data=feature_importance, x='cecha', y='waga')
    plt.title(f'Ważność cech dla {shelf}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict_shelf(nutrition_values):
    """
    Przewiduje półkę dla nowego produktu.

    Parametry:
    nutrition_values: słownik z wartościami odżywczymi
    """
    # Predykcja dla środkowej półki (prosty model)
    values_simple = np.array([[
        nutrition_values['cukry'],
        nutrition_values['kalorie']
    ]])
    values_simple_scaled = scaler_simple.transform(values_simple)
    middle_shelf_prob = rf_middle.predict_proba(values_simple_scaled)[0][1]

    # Predykcje dla wszystkich półek (zaawansowany model)
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

    # Predykcje dla każdej półki
    shelf_probabilities = {}
    for shelf in shelf_models.keys():
        X_selected = values_full_scaled[:, shelf_masks[shelf]]
        shelf_probabilities[shelf] = float(shelf_models[shelf].predict_proba(X_selected)[0][1])

    return {
        'prawdopodobienstwo_srodkowa_polka': float(middle_shelf_prob),
        'prawdopodobienstwa_wszystkich_polek': shelf_probabilities,
        'sugerowana_polka': max(shelf_probabilities.items(), key=lambda x: x[1])[0]
    }

# Przykład użycia - płatki podobne do Cocoa Puffs (wysoka zawartość cukru)
przyklad = {
    'kalorie': 110,
    'cukry': 13,
    'weglowodany': 12,
    'proteiny': 1,
    'tluszcz': 1,
    'blonnik': 0,
    'sod': 180,
    'potas': 55
}

print("\nPrzykład predykcji dla nowego produktu (podobnego do Cocoa Puffs):")
print(predict_shelf(przyklad))

# Przykład użycia - płatki podobne do Cheerios (niska zawartość cukru)
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

print("\nPrzykład predykcji dla nowego produktu (podobnego do Cheerios):")
print(predict_shelf(przyklad2))
