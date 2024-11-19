# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFE, SelectKBest, f_regression
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def load_data():
#     columns = ['Density', 'Pct.BF', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
#                'Abdomen', 'Waist', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Bicep',
#                'Forearm', 'Wrist']
#     return pd.read_csv('dane.txt', delim_whitespace=True, names=columns, skiprows=1)
#
#
# def analyze_feature_selection(df):
#     """
#     Implementacja trzech metod selekcji zmiennych:
#     1. Analiza korelacji (eliminacja zmiennych silnie skorelowanych)
#     2. Rekurencyjna eliminacja cech (RFE)
#     3. Selekcja w oparciu o testy statystyczne (SelectKBest)
#     """
#     # Usuniecie zmiennej Density (jest silnie powiązana z Pct.BF)
#     features = df.drop(['Density', 'Pct.BF'], axis=1)
#     target = df['Pct.BF']
#
#     # 1. Analiza korelacji
#     correlation_matrix = features.corr()
#     print("\n=== Macierz korelacji ===")
#
#     # Znajdź pary zmiennych z wysoką korelacją (>0.8)
#     high_corr_pairs = []
#     for i in range(len(correlation_matrix.columns)):
#         for j in range(i):
#             if abs(correlation_matrix.iloc[i, j]) > 0.8:
#                 high_corr_pairs.append(
#                     (correlation_matrix.columns[i],
#                      correlation_matrix.columns[j],
#                      correlation_matrix.iloc[i, j])
#                 )
#
#     print("\nSilnie skorelowane pary zmiennych (|r| > 0.8):")
#     for var1, var2, corr in high_corr_pairs:
#         print(f"{var1} - {var2}: {corr:.3f}")
#
#     # 2. Rekurencyjna eliminacja cech (RFE)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(features)
#
#     rfe = RFE(estimator=LinearRegression(), n_features_to_select=6)
#     rfe = rfe.fit(X_scaled, target)
#
#     selected_features_rfe = features.columns[rfe.support_]
#     print("\n=== Cechy wybrane przez RFE ===")
#     print(selected_features_rfe)
#
#     # 3. Selekcja na podstawie testów statystycznych
#     selector = SelectKBest(score_func=f_regression, k=6)
#     selector.fit(X_scaled, target)
#
#     selected_features_stats = features.columns[selector.get_support()]
#     print("\n=== Cechy wybrane przez SelectKBest ===")
#     print(selected_features_stats)
#
#     # Podsumowanie wybranych zmiennych
#     final_features = list(set(selected_features_rfe) & set(selected_features_stats))
#     print("\n=== Rekomendowane zmienne do modelu ===")
#     print(final_features)
#
#     return final_features, features, target
#
#
# def main():
#     # Wczytanie danych
#     df = load_data()
#
#     # Analiza i selekcja zmiennych
#     selected_features, X, y = analyze_feature_selection(df)
#
#     print("\nUzasadnienie wyboru metody eliminacji zmiennych:")
#     print("""
#     1. Eliminacja na podstawie korelacji:
#        - Usuwamy zmienne silnie skorelowane (>0.8), aby uniknąć współliniowości
#        - Wybieramy zmienną, która ma silniejszą korelację z % tłuszczu
#
#     2. Rekurencyjna eliminacja cech (RFE):
#        - Iteracyjnie usuwa najsłabsze zmienne
#        - Uwzględnia interakcje między zmiennymi
#        - Wybiera optymalny zestaw predyktorów
#
#     3. Selekcja statystyczna (SelectKBest):
#        - Wykorzystuje testy statystyczne (f_regression)
#        - Wybiera zmienne najsilniej związane z % tłuszczu
#        - Niezależna od modelu regresji
#
#     Finalna selekcja:
#     - Wybieramy zmienne, które zostały wskazane przez co najmniej dwie metody
#     - Zapewnia to stabilność i wiarygodność wyboru
#     - Redukuje ryzyko przeuczenia modelu
#     """)
#
#
# if __name__ == "__main__":
#     main()
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import RFE
# from statsmodels.regression.linear_model import OLS
# from statsmodels.tools import add_constant
# import seaborn as sns
#
#
# def load_data():
#     """Wczytanie danych z odpowiednim nagłówkiem"""
#     columns = ['Density', 'Pct.BF', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
#                'Abdomen', 'Waist', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Bicep',
#                'Forearm', 'Wrist']
#     return pd.read_csv('dane.txt', delim_whitespace=True, names=columns, skiprows=1)
#
#
# def plot_correlations(df):
#     """Wizualizacja korelacji ze zmienną objaśnianą"""
#     fig, axs = plt.subplots(4, 4, figsize=(15, 15))
#     for i, (name, X) in enumerate(df.items()):
#         row = i // 4
#         col = i % 4
#         axs[row][col].scatter(X, df['Pct.BF'], alpha=0.5)
#         axs[row][col].set_xlabel(name)
#         axs[row][col].set_ylabel('Pct.BF')
#     plt.tight_layout()
#     plt.show()
#
#
# def backward_elimination(df, y, significance_level=0.05):
#     """Backward Elimination na podstawie p-value"""
#     features = df.copy()
#     while True:
#         X_with_intercept = add_constant(features)
#         model = OLS(y, X_with_intercept).fit()
#         max_p_value = model.pvalues[1:].max()
#
#         if max_p_value > significance_level:
#             max_index = model.pvalues[1:].argmax()
#             removed_feature = features.columns[max_index]
#             print(f"Usuwam zmienną {removed_feature} (p-value: {max_p_value:.4f})")
#             features = features.drop(removed_feature, axis=1)
#         else:
#             break
#     return features.columns
#
#
# def forward_selection(df, y, significance_level=0.05):
#     """Forward Selection na podstawie p-value"""
#     selected_features = []
#     remaining_features = list(df.columns)
#
#     while remaining_features:
#         best_p_value = float('inf')
#         best_feature = None
#
#         for feature in remaining_features:
#             current_features = selected_features + [feature]
#             X_with_intercept = add_constant(df[current_features])
#             model = OLS(y, X_with_intercept).fit()
#             p_value = model.pvalues[-1]  # p-value dla nowo dodanej zmiennej
#
#             if p_value < best_p_value:
#                 best_p_value = p_value
#                 best_feature = feature
#
#         if best_p_value < significance_level:
#             print(f"Dodaję zmienną {best_feature} (p-value: {best_p_value:.4f})")
#             selected_features.append(best_feature)
#             remaining_features.remove(best_feature)
#         else:
#             break
#
#     return selected_features
#
#
# def rfe_selection(X, y, n_features=6):
#     """Rekurencyjna eliminacja cech"""
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
#     rfe = rfe.fit(X_scaled, y)
#
#     selected_features = X.columns[rfe.support_]
#     feature_ranking = pd.DataFrame({
#         'Feature': X.columns,
#         'Ranking': rfe.ranking_
#     }).sort_values('Ranking')
#
#     return selected_features, feature_ranking
#
#
# def main():
#     # Wczytanie danych
#     df = load_data()
#
#     # Wizualizacja korelacji
#     print("Generowanie wykresów korelacji...")
#     plot_correlations(df)
#
#     # Przygotowanie danych
#     X = df.drop(['Pct.BF', 'Density'], axis=1)  # Usuwamy Density jako silnie skorelowaną
#     y = df['Pct.BF']
#
#     print("\n=== Metoda 1: Backward Elimination ===")
#     backward_features = backward_elimination(X, y)
#     print("Wybrane cechy:", backward_features)
#
#     print("\n=== Metoda 2: Forward Selection ===")
#     forward_features = forward_selection(X, y)
#     print("Wybrane cechy:", forward_features)
#
#     print("\n=== Metoda 3: Recursive Feature Elimination ===")
#     rfe_features, feature_ranking = rfe_selection(X, y)
#     print("Wybrane cechy:", rfe_features)
#     print("\nRanking cech:")
#     print(feature_ranking)
#
#     # Porównanie wyników
#     print("\n=== Podsumowanie wybranych cech ===")
#     all_selected_features = set(backward_features) | set(forward_features) | set(rfe_features)
#     feature_counts = {feature: sum([
#         feature in backward_features,
#         feature in forward_features,
#         feature in rfe_features
#     ]) for feature in all_selected_features}
#
#     print("\nLiczba metod wybierających daną cechę:")
#     for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"{feature}: {count}")
#
#     # Rekomendacja końcowa
#     recommended_features = [f for f, c in feature_counts.items() if c >= 2]
#     print("\nRekomendowane cechy (wybrane przez co najmniej 2 metody):")
#     print(recommended_features)
#
#
# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def load_data():
    columns = ['Density', 'Pct.BF', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
               'Abdomen', 'Waist', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Bicep',
               'Forearm', 'Wrist']
    return pd.read_csv('dane.txt', sep='\s+', names=columns, skiprows=1)


def calculate_model_stats(y_hat, y_true, X):
    X = np.insert(X, 0, 1, axis=1)
    residuals = y_true - y_hat
    residual_sum_of_squares = np.sum(residuals ** 2)
    standard_variance = residual_sum_of_squares / (y_hat.shape[0] - X.shape[1])
    model_coefs_variances = standard_variance * np.linalg.inv(X.T @ X)

    stats = {}
    for number in range(model_coefs_variances.shape[0]):
        stats[f'Standard error a{number}'] = np.sqrt(model_coefs_variances[number, number])
    stats['Standard error y'] = np.sqrt(standard_variance)
    return stats


def plot_density_regression(df):
    """Wizualizacja regresji liniowej dla Density vs Pct.BF"""
    D = df['Density']
    B = df['Pct.BF']

    # Split danych
    D_train, D_test, B_train, B_test = train_test_split(D, B, test_size=0.2, random_state=42)

    # Trenowanie modelu
    model = LinearRegression()
    model.fit(D_train.values.reshape(-1, 1), B_train)

    # Obliczenie R^2
    score = model.score(D_test.values.reshape(-1, 1), B_test)

    # Obliczenie błędów
    errors = calculate_model_stats(
        model.predict(D_train.values.reshape(-1, 1)),
        B_train.values.reshape(-1, 1),
        D_train.values.reshape(-1, 1)
    )
    B_error = errors['Standard error y']

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))

    # Generowanie linii predykcji
    D_range = np.linspace(min(D), max(D), 100).reshape(-1, 1)
    B_pred = model.predict(D_range)

    # Przedział ufności
    plt.fill_between(
        D_range.reshape(-1),
        (B_pred - B_error).reshape(-1),
        (B_pred + B_error).reshape(-1),
        color='yellow',
        alpha=0.3,
        label='Przedział ufności'
    )

    # Dane treningowe i testowe
    plt.scatter(D_train, B_train, color='blue', label='Dane treningowe', alpha=0.6)
    plt.scatter(D_test, B_test, color='red', label='Dane testowe', alpha=0.6)

    # Linia regresji
    plt.plot(D_range, B_pred, color='green', linewidth=2, label=f'Regresja (R² = {score:.4f})')

    plt.xlabel('Gęstość (Density)')
    plt.ylabel('Procent tłuszczu (Pct.BF)')
    plt.title('Regresja liniowa: Gęstość vs Procent tłuszczu')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model.coef_[0], model.intercept_, score


def plot_correlations(df):
    """Wizualizacja korelacji ze zmienną objaśnianą"""
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    for i, (name, X) in enumerate(df.items()):
        row = i // 4
        col = i % 4
        axs[row][col].scatter(X, df['Pct.BF'], alpha=0.5)
        axs[row][col].set_xlabel(name)
        axs[row][col].set_ylabel('Pct.BF')
    plt.tight_layout()
    plt.show()


def backward_elimination(df, y, significance_level=0.05):
    """Backward Elimination na podstawie p-value"""
    features = df.copy()
    while True:
        X_with_intercept = add_constant(features)
        model = OLS(y, X_with_intercept).fit()
        max_p_value = model.pvalues[1:].max()

        if max_p_value > significance_level:
            max_index = model.pvalues[1:].argmax()
            removed_feature = features.columns[max_index]
            print(f"Usuwam zmienną {removed_feature} (p-value: {max_p_value:.4f})")
            features = features.drop(removed_feature, axis=1)
        else:
            break
    return features.columns


def forward_selection(df, y, significance_level=0.05):
    """Forward Selection na podstawie p-value"""
    selected_features = []
    remaining_features = list(df.columns)

    while remaining_features:
        best_p_value = float('inf')
        best_feature = None

        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_with_intercept = add_constant(df[current_features])
            model = OLS(y, X_with_intercept).fit()
            p_value = model.pvalues[-1]  # p-value dla nowo dodanej zmiennej

            if p_value < best_p_value:
                best_p_value = p_value
                best_feature = feature

        if best_p_value < significance_level:
            print(f"Dodaję zmienną {best_feature} (p-value: {best_p_value:.4f})")
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    return selected_features


def rfe_selection(X, y, n_features=6):
    """Rekurencyjna eliminacja cech"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
    rfe = rfe.fit(X_scaled, y)

    selected_features = X.columns[rfe.support_]
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')

    return selected_features, feature_ranking


def main():
    # Wczytanie danych i podstawowa analiza
    df = load_data()

    # Wykonanie regresji i wyświetlenie wykresu
    coef, intercept, r2 = plot_density_regression(df)
    print(f"\nRegresja liniowa Density vs Pct.BF:")
    print(f"Współczynnik kierunkowy (a): {coef:.4f}")
    print(f"Wyraz wolny (b): {intercept:.4f}")
    print(f"R²: {r2:.4f}")

    # Wizualizacja korelacji
    print("Generowanie wykresów korelacji...")
    plot_correlations(df)

    # Przygotowanie danych
    X = df.drop(['Pct.BF', 'Density'], axis=1)  # Usuwamy Density jako silnie skorelowaną
    y = df['Pct.BF']

    print("\n=== Metoda 1: Backward Elimination ===")
    backward_features = backward_elimination(X, y)
    print("Wybrane cechy:", backward_features)

    print("\n=== Metoda 2: Forward Selection ===")
    forward_features = forward_selection(X, y)
    print("Wybrane cechy:", forward_features)

    print("\n=== Metoda 3: Recursive Feature Elimination ===")
    rfe_features, feature_ranking = rfe_selection(X, y)
    print("Wybrane cechy:", rfe_features)
    print("\nRanking cech:")
    print(feature_ranking)

    # Porównanie wyników
    print("\n=== Podsumowanie wybranych cech ===")
    all_selected_features = set(backward_features) | set(forward_features) | set(rfe_features)
    feature_counts = {feature: sum([
        feature in backward_features,
        feature in forward_features,
        feature in rfe_features
    ]) for feature in all_selected_features}

    print("\nLiczba metod wybierających daną cechę:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {count}")

    # Rekomendacja końcowa
    recommended_features = [f for f, c in feature_counts.items() if c >= 2]
    print("\nRekomendowane cechy (wybrane przez co najmniej 2 metody):")
    print(recommended_features)


if __name__ == "__main__":
    main()