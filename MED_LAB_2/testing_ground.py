import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.gridspec import GridSpec

# Ustawienia dla wykresów
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'


# Wczytanie danych
df = pd.read_csv('Płatki-sniadaniowe-cereals.txt', sep='\t')

# Definicja zmiennych do analizy
nutrients = ['kalorie', 'cukry', 'weglowodany', 'proteiny', 'tluszcz', 'blonnik']
nutrient_names = ['Kalorie', 'Cukry', 'Węglowodany', 'Proteiny', 'Tłuszcz', 'Błonnik']


# 1. Wykresy skrzypcowe dla każdej półki osobno
def create_detailed_violin_plots():
    # Tworzymy osobny wykres dla każdej półki
    for shelf in sorted(df['Liczba_polek'].unique()):
        shelf_data = df[df['Liczba_polek'] == shelf]

        plt.figure(figsize=(15, 10))
        for i, (nutrient, name) in enumerate(zip(nutrients, nutrient_names)):
            plt.subplot(2, 3, i + 1)
            sns.violinplot(y=nutrient, data=shelf_data, inner='box')
            plt.title(f'{name} - Półka {shelf}')
            plt.ylabel('Wartość')

        plt.suptitle(f'Rozkłady składników odżywczych dla półki {shelf}', y=1.02)
        plt.tight_layout()
        plt.show()


# 2. Wykresy radarowe dla profilu odżywczego każdej półki
def create_radar_plots():
    # Obliczenie średnich wartości dla każdej półki
    shelf_means = df.groupby('Liczba_polek')[nutrients].mean()

    # Normalizacja danych do zakresu 0-1
    shelf_means_norm = (shelf_means - shelf_means.min()) / (shelf_means.max() - shelf_means.min())

    angles = np.linspace(0, 2 * np.pi, len(nutrients), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for shelf in sorted(df['Liczba_polek'].unique()):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        values = shelf_means_norm.loc[shelf].values
        values = np.concatenate((values, [values[0]]))

        ax.plot(angles, values, 'o-', linewidth=2, label=f'Półka {shelf}')
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(nutrient_names)

        plt.title(f'Profil odżywczy - Półka {shelf}')

        # Dodanie oryginalnych wartości
        orig_values = shelf_means.loc[shelf].round(1)
        legend_text = '\n'.join([f'{name}: {value}'
                                 for name, value in zip(nutrient_names, orig_values)])
        plt.figtext(1.1, 0.5, f'Wartości średnie:\n\n{legend_text}',
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.show()


# 3. Analiza składu według producenta
def analyze_by_manufacturer():
    manufacturers = df['producent'].value_counts().head(5).index
    df_filtered = df[df['producent'].isin(manufacturers)]

    plt.figure(figsize=(15, 10))
    for i, (nutrient, name) in enumerate(zip(nutrients, nutrient_names)):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='producent', y=nutrient, data=df_filtered)
        plt.title(f'{name} według producenta')
        plt.xticks(rotation=45)
    plt.suptitle('Analiza składników według producentów', y=1.02)
    plt.tight_layout()
    plt.show()


# 4. Heatmapa średnich wartości
def create_nutrient_heatmap():
    shelf_means = df.groupby('Liczba_polek')[nutrients].mean()
    shelf_means_normalized = (shelf_means - shelf_means.mean()) / shelf_means.std()

    plt.figure(figsize=(12, 6))
    sns.heatmap(shelf_means_normalized.T, annot=shelf_means.T.round(1),
                fmt='.1f', cmap='RdYlBu_r', center=0)
    plt.title('Średnie wartości składników według półek\n(kolory pokazują znormalizowane wartości)')
    plt.xlabel('Numer półki')
    plt.ylabel('Składnik')
    plt.show()


# 5. Wykresy przedziałów ufności
def plot_confidence_intervals():
    plt.figure(figsize=(15, 10))
    for i, (nutrient, name) in enumerate(zip(nutrients, nutrient_names)):
        plt.subplot(2, 3, i + 1)
        sns.barplot(x='Liczba_polek', y=nutrient, data=df, ci=95)
        plt.title(f'Średnia {name} z 95% przedziałem ufności')
        plt.xlabel('Numer półki')
        plt.ylabel(name)
    plt.suptitle('Średnie wartości składników z przedziałami ufności', y=1.02)
    plt.tight_layout()
    plt.show()


# 6. Analiza korelacji
def plot_correlation_analysis():
    corr_matrix = df[nutrients].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, center=0)
    plt.title('Korelacje między składnikami odżywczymi')
    plt.show()


# 7. Szczegółowa analiza statystyczna
def detailed_statistical_analysis():
    print("\nSzczegółowa analiza statystyczna składników:")

    for nutrient, name in zip(nutrients, nutrient_names):
        print(f"\n{name}:")

        # Statystyki opisowe
        stats_by_shelf = df.groupby('Liczba_polek')[nutrient].agg(['count', 'mean', 'std', 'min', 'max'])
        print("\nStatystyki według półek:")
        print(stats_by_shelf.round(2))

        # Test Kruskal-Wallis
        groups = [group for _, group in df.groupby('Liczba_polek')[nutrient]]
        h_stat, p_val = stats.kruskal(*groups)
        print(f"\nTest Kruskal-Wallis:")
        print(f"H-statistic = {h_stat:.2f}")
        print(f"p-value = {p_val:.4f}")


# Wykonanie wszystkich analiz
print("Rozpoczynam kompleksową analizę płatków śniadaniowych...")
create_detailed_violin_plots()
create_radar_plots()
analyze_by_manufacturer()
create_nutrient_heatmap()
plot_confidence_intervals()
plot_correlation_analysis()
detailed_statistical_analysis()

# Podsumowanie analiz
print("\nPodstawowe wnioski z analizy:")
for nutrient in nutrients:
    shelf_means = df.groupby('Liczba_polek')[nutrient].mean()
    max_shelf = shelf_means.idxmax()
    min_shelf = shelf_means.idxmin()
    diff_percent = ((shelf_means[max_shelf] - shelf_means[min_shelf]) /
                    shelf_means[min_shelf] * 100)

    print(f"\n{nutrient.capitalize()}:")
    print(f"- Różnica między półkami: {diff_percent:.1f}%")
    print(f"- Najwyższa średnia (półka {max_shelf}): {shelf_means[max_shelf]:.2f}")
    print(f"- Najniższa średnia (półka {min_shelf}): {shelf_means[min_shelf]:.2f}")