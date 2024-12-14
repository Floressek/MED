import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


class DataValidator:
    @staticmethod
    def validate_data(data):
        """Sprawdza poprawność danych wejściowych"""
        # Sprawdzanie braków danych
        missing = data.isnull().sum()
        if missing.any():
            print("Ostrzeżenie: Znaleziono brakujące wartości:")
            print(missing[missing > 0])

        # Sprawdzanie wartości odstających (outliers)
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = (z_scores > 3).sum()
        if outliers.any():
            print("\nOstrzeżenie: Znaleziono potencjalne wartości odstające:")
            print(outliers[outliers > 0])

        # Sprawdzanie zakresów wartości
        range_checks = {
            'Pct.BF': (0, 100),  # procent tłuszczu między 0-100%
            'Age': (0, 120),  # wiek między 0-120 lat
            'Weight': (30, 300),  # waga między 30-300 kg/pounds
            'Height': (120, 220),  # wzrost między 120-220 cm
        }

        for column, (min_val, max_val) in range_checks.items():
            if column in data.columns:
                invalid = data[(data[column] < min_val) | (data[column] > max_val)]
                if not invalid.empty:
                    print(f"\nOstrzeżenie: Znaleziono wartości poza zakresem dla {column}:")
                    print(f"Min: {data[column].min()}, Max: {data[column].max()}")

        # Sprawdzanie korelacji
        correlations = data.corr()
        high_corr = (np.abs(correlations) > 0.9) & (np.abs(correlations) < 1.0)
        if high_corr.any().any():
            print("\nOstrzeżenie: Znaleziono silnie skorelowane zmienne:")
            for col in correlations.columns:
                highly_corr = correlations[col][high_corr[col]].index.tolist()
                if highly_corr:
                    print(f"{col} jest silnie skorelowane z: {highly_corr}")


def load_data_from_document(doc_content):
    """Wczytuje dane z dokumentu i przeprowadza podstawową walidację"""
    # Definiujemy nazwy kolumn
    columns = ['Density', 'Pct.BF', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Waist',
               'Hip', 'Thigh', 'Knee', 'Ankle', 'Bicep', 'Forearm', 'Wrist']

    # Wczytujemy linie z dokumentu, pomijając nagłówek
    lines = doc_content.strip().split('\n')
    data = []

    for line in lines:
        if line and not line.startswith('Density'):
            try:
                # Próbujemy przekonwertować wartości na liczby
                values = [float(x) for x in line.split()]
                if len(values) == len(columns):
                    data.append(values)
                else:
                    print(f"Ostrzeżenie: Pominięto linię z nieprawidłową liczbą kolumn: {line}")
            except ValueError as e:
                print(f"Ostrzeżenie: Pominięto linię z nieprawidłowymi danymi: {line}")
                print(f"Błąd: {e}")

    # Tworzymy DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Przeprowadzamy walidację
    validator = DataValidator()
    validator.validate_data(df)

    return df


class StepwiseSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.features = list(X.columns)

    def _calculate_pvalues(self, X, y):
        """Oblicza p-wartości dla zmiennych w modelu"""
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.pvalues[1:], model.rsquared

    def forward_selection(self, threshold=0.05):
        """Forward Selection"""
        selected = []
        remaining = self.features.copy()
        current_score = 0

        print("\nForward Selection:")
        print("------------------")

        while remaining and len(selected) < len(self.features):
            best_pval = 999
            best_feature = None
            best_score = current_score

            for feature in remaining:
                features_to_test = selected + [feature]
                X_test = self.X[features_to_test]
                pvalues, score = self._calculate_pvalues(X_test, self.y)
                pval = pvalues[feature]

                if pval < best_pval and score > best_score:
                    best_pval = pval
                    best_feature = feature
                    best_score = score

            if best_feature and best_pval < threshold:
                selected.append(best_feature)
                remaining.remove(best_feature)
                current_score = best_score
                print(f"Dodano: {best_feature} (p-value: {best_pval:.4f}, R²: {best_score:.4f})")
            else:
                break

        return selected

    def backward_elimination(self, threshold=0.05):
        """Backward Elimination"""
        selected = self.features.copy()

        print("\nBackward Elimination:")
        print("---------------------")

        while selected:
            pvalues, score = self._calculate_pvalues(self.X[selected], self.y)
            max_pval = pvalues.max()
            if max_pval > threshold:
                worst_feature = pvalues.idxmax()
                selected.remove(worst_feature)
                print(f"Usunięto: {worst_feature} (p-value: {max_pval:.4f}, R²: {score:.4f})")
            else:
                break

        return selected

    def stepwise_selection(self, threshold_in=0.05, threshold_out=0.05):
        """Stepwise Selection"""
        selected = []
        remaining = self.features.copy()

        print("\nStepwise Selection:")
        print("-------------------")

        while True:
            # Forward step
            best_pval = 999
            best_feature = None
            best_score = 0

            for feature in remaining:
                features_to_test = selected + [feature]
                X_test = self.X[features_to_test]
                pvalues, score = self._calculate_pvalues(X_test, self.y)
                pval = pvalues[feature]

                if pval < best_pval and pval < threshold_in:
                    best_pval = pval
                    best_feature = feature
                    best_score = score

            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                print(f"Dodano: {best_feature} (p-value: {best_pval:.4f}, R²: {best_score:.4f})")

                # Backward step
                while len(selected) > 1:
                    pvalues, score = self._calculate_pvalues(self.X[selected], self.y)
                    max_pval = pvalues.max()
                    if max_pval > threshold_out:
                        worst_feature = pvalues.idxmax()
                        selected.remove(worst_feature)
                        print(f"Usunięto: {worst_feature} (p-value: {max_pval:.4f}, R²: {score:.4f})")
                    else:
                        break
            else:
                break

        return selected

    def compare_methods(self):
        """Porównuje wyniki wszystkich trzech metod"""
        print("\nPorównanie metod selekcji zmiennych:")
        print("=====================================")

        forward = self.forward_selection()
        backward = self.backward_elimination()
        stepwise = self.stepwise_selection()

        results = pd.DataFrame({
            'Forward': [feature in forward for feature in self.features],
            'Backward': [feature in backward for feature in self.features],
            'Stepwise': [feature in stepwise for feature in self.features]
        }, index=self.features)

        print("\nWybrane zmienne przez każdą metodę:")
        print(results)

        # Ocena modeli
        models_r2 = {}
        for name, selected in [('Forward', forward), ('Backward', backward), ('Stepwise', stepwise)]:
            if selected:  # sprawdzamy czy lista nie jest pusta
                X_selected = self.X[selected]
                model = LinearRegression()
                model.fit(X_selected, self.y)
                r2 = r2_score(self.y, model.predict(X_selected))
                models_r2[name] = r2
            else:
                print(f"\nOstrzeżenie: Metoda {name} nie wybrała żadnych zmiennych!")
                models_r2[name] = 0

        print("\nWyniki modeli (R²):")
        for method, r2 in models_r2.items():
            print(f"{method}: {r2:.4f}")

        return results, models_r2


# Użycie
if __name__ == "__main__":
    # Wczytanie danych z pliku
    try:
        # Wersja z pandas - jeśli dane są w formacie csv
        data = pd.read_csv(r"C:\Users\szyme\PycharmProjects\MED\MED_LAB_1\data\dane.txt", delim_whitespace=True)

        print(f"\nWczytano dane o wymiarach: {data.shape}")

        # Wybór zmiennych do analizy
        features = [col for col in data.columns if col not in ['Density', 'Pct.BF']]
        # features = [col for col in data.columns if col not in ['Pct.BF']]
        X = data[features]
        y = data['Pct.BF']

        # Analiza
        selector = StepwiseSelector(X, y)
        results, scores = selector.compare_methods()

    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku z danymi!")
    except pd.errors.EmptyDataError:
        print("Błąd: Plik z danymi jest pusty!")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")
