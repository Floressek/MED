import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import re
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

import warnings

warnings.filterwarnings("ignore")


class KlasyfikatorTweetowMandrill:
    def __init__(self, usun_duplikaty=False):
        self.usun_duplikaty = usun_duplikaty
        # Definiujemy własną listę stop words
        self.stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                           'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                           'that', 'the', 'to', 'was', 'were', 'will', 'with', 'the',
                           'this', 'but', 'they', 'have', 'had', 'what', 'when',
                           'where', 'who', 'which', 'why', 'can', 'could', 'should',
                           'would', 'may', 'might', 'must', 'shall', 'his', 'her',
                           'their', 'our', 'my', 'your', 'we', 'you', 'i', 'am']

        self.vectorizer = CountVectorizer(
            stop_words=self.stop_words,
            ngram_range=(1, 2),
            min_df=2
        )
        self.classifier = MultinomialNB()
        self.translator = GoogleTranslator(source='auto', target='en')

    def wykryj_jezyk(self, text):
        try:
            return detect(str(text))
        except LangDetectException:
            return 'unknown'

    def tlumacz_tekst(self, text, lang):
        """Tłumaczy tekst na angielski, jeśli nie jest po angielsku"""
        import time
        if lang != 'en' and lang != 'unknown':
            try:
                time.sleep(1)  # opóźnienie 1 sekunda między tłumaczeniami
                return self.translator.translate(text)
            except Exception as e:
                print(f"\nBłąd tłumaczenia: {e}")
                return text
        return text

    def preprocessuj_tekst(self, text):
        if pd.isna(text):
            return ""
        text = str(text)

        # Zachowujemy help.mandrill.com w tekście
        text = re.sub(r'http://help\.mandrill\.com', ' MANDRILL_HELP_URL ', text)
        text = re.sub(r'https://help\.mandrill\.com', ' MANDRILL_HELP_URL ', text)

        # Pozostałe preprocessing
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', 'USER', text)  # zamieniamy @username na USER
        text = re.sub(r'[^\w\s\'\-àáâãäçèéêëìíîïñòóôõöùúûüýÿ]', ' ', text)

        # Usuwamy stop words
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        text = ' '.join(words)

        return text

    def przygotuj_dane(self, plik_excel):
        df_mandrill = pd.read_excel(plik_excel, sheet_name=0)
        df_inne = pd.read_excel(plik_excel, sheet_name=1)

        print("Kolumny w pierwszym arkuszu:")
        print(df_mandrill.columns.tolist())
        print("\nKolumny w drugim arkuszu:")
        print(df_inne.columns.tolist())

        tekst_kolumna = 'Post'

        df_mandrill['etykieta'] = 1
        df_inne['etykieta'] = 0

        df = pd.concat([df_mandrill, df_inne], ignore_index=True)

        # Wykrywamy język przed preprocessingiem
        from tqdm import tqdm
        print("\nWykrywanie języków...")
        tqdm.pandas(desc="Wykrywanie języka")
        df['jezyk'] = df[tekst_kolumna].progress_apply(self.wykryj_jezyk)

        print("\nTłumaczenie tekstów na angielski...")
        tqdm.pandas(desc="Tłumaczenie")
        df['tekst_eng'] = df.progress_apply(lambda row:
                                            self.tlumacz_tekst(row[tekst_kolumna], row['jezyk']), axis=1)

        # Preprocessujemy przetłumaczony tekst
        df['przetworzony_tekst'] = df['tekst_eng'].apply(self.preprocessuj_tekst)

        if self.usun_duplikaty:
            print(f"\nLiczba wierszy przed usunięciem duplikatów: {len(df)}")
            df = df.drop_duplicates(subset=['przetworzony_tekst'])
            print(f"Liczba wierszy po usunięciu duplikatów: {len(df)}")

        print("\nRozkład języków w zbiorze danych:")
        print(df['jezyk'].value_counts())

        print("\nRozkład klas:")
        print(df['etykieta'].value_counts().sort_index())

        return df

    def plot_confusion_matrix(self, cm, title="Macierz pomyłek"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Inne', 'Mandrill'],
                    yticklabels=['Inne', 'Mandrill'])
        plt.title(title)
        plt.ylabel('Prawdziwa klasa')
        plt.xlabel('Przewidywana klasa')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names, importance_scores, n_features=20):
        # Sort features by absolute importance
        indices = np.argsort(np.abs(importance_scores))[-n_features:]

        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(n_features), importance_scores[indices])
        plt.yticks(range(n_features), feature_names[indices])
        plt.xlabel('Ważność cechy (log probability difference)')
        plt.title('Najważniejsze cechy w klasyfikacji')

        # Color bars based on positive/negative importance
        for i, bar in enumerate(bars):
            if importance_scores[indices[i]] < 0:
                bar.set_color('lightcoral')
            else:
                bar.set_color('lightblue')

        plt.tight_layout()
        plt.show()

    def plot_prediction_confidence(self, y_test, y_pred, probabilities):
        plt.figure(figsize=(10, 6))

        # Get max probability for each prediction
        confidence = np.max(probabilities, axis=1)

        # Create separate lists for correct and incorrect predictions
        correct_conf = confidence[y_test == y_pred]
        incorrect_conf = confidence[y_test != y_pred]

        # Plot histograms
        plt.hist(correct_conf, alpha=0.5, label='Poprawne przewidywania',
                 bins=20, color='green')
        plt.hist(incorrect_conf, alpha=0.5, label='Błędne przewidywania',
                 bins=20, color='red')

        plt.xlabel('Pewność przewidywania')
        plt.ylabel('Liczba przewidywań')
        plt.title('Rozkład pewności przewidywań')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def trenuj(self, df):
        df = df[df['przetworzony_tekst'].str.len() > 0].copy()

        X = self.vectorizer.fit_transform(df['przetworzony_tekst'])
        y = df['etykieta']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        test_indices = np.arange(len(df))[X_test.nonzero()[0]]
        test_texts = df.iloc[test_indices]['Post'].values
        test_eng_texts = df.iloc[test_indices]['tekst_eng'].values
        test_processed = df.iloc[test_indices]['przetworzony_tekst'].values
        test_languages = df.iloc[test_indices]['jezyk'].values

        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=['Inne', 'Mandrill']))

        print("\nMacierz pomyłek:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        self.plot_confusion_matrix(cm)

        # Plot feature importance
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        log_probs = self.classifier.feature_log_prob_
        feature_importance = log_probs[1] - log_probs[0]
        self.plot_feature_importance(feature_names, feature_importance)

        # Plot prediction confidence
        probabilities = self.classifier.predict_proba(X_test)
        self.plot_prediction_confidence(y_test, y_pred, probabilities)
        print("\nInterpretacja macierzy pomyłek:")
        print(f"True Negatives (poprawnie jako Inne): {cm[0][0]}")
        print(f"False Positives (błędnie jako Mandrill): {cm[0][1]}")
        print(f"False Negatives (błędnie jako Inne): {cm[1][0]}")
        print(f"True Positives (poprawnie jako Mandrill): {cm[1][1]}")

        print("\nAnaliza klasyfikacji:")
        print("\nPrzykłady POPRAWNYCH klasyfikacji:")
        correct_count = 0
        for i, (true, pred) in enumerate(zip(y_test, y_pred)):
            if true == pred and correct_count < 5:
                correct_count += 1
                prob = self.classifier.predict_proba(X_test[i])
                print(f"\nTekst oryginalny: {test_texts[i]}")
                if test_languages[i] != 'en':
                    print(f"Tłumaczenie: {test_eng_texts[i]}")
                print(f"Tekst po preprocessingu: {test_processed[i]}")
                print(f"Język: {test_languages[i]}")
                print(f"Prawdziwa klasa: {'Mandrill' if true == 1 else 'Inne'}")
                print(f"Przewidziana klasa: {'Mandrill' if pred == 1 else 'Inne'}")
                print(f"Pewność przewidywania: {max(prob[0]):.2f}")

        print("\nPrzykłady BŁĘDNYCH klasyfikacji:")
        for i, (true, pred) in enumerate(zip(y_test, y_pred)):
            if true != pred:
                prob = self.classifier.predict_proba(X_test[i])
                print(f"\nTekst oryginalny: {test_texts[i]}")
                if test_languages[i] != 'en':
                    print(f"Tłumaczenie: {test_eng_texts[i]}")
                print(f"Tekst po preprocessingu: {test_processed[i]}")
                print(f"Język: {test_languages[i]}")
                print(f"Prawdziwa klasa: {'Mandrill' if true == 1 else 'Inne'}")
                print(f"Przewidziana klasa: {'Mandrill' if pred == 1 else 'Inne'}")
                print(f"Pewność przewidywania: {max(prob[0]):.2f}")

        self.analizuj_cechy()

        return X_test, y_test, y_pred

    def analizuj_cechy(self):
        feature_names = self.vectorizer.get_feature_names_out()
        log_probs = self.classifier.feature_log_prob_
        feature_importance = log_probs[1] - log_probs[0]

        top_mandrill_idx = np.argsort(feature_importance)[-10:]
        print("\nNajważniejsze słowa dla klasy Mandrill:")
        for idx in reversed(top_mandrill_idx):
            print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")

        top_other_idx = np.argsort(feature_importance)[:10]
        print("\nNajważniejsze słowa dla klasy Inne:")
        for idx in top_other_idx:
            print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")


def main():
    for usun_duplikaty in [False, True]:
        print(f"\n{'=' * 80}")
        print(f"Uruchamiam klasyfikator {'z' if usun_duplikaty else 'bez'} usuwania duplikatów")
        print('=' * 80)

        klasyfikator = KlasyfikatorTweetowMandrill(usun_duplikaty=usun_duplikaty)
        print("Rozpoczynam wczytywanie danych...")
        df = klasyfikator.przygotuj_dane('MED-lab-3-Zad 3-Mandrill-Dane.xlsx')

        print("\nRozpoczynam trenowanie modelu...")
        X_test, y_test, y_pred = klasyfikator.trenuj(df)


if __name__ == "__main__":
    main()