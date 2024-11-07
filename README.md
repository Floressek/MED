# Metody eksploracji danych - Laboratorium 1

**Autor**: Romuald Hoffmann

---

## Temat: Analiza regresji - regresja liniowa

### Przekształcanie funkcji nieliniowych w równoważne liniowe
### Budowa modelu na podstawie danych (ogólnie dostępnych)

---

### Zadanie

Mamy zgromadzone dane dotyczące przedsiębiorstwa internetowego Meta, w tym odnoszące się do portalu „Facebook”. Dane zawarte są w poniższych tabelach i dotyczą:

- liczby użytkowników (klientów) w rozliczeniu na kwartały w poszczególnych latach,
- przychodów i zysków liczone w milionach dolarów amerykańskich,
- zatrudnienia.

#### Tabela 1. Liczba użytkowników portalu społecznościowego „Facebook”

| Kwartał | Liczba użytkowników (mln) |
| ------- | ------------------------- |
| Q3 '08  | 100                       |
| Q1 '09  | 197                       |
| Q2 '09  | 242                       |
| Q3 '09  | 305                       |
| Q4 '09  | 360                       |
| Q1 '10  | 431                       |
| Q2 '10  | 482                       |
| Q3 '10  | 550                       |
| Q4 '10  | 608                       |
| ...     | ...                       |
| Q4 '17  | 2129                      |

#### Tabela 2. Przychody, zysk i zatrudnienie przedsiębiorstwa „Facebook”

| Rok | Przychód (mln $) | Zysk (mln $) | Zatrudnienie |
| --- | ---------------- | ------------ | ------------ |
| 2007 | 153             | -138         | 450          |
| 2008 | 272             | -56          | 850          |
| 2009 | 777             | 229          | 1218         |
| 2010 | 1974            | 606          | 2127         |
| 2011 | 3711            | 1000         | 3200         |
| ...  | ...             | ...          | ...          |
| 2017 | 40653           | 15934        | 25105        |

---

### Instrukcje do analizy

1. **Analiza danych**: Zastanów się, co chcesz zbadać i dlaczego (np. jakie pytania badawcze chcesz sobie odpowiedzieć).
   
2. **Propozycja modelu**: Wybierz model lub modele badające wybrane zależności i wylicz m.in. ich parametry strukturalne, odchylenia standardowe, miary dopasowania oraz przetestuj hipotezy.
   
3. **Analiza na podstawie danych**: Użyj danych przedstawionych w tabelach do przeprowadzenia analizy.
   
4. **Uzasadnienie modelu**: Uzasadnij wybór modeli oraz określ ich potencjalne zastosowanie, np. w predykcji.

5. **Wnioski**: Na podstawie opracowanych modeli sformułuj własne wnioski.

6. **Analiza danych z lat 2018-2020 oraz 2021-2023**: Znajdź dane z tych lat i sprawdź działanie modeli na procesie predykcji dla tego okresu.

7. **Sprawozdanie**: Wyniki analiz, w tym postawione pytania badawcze, hipotezy, wzory, wyniki obliczeń i wnioski, umieść w sprawozdaniu. Obliczenia można wykonać w dowolnym narzędziu używanym na zajęciach.

---

## Modele nieliniowe - linearyzowane

### Przykłady metod linearyzacji wybranych funkcji nieliniowych

1. **Model wykładniczy z jedną zmienną objaśniającą**:
   $[
   \hat{y} = b \cdot a^X
   $
   Logarytmując obustronnie:
   $
   \log \hat{y} = \log b + X \cdot \log a
   $]
   
2. **Model wykładniczy z dwiema zmiennymi objaśniającymi**:
   $[
   \hat{y} = b \cdot a^{X_1} \cdot c^{X_2}
   $
   Logarytmując obustronnie:
   $
   \log \hat{y} = \log b + X_1 \cdot \log a + X_2 \cdot \log c
   $]

3. **Model potęgowy z dwiema zmiennymi objaśniającymi**:
   $[
   \hat{y} = b \cdot X^{a_1} \cdot Z^{a_2}
   $
   Logarytmując obustronnie:
   $
   \log \hat{y} = \log b + a_1 \cdot \log X + a_2 \cdot \log Z
   $]

4. **Model wykładniczo-potęgowy**:
   $[
   \hat{y} = b \cdot a^{X_1} \cdot Z^{a_2}
   $
   Logarytmując obustronnie:
   $
   \log \hat{y} = \log b + X_1 \cdot \log a + a_2 \cdot \log Z
   $]

5. **Wielomian stopnia trzeciego**:
   $[
   \hat{y} = a_0 + a_1 \cdot X + a_2 \cdot X^2 + a_3 \cdot X^3
   $]

6. **Funkcja wymierna ułamkowa (hiperbola)**:
   $[
   \hat{y} = a_1 \cdot \frac{1}{X} + a_0
   $]

---
