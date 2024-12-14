import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

data = {
    'College level': [1, 2, 5, 1, 4, 3, 2, 1, 5, 2, 3, 4, 1, 2,
                      5, 4, 3, 1, 4, 5, 2, 5, 3, 4, 3, 2, 5, 1],
    'Marital status': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                       1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  # 1 - małżeństwo, 0 - wolny
}

data_df = pd.DataFrame(data)
print(data_df)

# MODEL LOGITOW


def calculate_logit(data):
    grouped = data.groupby('College level')['Marital status'].agg(
        numberOfStudents='count',
        married='sum'
    ).reset_index()

    # obliczenie prawdopodobieństwa małżeństwa dla każdego poziomu
    grouped['p'] = grouped['married'] / grouped['numberOfStudents']
    grouped['logit'] = np.log(grouped['p'] / (1 - grouped['p']))
    return grouped


# przygotowywanie danych logitowych:
logit_data = calculate_logit(data_df)
print(logit_data)
# %%
linear_model = LinearRegression()
linear_model.fit(data_df[['College level']], data_df['Marital status'])

# dopasowanie modelu logitowego:
logit_model = LinearRegression()
logit_model.fit(logit_data[['College level']], logit_data[['logit']])

# zakres predykcji:
X_range = np.linspace(data_df['College level'].min() - 1, data_df['College level'].max() + 1, 100).reshape(-1, 1)
linear_predicts = linear_model.predict(X_range)

# p-ństwa dla modelu logitowego
beta0 = logit_model.intercept_[0] # wyraz wolny
beta1 = logit_model.coef_[0][0] # współczynnik kierunkowy
print(f"b0: {beta0}")
print(f"b1: {beta1}")
# %%
logit_predicts = 1 / (1 + np.exp(-(beta0 + beta1 * X_range)))  # sigmoid func
print(logit_predicts)  # predykcje modelu logitowego

# Wykres
plt.figure(figsize=(10, 6))
plt.scatter(data_df['College level'], data_df['Marital status'], label='Dane rzeczywiste', alpha=0.7)
plt.plot(X_range, linear_predicts, color='red', label='Krzywa sigmoidalna (logitowy model)')
plt.plot(X_range, logit_predicts, color='blue', label='Linia regresji liniowej')

plt.ylabel('Marital status')
plt.xlabel('College level')
plt.legend()
plt.show()

# MODEL LINIOWY - współczynniki i R2

linear_coef = linear_model.coef_[0]
linear_intercept = linear_model.intercept_
linear_r2 = linear_model.score(data_df[['College level']], data_df['Marital status'])

print("Współczynnik kierunkowy a1: ", linear_coef)
print("Wyraz wolny a0: ", linear_intercept)
print("R2: ", linear_r2)

print(f"Model liniowy: m = {linear_coef} + {linear_intercept} * c")
# %%
# Przygotowanie danych rzeczywistych
logit_data = calculate_logit(data_df)

# wyciągnięcie predykcji modelu dla tych samych poziomów studiów
predicted_probs = 1 / (1 + np.exp(-(beta0 + beta1 * logit_data['College level'].values)))

logit_data['predicted_p'] = predicted_probs

print(logit_data[['College level', 'p', 'predicted_p']])

logit_data['error'] = logit_data['p'] - logit_data['predicted_p']

# (MAE):
mae = abs(logit_data['error']).mean()
print(f"Średni błąd bezwzględny (MAE): {mae}")

