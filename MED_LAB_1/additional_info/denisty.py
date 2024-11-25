import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score


# Wersja z pandas - jeśli dane są w formacie csv
data = pd.read_csv(r"C:\Users\szyme\PycharmProjects\MED\MED_LAB_1\data\dane.txt", delim_whitespace=True)

def analyze_correlations(df, target='Pct.BF'):
    """Analyze correlations with target variable"""
    correlations = df.corr()[target].sort_values(ascending=False)
    print(f"\nCorrelations with {target}:")
    print(correlations)

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    return correlations


def build_regression_model(X, y, feature_names):
    """Build and evaluate a regression model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print results
    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('Coefficient', ascending=False))

    return model, scaler, r2, rmse


def perform_feature_selection(X, y, feature_names, n_features=5):
    """Perform recursive feature elimination"""
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
    rfe = rfe.fit(X, y)

    selected_features = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Rank': rfe.ranking_
    })

    print("\nFeature Selection Results:")
    print(selected_features.sort_values('Rank'))

    return selected_features


# Analyze correlations
correlations = analyze_correlations(data)

# Prepare features and target
X = data.drop(['Pct.BF', 'Density'], axis=1)  # Remove Density as it's directly related to body fat
y = data['Pct.BF']

# Build initial model with all features
print("\nInitial Model with All Features:")
model, scaler, r2, rmse = build_regression_model(X, y, X.columns)

# Perform feature selection
selected_features = perform_feature_selection(X, y, X.columns)

# Build model with selected features
best_features = selected_features[selected_features['Selected']]['Feature'].tolist()
X_selected = X[best_features]
print("\nModel with Selected Features:")
model_selected, scaler_selected, r2_selected, rmse_selected = build_regression_model(X_selected, y, best_features)



# 1. Najpierw pokazmy relację Density -> Pct.BF
def plot_density_relation(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Density'], data['Pct.BF'], alpha=0.5)
    plt.xlabel('Density')
    plt.ylabel('Body Fat %')
    plt.title('Relationship between Density and Body Fat %')

    # Dodajmy linię teoretyczną ze wzoru Siriego
    density_range = np.linspace(data['Density'].min(), data['Density'].max(), 100)
    bodyfat = (495 / density_range) - 450
    plt.plot(density_range, bodyfat, 'r-', label='Siri Equation')
    plt.legend()
    plt.show()


# 2. Teraz zbudujmy model bez używania Density
def build_practical_model(data):
    # Używamy tylko praktycznych zmiennych
    features = ['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Waist']
    X = data[features]
    y = data['Pct.BF']

    # Standaryzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model z cross-walidacją
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    # Fit na całym zbiorze dla współczynników
    model.fit(X_scaled, y)

    # Analiza ważności cech
    importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\nModel bez używania Density:")
    print(f"Cross-validated R² score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("\nWażność zmiennych:")
    print(importance)

    return model, scaler


# 3. Porównajmy przewidywania
def compare_predictions(data, model, scaler):
    features = ['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Waist']
    X = data[features]
    X_scaled = scaler.transform(X)

    data['Predicted_BF'] = model.predict(X_scaled)
    data['BF_from_Density'] = (495 / data['Density']) - 450

    plt.figure(figsize=(10, 6))
    plt.scatter(data['Pct.BF'], data['Predicted_BF'], alpha=0.5)
    plt.plot([0, 50], [0, 50], 'r--')  # linia idealna
    plt.xlabel('Rzeczywisty % tłuszczu')
    plt.ylabel('Przewidywany % tłuszczu')
    plt.title('Porównanie rzeczywistych i przewidywanych wartości')
    plt.show()


# Wykonanie analizy
plot_density_relation(data)
model, scaler = build_practical_model(data)
compare_predictions(data, model, scaler)