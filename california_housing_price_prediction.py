# California Housing Price Prediction

# Dataset  : California Census Housing Data (1990)
# Goal     : Predict median house value
# Models   : Linear Regression vs Random Forest Regressor
# Result   : Random Forest → R² = 0.81, RMSE = $50,253

# IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully.")


# STEP 1 — DATA CHECK

df = pd.read_csv('housing.csv')

print(f'Dataset Shape: {df.shape[0]:,} rows * {df.shape[1]:,} columns')
print(f'\nColumns: {list(df.columns)}')
print(f'\nFirst 5 Rows:')
print(df.head())
print(df.info())

# Key observations:
# - 20,640 rows, 10 columns
# - total_bedrooms has 207 missing values (~1% of data)
# - ocean_proximity is text (object) — needs encoding
# - All other columns are numeric


# STEP 2 — CLEANING THE DATA


# Filling missing bedroom values with the column mean
# We use the mean because it represents a typical house in the dataset
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

# Encoding ocean_proximity from text to numbers

le = LabelEncoder()
df['ocean_proximity_encoded'] = le.fit_transform(df['ocean_proximity'])

# Confirming no missing values remain
print(f'Missing values after cleaning: {df.isnull().sum().sum()}')
print(f'\nOcean Proximity Encoding:')
print(df[['ocean_proximity', 'ocean_proximity_encoded']].drop_duplicates().sort_values('ocean_proximity_encoded'))


# STEP 3 — ENGINEERING FEATURES

# Rooms per household — measures house spaciousness
df['rooms_per_household'] = df['total_rooms'] / df['households']

# Bedroom ratio — high ratio = cramped, mostly bedrooms
df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']

# Population per household — measures neighborhood density
df['population_per_household'] = df['population'] / df['households']

print("Engineered Features (first 5 rows):")
print(df[['rooms_per_household', 'bedroom_ratio', 'population_per_household']].head())


# STEP 4 — DEFINING FEATURES & TARGET

features = [
    'longitude',                 
    'latitude',                 
    'housing_median_age',       
    'total_rooms',                
    'total_bedrooms',             
    'population',                 
    'households',                
    'median_income',              
    'ocean_proximity_encoded',    
    'rooms_per_household',        
    'bedroom_ratio',              
    'population_per_household']

# Target
X = df[features]
y = df['median_house_value']

print(f'\nNumber of features: {len(features)}')
print(f'Target — Median House Value:')
print(f'  Mean : ${y.mean():,.0f}')
print(f'  Min  : ${y.min():,.0f}')
print(f'  Max  : ${y.max():,.0f}')

# Visualizing target distribution
plt.figure(figsize=(8, 4))
plt.hist(y, bins=50, color='steelblue', edgecolor='white')
plt.xlabel('Median House Value ($)')
plt.ylabel('Number of Districts')
plt.title('Distribution of California House Prices')
plt.tight_layout()
plt.show()


# STEP 5 — TRAINING THE MODELS


# Split: 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training rows : {len(X_train):,}')
print(f'Testing rows  : {len(X_test):,}')

# --- Model 1: Linear Regression ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# --- Model 2: Random Forest Regressor ---
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# STEP 6 — READING THE OUTPUT

print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f'Linear Regression  → RMSE: ${rmse_lr:,.0f}   R²: {r2_lr:.4f}')
print(f'Random Forest      → RMSE: ${rmse_rf:,.0f}   R²: {r2_rf:.4f}')
print("=" * 50)

# Interpretation:
# RMSE = average prediction error in dollars
# R²   = how much variation the model explains (1.0 = perfect, 0 = useless)
# Random Forest wins: explains 81% of house price variation

# Which features did the Random Forest rely on most?
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f'\nFeature Importance (Random Forest):')
print(importance.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.barh(importance['Feature'], importance['Importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Which Features Drive California House Prices?')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Key Insight:
# median_income is by far the strongest predictor (0.52 importance)
# Location (latitude + longitude) contributes ~13% combined
# The wealthier the neighborhood, the higher the house price
# "Location, location, location" — but income matters more
