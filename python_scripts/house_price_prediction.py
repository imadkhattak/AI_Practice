"""
End-to-End House Price Prediction using Multiple Linear Regression
==================================================================
Steps covered:
1. Environment Setup & Imports
2. Data Acquisition / Generation (Realistic synthetic house dataset)
3. Exploratory Data Analysis (EDA)
4. Data Cleaning (Missing values, duplicates, outliers)
5. Data Visualization (Distributions, Heatmaps, Scatter plots)
6. Feature Engineering & Feature Selection
7. Train/Test Split (Preventing Data Leakage)
8. Feature Standardization (StandardScaler)
9. Model Training (Linear Regression)
10. Model Evaluation (MAE, MSE, RMSE, R² Score, Residuals)
11. Inference / Predicting on New Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
sns.set_theme(style="whitegrid")
np.random.seed(42)

# ==============================================================================
# STEP 1: CREATE A REALISTIC DATASET (HOUSE PRICES)
# ==============================================================================
print("--- [1] Generating Dataset ---")
n_samples = 1000

sqft = np.random.normal(loc=2000, scale=600, size=n_samples).clip(600, 5000)
bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.20, 0.45, 0.22, 0.08])
bathrooms = (bedrooms * 0.75 + np.random.choice([0, 0.5, 1], size=n_samples)).clip(1, 5)
age = np.random.randint(1, 50, size=n_samples)
distance_to_center_km = np.random.uniform(1.0, 35.0, size=n_samples)
neighborhood_score = np.random.uniform(1.0, 10.0, size=n_samples)  # 1 to 10 scale

# Ground-truth formula with noise
price = (
    sqft * 160
    + bedrooms * 12000
    + bathrooms * 8000
    - age * 1200
    - distance_to_center_km * 3500
    + neighborhood_score * 15000
    + np.random.normal(0, 25000, size=n_samples)
)

df = pd.DataFrame({
    'SquareFeet': np.round(sqft, 1),
    'Bedrooms': bedrooms,
    'Bathrooms': np.round(bathrooms, 1),
    'AgeYears': age,
    'DistanceToCenter_km': np.round(distance_to_center_km, 1),
    'NeighborhoodScore': np.round(neighborhood_score, 1),
    'Price': np.round(price, 2)
})

# Inject some realistic flaws for demonstration (missing values & duplicate rows)
df.loc[np.random.choice(df.index, size=20, replace=False), 'SquareFeet'] = np.nan
df.loc[np.random.choice(df.index, size=15, replace=False), 'NeighborhoodScore'] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # add 5 duplicates

print(f"Initial shape: {df.shape}")
print(df.head())


# ==============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n--- [2] Exploratory Data Analysis (EDA) ---")
print("\nData Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values Count:")
print(df.isnull().sum())

print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")


# ==============================================================================
# STEP 3: DATA CLEANING
# ==============================================================================
print("\n--- [3] Data Cleaning ---")

# 1. Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}")

# 2. Impute missing values (using median to prevent skewness from outliers)
df['SquareFeet'] = df['SquareFeet'].fillna(df['SquareFeet'].median())
df['NeighborhoodScore'] = df['NeighborhoodScore'].fillna(df['NeighborhoodScore'].median())
print(f"Missing values remaining: {df.isnull().sum().sum()}")

# 3. Outlier handling using IQR (Interquartile Range) for numerical columns
numerical_cols = ['SquareFeet', 'Price']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap/clip outliers to boundaries instead of dropping to retain data volume
    df[col] = np.clip(df[col], lower_bound, upper_bound)

print("Outlier clipping completed.")


# ==============================================================================
# STEP 4: DATA VISUALIZATION
# ==============================================================================
print("\n--- [4] Generating Data Visualizations ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target variable distribution
sns.histplot(df['Price'], kde=True, ax=axes[0, 0], color='royalblue')
axes[0, 0].set_title('House Price Distribution', fontsize=12, fontweight='bold')

# 2. SquareFeet vs Price
sns.scatterplot(x='SquareFeet', y='Price', data=df, ax=axes[0, 1], alpha=0.6, color='coral')
axes[0, 1].set_title('Square Feet vs. Price', fontsize=12, fontweight='bold')

# 3. Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 0])
axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 4. Neighborhood Score vs Price boxplot (binned score)
df['ScoreBin'] = pd.cut(df['NeighborhoodScore'], bins=5)
sns.boxplot(x='ScoreBin', y='Price', data=df, ax=axes[1, 1], hue='ScoreBin', palette='Blues', legend=False)
axes[1, 1].set_title('Neighborhood Score Bins vs. Price', fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.close()
print("Saved visualization plots to 'eda_plots.png'.")

# Drop helper bin column
df = df.drop(columns=['ScoreBin'])


# ==============================================================================
# STEP 5: FEATURE SELECTION & PREPARATION
# ==============================================================================
print("\n--- [5] Feature Selection & Target Separation ---")

# Inspect correlations with target 'Price'
corr_with_target = df.corr()['Price'].sort_values(ascending=False)
print("Correlation of features with Price:\n", corr_with_target)

# Define Features (X) and Target (y)
X = df.drop(columns=['Price'])
y = df['Price']

print(f"Features: {list(X.columns)}")
print(f"Target: Price")


# ==============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ==============================================================================
# IMPORTANT: Split BEFORE scaling to avoid data leakage from test set to training set!
print("\n--- [6] Train-Test Split (80% Train, 20% Test) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ==============================================================================
# STEP 7: STANDARDIZATION (SCALING)
# ==============================================================================
print("\n--- [7] Feature Standardization (StandardScaler) ---")
# Formula: z = (x - mean) / std_dev
scaler = StandardScaler()

# fit_transform on training data ONLY
X_train_scaled = scaler.fit_transform(X_train)

# transform on testing data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for clean readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("Features successfully standardized (Mean ~ 0, Std ~ 1).")


# ==============================================================================
# STEP 8: MODEL TRAINING (LINEAR REGRESSION)
# ==============================================================================
print("\n--- [8] Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Model Intercept (b0): ${model.intercept_:,.2f}")
print("Model Coefficients (weights):")
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(coeff_df)


# ==============================================================================
# STEP 9: MODEL EVALUATION
# ==============================================================================
print("\n--- [9] Model Evaluation on Test Data ---")
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE)  : ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²) Score       : {r2:.4f} ({r2*100:.2f}% variance explained)")

# Residuals Plot (Actual vs Predicted & Error Distribution)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Actual vs Predicted
sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='teal', alpha=0.7)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
axes[0].set_title('Actual vs. Predicted Prices', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')

# 2. Residual Distribution (Errors should be normally distributed around 0)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('Residuals (Errors) Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Prediction Error ($)')

plt.tight_layout()
plt.savefig('model_evaluation_plots.png')
plt.close()
print("Saved evaluation plots to 'model_evaluation_plots.png'.")


# ==============================================================================
# STEP 10: INFERENCE / PREDICTING ON NEW UNSEEN DATA
# ==============================================================================
print("\n--- [10] Inference on New House Samples ---")
new_houses = pd.DataFrame([
    {
        'SquareFeet': 2500,
        'Bedrooms': 4,
        'Bathrooms': 3.0,
        'AgeYears': 5,
        'DistanceToCenter_km': 6.5,
        'NeighborhoodScore': 8.5
    },
    {
        'SquareFeet': 1200,
        'Bedrooms': 2,
        'Bathrooms': 1.0,
        'AgeYears': 30,
        'DistanceToCenter_km': 22.0,
        'NeighborhoodScore': 4.0
    }
])

# Scale the new data using the ALREADY FITTED scaler
new_houses_scaled = pd.DataFrame(scaler.transform(new_houses), columns=X.columns)
predicted_prices = model.predict(new_houses_scaled)

for i, pred in enumerate(predicted_prices):
    print(f"Sample House #{i+1} specs:\n{new_houses.iloc[i].to_dict()}")
    print(f"--> Estimated Price: ${pred:,.2f}\n")
