import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

# 1. Load Data
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# Remove trend columns
trend_columns = [col for col in df.columns if col.endswith('_Trend')]
df.drop(columns=trend_columns, errors='ignore', inplace=True)

# Check for 'Adj Close'
if 'Adj Close' not in df.columns:
    raise ValueError("No 'Adj Close' column found.")

# Add rolling averages
df['Adj_Close_7d'] = df['Adj Close'].rolling(window=7, min_periods=7).mean()
df['Adj_Close_30d'] = df['Adj Close'].rolling(window=30, min_periods=30).mean()
df.dropna(inplace=True)

# 2. Create Lagged Features
base_predictors = [
    'Adj Close', 'SP_close', 'DJ_close', 'USDI_Price', 'EU_Price', 'GDX_Close',
    'SF_Price', 'PLT_Price', 'PLD_Price', 'RHO_PRICE', 'USO_Close',
    'OF_Price', 'OS_Price', 'Adj_Close_7d', 'Adj_Close_30d'
]
base_predictors = [p for p in base_predictors if p in df.columns]

lagged_df = pd.DataFrame(index=df.index)
lagged_df['Adj Close'] = df['Adj Close']
for var in base_predictors:
    lagged_df[var + '_prev'] = df[var].shift(1)
lagged_df.dropna(inplace=True)

# 3. Baseline MSE (Prev Day as Prediction)
target = 'Adj Close'
predictors = [c for c in lagged_df.columns if c.endswith('_prev')]
X = lagged_df[predictors].values
y = lagged_df[target].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)
X_baseline = lagged_df['Adj Close_prev'].values

baseline_mse_scores = []
for train_idx, test_idx in kf.split(X_baseline):
    mse = mean_squared_error(y[test_idx], X_baseline[test_idx])
    baseline_mse_scores.append(mse)
mean_baseline_mse = np.mean(baseline_mse_scores)
print(f"Baseline MSE: {mean_baseline_mse:.4f}")

# 4. Random Forest Tuning via GridSearchCV
def mse_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

mse_scorer = make_scorer(mse_score, greater_is_better=False)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 15, 25, 35],
    'max_features': ['sqrt', 0.5, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid=param_grid,
    scoring=mse_scorer,
    cv=kf,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X, y)

best_mse = abs(grid_search.best_score_)
best_params = grid_search.best_params_
print("\nBest parameters:", best_params)
print(f"Best MSE (GridSearchCV): {best_mse:.4f}")

best_rf = grid_search.best_estimator_

rf_mse_scores = []
for train_idx, test_idx in kf.split(X):
    best_rf.fit(X[train_idx], y[train_idx])
    preds = best_rf.predict(X[test_idx])
    rf_mse_scores.append(mean_squared_error(y[test_idx], preds))

mean_rf_mse = np.mean(rf_mse_scores)
std_rf_mse = np.std(rf_mse_scores)
print("\nTuned RF 10-Fold MSE:", rf_mse_scores)
print(f"Tuned RF Mean MSE: {mean_rf_mse:.4f} ± {std_rf_mse:.4f}")

print("COMPLETE")