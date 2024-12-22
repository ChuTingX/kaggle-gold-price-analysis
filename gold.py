# 1. Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models and evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

# Exhaustive feature selection
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Seaborn style
sns.set_style("whitegrid")
sns.set_context("talk")

# 2. Data Loading & Preparation
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")

# Drop trend columns
trend_columns = [col for col in df.columns if col.endswith('_Trend')]
df.drop(columns=trend_columns, inplace=True, errors='ignore')

# Check that 'Adj Close' is present
if 'Adj Close' not in df.columns:
    raise ValueError("Expected 'Adj Close' column not found!")

# Rolling averages
df['Adj_Close_7d'] = df['Adj Close'].rolling(window=7, min_periods=7).mean()
df['Adj_Close_30d'] = df['Adj Close'].rolling(window=30, min_periods=30).mean()
df.dropna(inplace=True)

# Base predictors
base_predictors = [
    'Adj Close', 'SP_close', 'DJ_close', 'USDI_Price', 'EU_Price', 'GDX_Close',
    'SF_Price', 'PLT_Price', 'PLD_Price', 'RHO_PRICE',
    'USO_Close', 'OF_Price', 'OS_Price',
    'Adj_Close_7d', 'Adj_Close_30d'
]
base_predictors = [p for p in base_predictors if p in df.columns]

# Create lagged features
lagged_df = pd.DataFrame(index=df.index)
lagged_df['Adj Close'] = df['Adj Close']
for var in base_predictors:
    lagged_df[var + '_prev'] = df[var].shift(1)
lagged_df.dropna(inplace=True)

print("Data Shape after prep:", lagged_df.shape)
print(lagged_df)

# 3. Baseline Model (Prev Day as Prediction)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
predictor_cols = [c for c in lagged_df.columns if c.endswith('_prev')]
X = lagged_df[predictor_cols].values
y = lagged_df['Adj Close'].values

# Check baseline column
if 'Adj Close_prev' not in lagged_df.columns:
    raise ValueError("No 'Adj Close_prev' column found!")

X_baseline = lagged_df['Adj Close_prev'].values
baseline_mse_scores = []
for train_i, test_i in kf.split(X_baseline):
    y_test = y[test_i]
    y_pred_base = X_baseline[test_i]
    baseline_mse_scores.append(mean_squared_error(y_test, y_pred_base))

mean_baseline_mse = np.mean(baseline_mse_scores)
print("\n=== BASELINE MODEL (Prev Day) ===")
print("Baseline 10-Fold MSE:", baseline_mse_scores)
print(f"Mean MSE = {mean_baseline_mse:.4f}")

# 4. Random Forest (Fixed Params)
best_rf = RandomForestRegressor(
    bootstrap=True,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)

rf_mse_scores = []
for train_i, test_i in kf.split(X):
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]
    best_rf.fit(X_train, y_train)
    y_pred_rf = best_rf.predict(X_test)
    rf_mse_scores.append(mean_squared_error(y_test, y_pred_rf))

mean_rf_mse = np.mean(rf_mse_scores)
print("\n=== RANDOM FOREST ===")
print("RF 10-Fold MSE:", rf_mse_scores)
print(f"Mean MSE = {mean_rf_mse:.4f}")

# 5. Linear Regression with Exhaustive Search
def neg_mse_scorer(estimator, X_fs, y_fs):
    return -mean_squared_error(y_fs, estimator.predict(X_fs))

lr = LinearRegression()
efs = EFS(
    estimator=lr,
    min_features=1,
    max_features=len(predictor_cols),
    scoring=neg_mse_scorer,
    cv=kf,
    n_jobs=-1
).fit(X, y)

best_subset = efs.best_idx_
best_score = efs.best_score_
lr_best_feats = [predictor_cols[i] for i in best_subset]

print("\n=== LINEAR REGRESSION (All-Subset) ===")
print("Best subset (indices):", best_subset)
print("Best subset (names):", lr_best_feats)
print(f"CV MSE of best subset: {-best_score:.4f}")

X_lr_sub = lagged_df[lr_best_feats].values
lr_mses = []
for train_i, test_i in kf.split(X_lr_sub):
    X_train, X_test = X_lr_sub[train_i], X_lr_sub[test_i]
    y_train, y_test = y[train_i], y[test_i]

    tmp_lr = LinearRegression()
    tmp_lr.fit(X_train, y_train)
    y_pred = tmp_lr.predict(X_test)
    lr_mses.append(mean_squared_error(y_test, y_pred))

mean_lr_mse = np.mean(lr_mses)
print("All-Subset LR 10-Fold MSE:", lr_mses)
print(f"Mean MSE = {mean_lr_mse:.4f}")

# 6. Lasso and Ridge Tuning with Scaling
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
alphas = np.logspace(-5, 2, 20)

# Lasso
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000, random_state=42))
])
lasso_param_grid = {'lasso__alpha': alphas}
lasso_gs = GridSearchCV(lasso_pipeline, lasso_param_grid, scoring=mse_scorer, cv=kf, n_jobs=-1)
lasso_gs.fit(X, y)

lasso_best_alpha = lasso_gs.best_params_['lasso__alpha']
lasso_best_mse = abs(lasso_gs.best_score_)
best_lasso = lasso_gs.best_estimator_

lasso_mses = []
for train_i, test_i in kf.split(X):
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]
    best_lasso.fit(X_train, y_train)
    y_pred_lasso = best_lasso.predict(X_test)
    lasso_mses.append(mean_squared_error(y_test, y_pred_lasso))
mean_lasso_mse = np.mean(lasso_mses)

# Ridge
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(max_iter=10000, random_state=42))
])
ridge_param_grid = {'ridge__alpha': alphas}
ridge_gs = GridSearchCV(ridge_pipeline, ridge_param_grid, scoring=mse_scorer, cv=kf, n_jobs=-1)
ridge_gs.fit(X, y)

ridge_best_alpha = ridge_gs.best_params_['ridge__alpha']
ridge_best_mse = abs(ridge_gs.best_score_)
best_ridge = ridge_gs.best_estimator_

ridge_mses = []
for train_i, test_i in kf.split(X):
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]
    best_ridge.fit(X_train, y_train)
    y_pred_ridge = best_ridge.predict(X_test)
    ridge_mses.append(mean_squared_error(y_test, y_pred_ridge))
mean_ridge_mse = np.mean(ridge_mses)

# 7. Ensemble (RF, Lasso, Ridge, Subset LR)
ensemble_mses = []
for train_i, test_i in kf.split(X):
    X_train_full, X_test_full = X[train_i], X[test_i]
    y_train_fold, y_test_fold = y[train_i], y[test_i]

    # Random Forest
    best_rf.fit(X_train_full, y_train_fold)
    pred_rf = best_rf.predict(X_test_full)

    # Lasso
    best_lasso.fit(X_train_full, y_train_fold)
    pred_lasso = best_lasso.predict(X_test_full)

    # Ridge
    best_ridge.fit(X_train_full, y_train_fold)
    pred_ridge = best_ridge.predict(X_test_full)

    # Linear Regression on best subset
    X_train_lr = X_train_full[:, best_subset]
    X_test_lr = X_test_full[:, best_subset]
    tmp_lr2 = LinearRegression()
    tmp_lr2.fit(X_train_lr, y_train_fold)
    pred_lr = tmp_lr2.predict(X_test_lr)

    # Average predictions
    ensemble_pred = (pred_rf + pred_lasso + pred_ridge + pred_lr) / 4.0
    ensemble_mses.append(mean_squared_error(y_test_fold, ensemble_pred))

mean_ensemble_mse = np.mean(ensemble_mses)
std_ensemble_mse = np.std(ensemble_mses)

# 8. Comparison of All Models
print("\n=== FINAL COMPARISON (MSE) ===")
print(f"Baseline (Prev Day): {mean_baseline_mse:.4f}")
print(f"Random Forest      : {mean_rf_mse:.4f}")
print(f"All-Subset LR      : {mean_lr_mse:.4f}")
print(f"Lasso (a={lasso_best_alpha:.5f}) : {mean_lasso_mse:.4f}")
print(f"Ridge (a={ridge_best_alpha:.5f}) : {mean_ridge_mse:.4f}")
print(f"Ensemble           : {mean_ensemble_mse:.4f} ± {std_ensemble_mse:.4f}")

model_mse_dict = {
    'Baseline': mean_baseline_mse,
    'RandomForest': mean_rf_mse,
    'AllSubsetLR': mean_lr_mse,
    f'Lasso(a={lasso_best_alpha:.5f})': mean_lasso_mse,
    f'Ridge(a={ridge_best_alpha:.5f})': mean_ridge_mse,
    'Ensemble': mean_ensemble_mse
}

print("COMPLETE")

# 9. FEATURE IMPORTANCE / COEFFICIENTS

# (A) RANDOM FOREST FEATURE IMPORTANCES
rf_importances = best_rf.feature_importances_
rf_importance_df = pd.DataFrame({
    "Feature": predictor_cols,
    "Importance": rf_importances
}).sort_values("Importance", ascending=False)

print("\n=== RANDOM FOREST FEATURE IMPORTANCE (DESC) ===")
print(rf_importance_df)

# (B) LINEAR REGRESSION (BEST SUBSET) COEFFICIENTS
# Fit a final LR on the entire data using only the best subset
X_lr_sub_full = lagged_df[lr_best_feats].values
y_full = lagged_df["Adj Close"].values

final_lr = LinearRegression()
final_lr.fit(X_lr_sub_full, y_full)
lr_coefs = final_lr.coef_

lr_importance_df = pd.DataFrame({
    "Feature": lr_best_feats,
    "Coefficient": lr_coefs,
    "AbsCoefficient": np.abs(lr_coefs)
}).sort_values("AbsCoefficient", ascending=False)

print("\n=== ALL-SUBSET LR COEFFICIENTS (SORTED BY ABS) ===")
print(lr_importance_df)

# (C) LASSO COEFFICIENTS
lasso_coefs = best_lasso.named_steps["lasso"].coef_
lasso_importance_df = pd.DataFrame({
    "Feature": predictor_cols,
    "Coefficient": lasso_coefs,
    "AbsCoefficient": np.abs(lasso_coefs)
}).sort_values("AbsCoefficient", ascending=False)

print("\n=== LASSO COEFFICIENTS (SORTED BY ABS) ===")
print(lasso_importance_df)

# (D) RIDGE COEFFICIENTS
ridge_coefs = best_ridge.named_steps["ridge"].coef_
ridge_importance_df = pd.DataFrame({
    "Feature": predictor_cols,
    "Coefficient": ridge_coefs,
    "AbsCoefficient": np.abs(ridge_coefs)
}).sort_values("AbsCoefficient", ascending=False)

print("\n=== RIDGE COEFFICIENTS (SORTED BY ABS) ===")
print(ridge_importance_df)
