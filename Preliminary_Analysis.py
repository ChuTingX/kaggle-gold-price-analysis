import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and preview data
df = pd.read_csv("FINAL_USO.csv", parse_dates=["Date"], index_col="Date")
print("Data loaded. Shape:", df.shape)
print(df.head(), "\n")

# 2. Check missing weekdays (potential holidays)
unique_weekdays = df.index.dayofweek.unique()
print("Weekdays present (Mon=0, Sun=6):", unique_weekdays)
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
missing_dates = full_range.difference(df.index)
print("Missing business days:", len(missing_dates))
if len(missing_dates) > 0:
    print("Missing dates (likely holidays):", missing_dates)
print()

# 3. Remove trend columns
trend_columns = [col for col in df.columns if col.endswith('_Trend')]
df_reduced = df.drop(columns=trend_columns, errors='ignore')

# 4. Select relevant columns
variables = [
    'Adj Close',
    'SP_close',
    'DJ_close',
    'USDI_Price',
    'EU_Price',
    'GDX_Close',
    'SF_Price',
    'PLT_Price',
    'PLD_Price',
    'RHO_PRICE',
    'USO_Close',
    'OF_Price',
    'OS_Price'
]
variables = [v for v in variables if v in df_reduced.columns]
analysis_df = df_reduced[variables]

# 5. Descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(analysis_df.describe().T)
print("\n=== MEDIAN VALUES ===")
print(analysis_df.median())

# 6. Correlation and heatmap
corr_matrix = analysis_df.corr()
print("\n=== CORRELATION MATRIX ===")
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 7. Time-series plots
if 'Adj Close' in analysis_df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_df.index, analysis_df['Adj Close'], color='gold', label='Gold ETF Adj Close')
    plt.title("Gold ETF Adjusted Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

other_vars = [v for v in variables if v != 'Adj Close']
if other_vars:
    plt.figure(figsize=(12, 6))
    for var in other_vars:
        plt.plot(analysis_df.index, analysis_df[var], label=var)
    plt.title("Selected Instruments Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price / Index Level")
    plt.legend()
    plt.grid(True)
    plt.show()

print("COMPLETE")
