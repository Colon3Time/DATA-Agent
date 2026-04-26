import pandas as pd
import numpy as np
import os
from pathlib import Path

INPUT_PATH = "C:/Users/Amorntep/DATA-Agent/projects/2026-04-25_test/input/sales_data_500.csv"
OUTPUT_DIR = "C:/Users/Amorntep/DATA-Agent/projects/2026-04-25_test/output/dana"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load - check column names first
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Columns: {list(df.columns)}")
print(f"[STATUS] Loaded: {df.shape}")

# Strip whitespace from column names in case they have spaces
df.columns = df.columns.str.strip()

# Check if price column exists (maybe misspelled or different name)
possible_price_cols = [c for c in df.columns if 'price' in c.lower() or 'amount' in c.lower() or 'cost' in c.lower()]
if 'price' not in df.columns and possible_price_cols:
    # Use the closest match
    df.rename(columns={possible_price_cols[0]: 'price'}, inplace=True)
    print(f"[STATUS] Renamed '{possible_price_cols[0]}' to 'price'")

# 2. Handle missing values
# date (2.6% missing) -> forward fill for time series
date_null_idx = df['date'].isnull()
df['date'] = df['date'].ffill()
df = df.dropna(subset=['date'])
print(f"[STATUS] Date missing filled: {df.shape}")

# category (3.8% missing) -> fill with 'Unknown'
df['category'] = df['category'].fillna('Unknown')

# price (2.2% missing) -> fill with median
price_median = df['price'].median()
df['price'] = df['price'].fillna(price_median)

# quantity (2.8% missing) -> fill with median (int)
qty_median = int(df['quantity'].median())
df['quantity'] = df['quantity'].fillna(qty_median).astype(int)

# region (3.2% missing) -> fill with mode
region_mode = df['region'].mode()[0]
df['region'] = df['region'].fillna(region_mode)

# 3. Handle outliers - price
# IQR method for price
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
price_outliers = len(df[(df['price'] < lower) | (df['price'] > upper)])
print(f"[STATUS] Price outliers (IQR): {price_outliers} points")

# Cap outliers at upper/lower bounds
df['price'] = df['price'].clip(lower=lower, upper=upper)

# 4. Fix data types
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Strip whitespace from string columns
for col in ['product', 'category', 'region']:
    if col in df.columns:
        df[col] = df[col].str.strip().str.title()

# 5. Recalculate total_amount = price * quantity
df['total_amount'] = df['price'] * df['quantity']
df['total_amount'] = df['total_amount'].round(2)

# 6. Drop duplicates if any
before_dedup = len(df)
df = df.drop_duplicates()
print(f"[STATUS] Duplicates removed: {before_dedup - len(df)}")

# Final check
print(f"\n[STATUS] Final shape: {df.shape}")
print(f"[STATUS] Missing remaining:\n{df.isnull().sum()}")

# Save output
output_path = os.path.join(OUTPUT_DIR, "dana_output.csv")
df.to_csv(output_path, index=False)
print(f"[STATUS] Output saved to: {output_path}")
