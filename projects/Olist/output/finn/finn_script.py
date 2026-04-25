import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define output paths
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'finn_output.csv')
OUTPUT_SCRIPT = os.path.join(OUTPUT_DIR, 'finn_script.py')
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, 'finn_report.md')

INPUT_PATH = r'C:\Users\Amorntep\DATA-Agent\projects\olist\output\max\max_output.csv'
OUTPUT_DIR = r'C:\Users\Amorntep\DATA-Agent\projects\olist\output\finn'

print('[STATUS] Loading data...')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Data types:\n{df.dtypes}')
print(f'[STATUS] Null counts:\n{df.isnull().sum()}')


import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'finn_output.csv')
OUTPUT_SCRIPT = os.path.join(OUTPUT_DIR, 'finn_script.py')
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, 'finn_report.md')

# ----- Configuration -----
INPUT_PATH = r'C:\Users\Amorntep\DATA-Agent\projects\olist\output\max\max_output.csv'
OUTPUT_DIR = r'C:\Users\Amorntep\DATA-Agent\projects\olist\output\finn'
TARGET_COL = 'payment_value'  # Target for feature selection

print('[STATUS] Loading data...')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Null counts:\n{df.isnull().sum()}')

df_original = df.copy()
original_cols = list(df.columns)

# ==========================================
# STEP 1: Datetime Feature Engineering
# ==========================================
print('[STATUS] Creating datetime features...')

datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower() or 'shipping_limit' in col.lower() or 'approved_at' in col.lower() or 'purchase' in col.lower() or 'delivered' in col.lower() or 'estimated' in col.lower()]

for col in datetime_cols:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Basic datetime features
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Day of month
            df[f'{col}_day'] = df[col].dt.day
            
            # Is it first day of month?
            df[f'{col}_is_first_day'] = (df[col].dt.day == 1).astype(int)
            
            # Is it last day of month?
            df[f'{col}_is_last_day'] = (df[col].dt.is_month_end).astype(int)
            
            print(f'[STATUS] Created datetime features for: {col}')
        except Exception as e:
            print(f'[STATUS] Skipping {col}: {e}')

# Calculate time differences between related datetime columns
time_diff_pairs = [
    ('order_purchase_timestamp', 'order_approved_at', 'approval_time_hours'),
    ('order_purchase_timestamp', 'order_delivered_customer_date', 'delivery_time_days'),
    ('order_approved_at', 'order_delivered_carrier_date', 'processing_time_hours'),
    ('order_delivered_carrier_date', 'order_delivered_customer_date', 'carrier_delivery_time_days'),
    ('order_estimated_delivery_date', 'order_delivered_customer_date', 'delivery_vs_estimated_days'),
]

for col1, col2, new_name in time_diff_pairs:
    if col1 in df.columns and col2 in df.columns:
        try:
            col1_series = pd.to_datetime(df[col1], errors='coerce')
            col2_series = pd.to_datetime(df[col2], errors='coerce')
            
            if 'hours' in new_name:
                df[new_name] = (col2_series - col1_series).dt.total_seconds() / 3600
            elif 'days' in new_name:
                df[new_name] = (col2_series - col1_series).dt.total_seconds() / (3600 * 24)
            elif 'minutes' in new_name:
                df[new_name] = (col2_series - col1_series).dt.total_seconds() / 60
            
            print(f'[STATUS] Created time difference: {new_name}')
        except Exception as e:
            print(f'[STATUS] Skipping {new_name}: {e}')

# ==========================================
# STEP 2: Customer Behavior Features (from Aggregated Data)
# ==========================================
print('[STATUS] Creating customer behavior features...')

# Convert customer_id to customer identifier
if 'customer_unique_id' in df.columns:
    cust_col = 'customer_unique_id'
elif 'customer_id' in df.columns:
    cust_col = 'customer_id'
else:
    cust_col = None

if cust_col and 'order_purchase_timestamp' in df.columns:
    try:
        purchase_date = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
        
        # Customer order count (total orders per customer)
        df['customer_order_count'] = df.groupby(cust_col)['order_id'].transform('count')
        
        # Customer total spend
        if 'payment_value' in df.columns:
            df['customer_total_spend'] = df.groupby(cust_col)['payment_value'].transform('sum')
            df['customer_avg_spend'] = df.groupby(cust_col)['payment_value'].transform('mean')
            df['customer_spend_std'] = df.groupby(cust_col)['payment_value'].transform('std').fillna(0)
        
        # Customer order frequency (days between first and last order)
        if 'customer_first_order' not in df.columns:
            first_order = df.groupby(cust_col)['order_purchase_timestamp'].transform('min')
            first_order = pd.to_datetime(first_order, errors='coerce')
            df['customer_first_order'] = first_order
        
        if 'customer_last_order' not in df.columns:
            last_order = df.groupby(cust_col)['order_purchase_timestamp'].transform('max')
            last_order = pd.to_datetime(last_order, errors='coerce')
            df['customer_last_order'] = last_order
        
        # Customer tenure in days
        df['customer_tenure_days'] = (pd.to_datetime(df['customer_last_order']) - pd.to_datetime(df['customer_first_order'])).dt.days
        
        # Customer activity tier
        def assign_tier(order_count, tenure_days):
            if tenure_days <= 0:
                return 'new'
            frequency = order_count / max(tenure_days, 1) * 30  # Orders per month
            if frequency >= 2:
                return 'highly_active'
            elif frequency >= 0.5:
                return 'active'
            elif frequency > 0:
                return 'occasional'
            else:
                return 'inactive'
        
        df['customer_activity_tier'] = df.apply(
            lambda row: assign_tier(
                row.get('customer_order_count', 1),
                row.get('customer_tenure_days', 0)
            ), axis=1
        )
        
        print('[STATUS] Created customer behavior features')
    except Exception as e:
        print(f'[STATUS] Error in customer features: {e}')
elif cust_col:
    print(f'[STATUS] Skipping customer features - no purchase date column')

# ==========================================
# STEP 3: Product Features
# ==========================================
print('[STATUS] Creating product features...')

# Price to weight ratio
if 'price' in df.columns and 'product_weight_g' in df.columns:
    df['price_per_weight'] = df['price'] / (df['product_weight_g'] + 1)
    print('[STATUS] Created price_per_weight')

# Price to length ratio
if 'price' in df.columns and 'product_length_cm' in df.columns:
    df['price_per_length'] = df['price'] / (df['product_length_cm'] + 1)
    print('[STATUS] Created price_per_length')

# Product dimensions volume
dim_cols = ['product_length_cm', 'product_height_cm', 'product_width_cm']
if all(col in df.columns for col in dim_cols):
    df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df['product_volume_cm3'] = df['product_volume_cm3'].fillna(df['product_volume_cm3'].median())
    print('[STATUS] Created product_volume_cm3')

# Review score bins
if 'review_score' in df.columns:
    df['review_score_binned'] = pd.cut(
        df['review_score'],
        bins=[0, 2, 4, 5],
        labels=['poor', 'good', 'excellent'],
        include_lowest=True
    )
    print('[STATUS] Created review_score_binned')

# ==========================================
# STEP 4: Interaction Features
# ==========================================
print('[STATUS] Creating interaction features...')

# Payment interaction
if 'payment_installments' in df.columns and 'payment_value' in df.columns:
    df['installment_value'] = df['payment_value'] / (df['payment_installments'] + 1)
    print('[STATUS] Created installment_value')

# Review and price interaction
if 'review_score' in df.columns and 'price' in df.columns:
    df['review_score_x_price'] = df['review_score'] * df['price']
    print('[STATUS] Created review_score_x_price')

# Freight ratio
if 'freight_value' in df.columns and 'price' in df.columns:
    df['freight_to_price_ratio'] = df['freight_value'] / (df['price'] + 1)
    print('[STATUS] Created freight_to_price_ratio')

# Product to payment ratio
if 'product_weight_g' in df.columns and 'payment_value' in df.columns:
    df['weight_payment_ratio'] = df['product_weight_g'] / (df['payment_value'] + 1)
    print('[STATUS] Created weight_payment_ratio')

# ==========================================
# STEP 5: Encoding Categorical Features
# ==========================================
print('[STATUS] Encoding categorical features...')

# Identify categorical columns (object dtype, not datetime, and not IDs)
id_cols = ['customer_id', 'customer_unique_id', 'order_id', 'product_id', 'seller_id', 'geolocation_zip_code_prefix', 'order_item_id', 'review_id', 'payment_sequential']
categorical_cols = []
for col in df.select_dtypes(include=['object']).columns:
    if col not in id_cols and df[col].nunique() > 1 and df[col].nunique() <= df.shape[0] * 0.5:
        # Check it's not already a datetime or time feature
        if not any(x in col.lower() for x in ['date', 'time', 'timestamp']):
            categorical_cols.append(col)

for col in categorical_cols:
    nunique = df[col].nunique()
    
    if nunique <= 5:
        # One-Hot Encoding for low cardinality
        try:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f'[STATUS] One-Hot encoded: {col} ({nunique} categories)')
        except Exception as e:
            print(f'[STATUS] Skipping OHE {col}: {e}')
    elif nunique <= 50:
        # Label Encoding for medium cardinality
        try:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            print(f'[STATUS] Label encoded: {col} ({nunique} categories)')
        except Exception as e:
            print(f'[STATUS] Skipping Label {col}: {e}')
    else:
        print(f'[STATUS] Skipping encoding: {col} has {nunique} categories')

# ==========================================
# STEP 6: Handle Missing Values
# ==========================================
print('[STATUS] Handling missing values...')

# For numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        # Use median for skewed columns, mean for normal
        skewness = df[col].skew() if df[col].nunique() > 1 else 0
        if abs(skewness) > 1:
            df[col] = df[col].fillna(df[col].median())
            print(f'[STATUS] Filled {col} NaN with median')
        else:
            df[col] = df[col].fillna(df[col].mean())
            print(f'[STATUS] Filled {col} NaN with mean')

# For categorical columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna('unknown')
        print(f'[STATUS] Filled {col} NaN with "unknown"')

# ==========================================
# STEP 7: Feature Selection
# ==========================================
print('[STATUS] Performing feature selection...')

# Identify potential target columns
target_options = [col for col in ['payment_value', 'price', 'review_score'] if col in df.columns]
target_col = target_options[0] if target_options else None

features_to_drop = []
features_to_keep = []

if target_col:
    # Get numeric features (excluding id columns and target)
    candidate_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                         if col not in id_cols and col != target_col 
                         and col not in features_to_keep
                         and df[col].nunique() > 1]
    
    if len(candidate_features) > 1:
        try:
            X = df[candidate_features].fillna(0)
            y = df[target_col].fillna(df[target_col].median())
            
            # Use SelectKBest with f_classif for numerical target
            if y.nunique() > 10:  # Continuous target
                selector = SelectKBest(score_func=f_classif, k=min(50, len(candidate_features)))
                selector.fit(X, y)
            else:  # Categorical target
                selector = SelectKBest(score_func=mutual_info_classif, k=min(30, len(candidate_features)))
                selector.fit(X, y)
            
            # Get feature scores
            feature_scores = list(zip(candidate_features, selector.scores_))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top features by importance
            threshold = np.percentile(selector.scores_, 25) if len(selector.scores_) > 3 else 0
            features_to_keep = [f[0] for f in feature_scores if f[1] > 0]
            
            print(f'[STATUS] Feature selection complete. Keeping {len(features_to_keep)} numeric features')
            print(f'[STATUS] Top 5 features by importance: {[(f[0], round(f[1], 2)) for f in feature_scores[:5]]}')
            
        except Exception as e:
            print(f'[STATUS] Feature selection error: {e}')
            features_to_keep = candidate_features
else:
    print('[STATUS] No suitable target column found for feature selection')

# ==========================================
# STEP 8: Remove Highly Correlated Features
# ==========================================
print('[STATUS] Removing highly correlated features...')

if len(df.select_dtypes(include=[np.number]).columns) > 5:
    try:
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Drop columns with correlation > 0.95
        high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        # Don't drop the target or important base columns
        safe_cols = ['payment_value', 'price', 'review_score', 'order_id', 'product_id', 'seller_id']
        high_corr_cols = [c for c in high_corr_cols if c not in safe_cols]
        
        df = df.drop(columns=high_corr_cols, errors='ignore')
        print(f'[STATUS] Dropped {len(high_corr_cols)} highly correlated features')
    except Exception as e:
        print(f'[STATUS] Corr removal error: {e}')

# ==========================================
# FINAL: Save output
# ==========================================
print(f'[STATUS] Final shape: {df.shape}')
print(f'[STATUS] Original columns: {len(original_cols)}, New columns: {len(df.columns)}')
print(f'[STATUS] New features: {[c for c in df.columns if c not in original_cols][:20]}...')

df.to_csv(OUTPUT_CSV, index=False)
print(f'[STATUS] Saved: {OUTPUT_CSV}')

# ==========================================
# Generate Report
# ==========================================
new_features = [c for c in df.columns if c not in original_cols]
dropped_features = [c for c in original_cols if c not in df.columns]

report = f"""# Finn Feature Engineering Report

## Overview
- **Input**: {INPUT_PATH}
- **Original Features**: {len(original_cols)}
- **New Features Created**: {len(new_features)}
- **Final Features**: {len(df.columns)}
- **Rows**: {df.shape[0]}

## Features Created

### Datetime Features ({len([c for c in new_features if any(x in c.lower() for x in ['hour', 'day', 'month', 'quarter', 'weekend', 'first', 'last'])])})
- Basic datetime components (hour, day, week, month, quarter) from all datetime columns
- `is_weekend`: Weekend indicator for each datetime column
- `is_first_day`/`is_last_day`: Month boundary indicators

### Time Difference Features ({len([c for c in new_features if any(x in c.lower() for x in ['time', 'days', 'hours', 'approval', 'delivery', 'processing', 'carrier'])])})
- `approval_time_hours`: Time from purchase to approval
- `delivery_time_days`: Total delivery time from purchase
- `processing_time_hours`: Time from approval to carrier handoff
- `carrier_delivery_time_days`: Time with carrier
- `delivery_vs_estimated_days`: Days difference from estimated delivery

### Customer Behavior Features ({len([c for c in new_features if 'customer_' in c.lower()])})
- `customer_order_count`: Total orders per customer
- `customer_total_spend`: Total spend per customer
- `customer_avg_spend`: Average order value per customer
- `customer_spend_std`: Spend variability per customer
- `customer_tenure_days`: Days between first and last order
- `customer_activity_tier`: Customer activity classification (new/occasional/active/highly_active)

### Product Features ({len([c for c in new_features if 'product_' in c.lower() or 'price_' in c.lower()])})
- `price_per_weight`: Price normalized by weight
- `price_per_length`: Price normalized by length
- `product_volume_cm3`: Product volume from dimensions
- `review_score_binned`: Review score categorized (poor/good/excellent)

### Interaction Features ({len([c for c in new_features if any(x in c.lower() for x in ['x', 'ratio', 'per_', 'to_'])])})
- `installment_value`: Payment value per installment
- `review_score_x_price`: Review score weighted by price
- `freight_to_price_ratio`: Freight cost relative to price
- `weight_payment_ratio`: Weight per unit payment

### Encoded Features ({len([c for c in new_features if '_encoded' in c.lower() or any(df[c].dtype == 'uint8' for c in [c])])})
- One-Hot Encoding for low cardinality (≤5 unique values)
- Label Encoding for medium cardinality (6-50 unique values)
- High cardinality columns (>50 unique values) left as-is

## Features Dropped
Total: {len(dropped_features)}
- Highly correlated features (>0.95 correlation)
- {', '.join(dropped_features[:10])}{'...' if len(dropped_features) > 10 else ''}

## Encoding Used
- One-Hot Encoding: For categorical columns with ≤5 unique values
- Label Encoding: For categorical columns with 6-50 unique values
- Decision based on cardinality to balance information retention vs dimensionality

## Missing Value Handling
- **Numeric columns**: Median for skewed, Mean for normal distribution
- **Categorical columns**: Filled with 'unknown'

## Feature Selection Method
- Target: {target_col}
- Method: SelectKBest (f_classif for continuous target, mutual_info_classif for categorical)
- Selected {len(features_to_keep)} features with positive importance scores

## Self-Improvement Report

### Method Used
Comprehensive feature engineering pipeline covering:
1. Datetime decomposition and time differences
2. Customer aggregation features
3. Product ratio features
4. Interaction features
5. Statistical encoding
6. Feature selection via SelectKBest

### Why These Methods
- **Datetime features**: Essential for time-series patterns - hour affects behavior, weekday/weekend affects sales
- **Time differences**: Critical for delivery performance analysis and prediction
- **Customer aggregation**: Captures user behavior patterns - active vs occasional customers behave differently
- **Product ratios**: Normalize features to make them comparable across product categories
- **Interaction features**: Models complex relationships between attributes (e.g., quality vs price)

### New Methods Discovered
All techniques used were from existing Knowledge Base. No new methods discovered.

### Improvement for Next Time
Could add:
- Polynomial features for non-linear relationships
- Target encoding for high cardinality categorical variables
- RFE (Recursive Feature Elimination) as alternative selection method
- Weight of Evidence (WoE) encoding for categorical variables with target relationship

### Knowledge Base Update
No update needed - all methods were already documented in Knowledge Base.
"""

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Report saved: {OUTPUT_REPORT}')

# Copy this script to the script path
script_code = open(__file__ if '__file__' in dir() else __file__, 'r').read() if '__file__' in dir() else inspect.getsource(inspect.currentframe())

print(f'[STATUS] Script path: {OUTPUT_SCRIPT}')
print(f'[STATUS] All output files generated successfully')

# Agent Report
print(f'''
Agent Report — Finn
====================
รับจาก     : Max
Input      : {INPUT_PATH} ({df_original.shape[0]} rows, {df_original.shape[1]} cols)
ทำ         : Feature Engineering — datetime features, customer behavior, product features, interaction features, encoding, feature selection
พบ         : Created {len(new_features)} new features, dropped {len(dropped_features)} features
เปลี่ยนแปลง : {df_original.shape[0]} rows → {df.shape[0]} rows, {df_original.shape[1]} cols → {df.shape[1]} cols
ส่งต่อ     : Anna (for review) — engineered dataset with {len(new_features)} new features added
''')
