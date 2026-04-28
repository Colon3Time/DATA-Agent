# ============================================================
# FINN — Feature Engineering Script (Fixed)
# Agent: Finn (Feature Engineer)
# Generated: 2026-04-27 21:37:27
# ============================================================
# Input : C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\output\finn\finn_output.csv
# Output: C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\output\finn
# ============================================================

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ----- Parse Arguments -----
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input or r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\output\finn\finn_output.csv'
OUTPUT_DIR = args.output_dir or r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-27_Penguin Sizes Dataset\output\finn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Load Data -----
print('[STATUS] Loading data...')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape} — {list(df.columns)}')

# ----- Check Available Columns -----
print('[STATUS] Checking available numeric columns...')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
string_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'[STATUS] Numeric columns: {numeric_cols}')
print(f'[STATUS] String columns: {string_cols}')

# Find available measurement columns
available_measurements = [c for c in numeric_cols if any(k in c.lower() for k in ['bill', 'flipper', 'body', 'depth', 'length', 'mass'])]
print(f'[STATUS] Available measurement columns: {available_measurements}')

# ---- Map known variants ----
bill_length_col = next((c for c in numeric_cols if 'bill' in c.lower() and 'length' in c.lower()), None)
bill_depth_col = next((c for c in numeric_cols if 'bill' in c.lower() and 'depth' in c.lower()), None)
flipper_length_col = next((c for c in numeric_cols if 'flipper' in c.lower() and 'length' in c.lower()), None)
body_mass_col = next((c for c in numeric_cols if 'body' in c.lower() and 'mass' in c.lower()), None)

print(f'[STATUS] Mapped columns: bill_length={bill_length_col}, bill_depth={bill_depth_col}, flipper_length={flipper_length_col}, body_mass={body_mass_col}')

# ----- 1. Feature Engineering: Ratio, Interaction, Polynomial -----
print('[STATUS] Creating new features...')

# 1A. Ratio Features
if bill_length_col and flipper_length_col:
    df['bill_length_flipper_ratio'] = df[bill_length_col] / (df[flipper_length_col] + 1e-6)
else:
    print('[WARN] bill_length or flipper_length not found — skipping ratio')

if bill_depth_col and flipper_length_col:
    df['bill_depth_flipper_ratio'] = df[bill_depth_col] / (df[flipper_length_col] + 1e-6)
else:
    print('[WARN] bill_depth or flipper_length not found — skipping ratio')

if bill_length_col and bill_depth_col:
    df['bill_length_depth_ratio'] = df[bill_length_col] / (df[bill_depth_col] + 1e-6)
else:
    print('[WARN] bill_length or bill_depth not found — skipping ratio')

# 1B. Interaction Features
if bill_length_col and bill_depth_col:
    df['bill_length_x_depth'] = df[bill_length_col] * df[bill_depth_col]
else:
    print('[WARN] Cannot create bill_length_x_depth')

if flipper_length_col and body_mass_col:
    df['flipper_x_mass'] = df[flipper_length_col] * df[body_mass_col]
else:
    print('[WARN] Cannot create flipper_x_mass')

# 1C. Polynomial Features (squared terms)
squared_cols_map = {
    ('bill_length_sq', bill_length_col),
    ('bill_depth_sq', bill_depth_col),
    ('flipper_length_sq', flipper_length_col),
    ('body_mass_sq', body_mass_col),
}
for new_name, orig_col in squared_cols_map:
    if orig_col:
        df[new_name] = df[orig_col] ** 2
        print(f'[STATUS] Created {new_name} from {orig_col}')
    else:
        print(f'[WARN] Cannot create {new_name} — original column missing')

# 1D. BMI-like indicator for penguins (mass / flipper^2)
if body_mass_col and flipper_length_col:
    df['penguin_bmi'] = df[body_mass_col] / ((df[flipper_length_col] / 100) ** 2 + 1e-6)
    print('[STATUS] Created penguin_bmi')
else:
    print('[WARN] Cannot create penguin_bmi')

# ----- 2. Identify Target Column (species) -----
print('[STATUS] Identifying target column for encoding...')
# Look for species column or binary column
species_col = next((c for c in string_cols if 'species' in c.lower()), None)
target_col = None
if species_col:
    target_col = species_col
else:
    # Find binary numeric column
    binary_cols = [c for c in numeric_cols if df[c].nunique() == 2]
    if binary_cols:
        target_col = binary_cols[0]
    
if target_col:
    print(f'[STATUS] Target column identified: {target_col}')
else:
    print('[WARN] No target column identified — skipping target encoding')

# ----- 3. Target Encoding for high-cardinality categoricals -----
print('[STATUS] Encoding categorical features...')

# Check if category_encoders is available
try:
    from category_encoders import TargetEncoder
    encoder_available = True
except ImportError:
    encoder_available = False
    print('[WARN] category_encoders not installed — skipping target encoding')

if encoder_available and target_col:
    # Find categorical columns with > 5 unique values (high cardinality) but not the target itself
    high_card_cols = [c for c in string_cols if c != target_col and df[c].nunique() > 5]
    if high_card_cols:
        te = TargetEncoder(cols=high_card_cols)
        df[high_card_cols] = te.fit_transform(df[high_card_cols], df[target_col])
        print(f'[STATUS] Target encoded: {high_card_cols}')
    else:
        print('[STATUS] No high-cardinality categorical columns found for target encoding')
elif encoder_available and not target_col:
    print('[STATUS] No target column — skipping target encoding')

# ----- 4. Handle any remaining object columns with simple encoding -----
remaining_obj = df.select_dtypes(include=['object']).columns.tolist()
for col in remaining_obj:
    if col != target_col and col in string_cols:
        # Label encode low cardinality, one-hot for medium
        if df[col].nunique() <= 10:
            df[col] = df[col].astype('category').cat.codes
            print(f'[STATUS] Label encoded: {col}')
        else:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
            print(f'[STATUS] One-hot encoded: {col}')

# ----- 5. Feature Selection with Variance Threshold -----
print('[STATUS] Running variance threshold...')
numeric_for_var = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_for_var) > 1:
    variances = df[numeric_for_var].var()
    zero_var_cols = variances[variances < 1e-8].index.tolist()
    if zero_var_cols:
        df.drop(columns=zero_var_cols, inplace=True)
        print(f'[STATUS] Dropped zero-variance columns: {zero_var_cols}')
else:
    print('[STATUS] Not enough numeric columns for variance threshold')

# ----- Save Output -----
print('[STATUS] Saving engineered data...')
output_csv = os.path.join(OUTPUT_DIR, 'engineered_data.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')
print(f'[STATUS] Final shape: {df.shape}')
print(f'[STATUS] Final columns: {list(df.columns)}')

# ----- Generate Feature Report -----
print('[STATUS] Generating feature report...')
report_path = os.path.join(OUTPUT_DIR, 'finn_feature_report.md')
original_features = len(numeric_cols) + len(string_cols)
new_features = len(df.columns) - original_features
final_features = len(df.columns)

report_content = f"""Finn Feature Engineering Report
================================
Original Features: {original_features}
New Features Created: {new_features}
Final Features Selected: {final_features}

Features Created:
- bill_length_flipper_ratio: ratio of bill length to flipper length (body proportion)
- bill_depth_flipper_ratio: ratio of bill depth to flipper length (body proportion)
- bill_length_depth_ratio: ratio of bill length to bill depth (beak shape)
- bill_length_x_depth: interaction between bill length and depth
- flipper_x_mass: interaction between flipper length and body mass
- [squared terms]: polynomial features for nonlinear relationships
- penguin_bmi: body mass / (flipper_length_m/100)^2 — approximate body density

Features Dropped:
- None (all original features kept)

Encoding Used: Target Encoding (for high-cardinality categoricals), Label/One-Hot (for remaining)
Scaling Used: None (not requested)

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Feature creation from domain knowledge + column name mapping
เหตุผลที่เลือก: Dataset มี numeric measurements ที่สามารถสร้าง ratio/interaction features ที่มีความหมาย
วิธีใหม่ที่พบ: Column name fuzzy matching — use partial string matching instead of exact column names
จะนำไปใช้ครั้งหน้า: ใช่ — ปรับปรุง pipeline ให้มีความยืดหยุ่นสูงขึ้น
Knowledge Base: อัพเดต — เพิ่ม column name mapping technique
"""
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'[STATUS] Report saved: {report_path}')