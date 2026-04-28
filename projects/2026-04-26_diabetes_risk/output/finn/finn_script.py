import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIG ==========
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
parser.add_argument('--target', default='Outcome')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
TARGET = args.target

# ถ้า input ไม่มีค่า ให้ใช้ default
if not INPUT_PATH:
    INPUT_PATH = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_diabetes_risk\input\pima_indians_diabetes.csv"

if not OUTPUT_DIR:
    OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_diabetes_risk\output\finn"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[STATUS] Input: {INPUT_PATH}")
print(f"[STATUS] Output Dir: {OUTPUT_DIR}")

# ========== LOAD DATA ==========
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded data: {df.shape}")
print(f"[STATUS] Columns: {list(df.columns)}")

# หาชื่อ target column อัตโนมัติถ้า 'Outcome' ไม่มีใน dataset
all_columns = list(df.columns)
if TARGET in all_columns:
    print(f"[STATUS] Target column found: '{TARGET}'")
else:
    # ลองหาชื่อ target ที่เป็นไปได้
    possible_targets = [c for c in all_columns if c.lower() in ['outcome', 'class', 'target', 'label', 'y', 'diabetes']]
    if possible_targets:
        TARGET = possible_targets[0]
        print(f"[STATUS] Target column auto-detected: '{TARGET}'")
    else:
        # เอาคอลัมน์สุดท้ายเป็น target (common in ML datasets)
        TARGET = all_columns[-1]
        print(f"[STATUS] Target column fallback (last column): '{TARGET}'")

print(f"[STATUS] Target value counts:\n{df[TARGET].value_counts()}")

# แยก X และ y
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ========== 1. INTERACTION FEATURES ==========
print("\n[STEP 1] Creating interaction features...")

# หา numerical columns ที่เหมาะสม
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"  Numeric columns available: {len(numeric_cols)}")

# สร้าง interaction pairs จาก columns ที่มีใน dataset
interaction_pairs = []
for col1 in numeric_cols:
    for col2 in numeric_cols:
        if col1 < col2:  # ป้องกันคู่ซ้ำ
            interaction_pairs.append((col1, col2))

# ถ้ามีมากเกินไป ให้ใช้เฉพาะที่มีความสำคัญสูง (Glucose, BMI, Age, Pregnancies-like)
if len(interaction_pairs) > 10:
    priority_cols = [c for c in numeric_cols if c.lower() in ['glucose', 'bmi', 'age', 'pregnancies', 'bloodpressure', 'insulin', 'skinthickness', 'dpf']]
    if len(priority_cols) >= 2:
        interaction_pairs = [(c1, c2) for i, c1 in enumerate(priority_cols) for c2 in priority_cols[i+1:]]
    else:
        interaction_pairs = interaction_pairs[:10]  # จำกัดจำนวน

for col1, col2 in interaction_pairs:
    new_name = f"{col1}x{col2}"
    if col1 in X.columns and col2 in X.columns:
        X[new_name] = X[col1] * X[col2]
        print(f"  [+] Created: {new_name}")

# ========== 2. POLYNOMIAL FEATURES (degree=2) ==========
print("\n[STEP 2] Creating polynomial features (degree=2)...")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# เลือกเฉพาะ non-binary numeric columns
poly_cols = [c for c in numeric_cols if X[c].nunique() > 2]

if len(poly_cols) > 0:
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_data = poly.fit_transform(X[poly_cols])
    
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=X.index)
    
    # Filter: เอาเฉพาะ interaction terms และ squared terms (ที่สร้างใหม่)
    original_cols_set = set(poly_cols)
    new_poly_features = [name for name in poly_feature_names if name not in original_cols_set]
    
    # จำกัดจำนวน poly features (ถ้ามีมากเกินไป)
    if len(new_poly_features) > 50:
        new_poly_features = new_poly_features[:50]
    
    poly_new = poly_df[new_poly_features]
    
    # แนบ poly features
    X = pd.concat([X, poly_new], axis=1)
    print(f"  [+] Added {len(new_poly_features)} polynomial features")

# ========== 3. BINNING FEATURES ==========
print("\n[STEP 3] Creating binned features...")

bin_cols = [c for c in numeric_cols if X[c].nunique() > 5 and c not in poly_cols]
if len(bin_cols) > 3:
    bin_cols = bin_cols[:5]  # จำกัดจำนวน

if len(bin_cols) > 0:
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    bin_data = discretizer.fit_transform(X[bin_cols])
    
    for i, col in enumerate(bin_cols):
        bin_name = f"{col}_binned"
        X[bin_name] = bin_data[:, i].astype(int)
        print(f"  [+] Created: {bin_name}")

# ========== 4. FEATURE SELECTION WITH RFECV ==========
print("\n[STEP 4] Running RFECV feature selection...")

X_selected = X.copy()

try:
    # ใช้เฉพาะ numeric columns
    X_num = X_selected.select_dtypes(include=[np.number])
    
    # แก้ NaN/Inf
    X_num = X_num.fillna(X_num.median())
    X_num = X_num.replace([np.inf, -np.inf], 0)
    
    # จำกัด features เพื่อให้ RFECV ทำงานได้
    max_features_rfe = min(30, X_num.shape[1])
    
    if max_features_rfe >= 2:
        # เลือก variance threshold ก่อน
        from sklearn.feature_selection import VarianceThreshold
        vt = VarianceThreshold(threshold=0.01 * (1 - 0.01))
        X_vt = vt.fit_transform(X_num)
        selected_mask = vt.get_support()
        X_vt_cols = X_num.columns[selected_mask].tolist()
        
        # ถ้าเหลือน้อยเกินไป ให้ใช้ทั้งหมด
        if X_vt.shape[1] < 3:
            X_vt = X_num.values
            X_vt_cols = X_num.columns.tolist()
        
        # RFECV
        rfe_cols = X_vt_cols[:max_features_rfe]
        X_rfe = pd.DataFrame(X_vt[:, :len(rfe_cols)], columns=rfe_cols, index=X_num.index)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=clf, cv=cv, scoring='f1_weighted', n_jobs=-1, min_features_to_select=max(1, len(rfe_cols)//4))
        rfecv.fit(X_rfe, y)
        
        selected_idx = rfecv.support_
        selected_cols = [rfe_cols[i] for i in range(len(rfe_cols)) if selected_idx[i]]
        
        # เก็บเฉพาะ features ที่ถูกเลือก
        all_selected_cols = selected_cols + [c for c in X_selected.columns if c not in X_num.columns]
        X_selected = X_selected[all_selected_cols]
        
        print(f"  [+] RFECV selected {len(selected_cols)}/{len(rfe_cols)} features")
    else:
        print(f"  [!] Too few features ({max_features_rfe}) for RFECV, skipping")
        
except Exception as e:
    print(f"  [!] RFECV error: {e}")
    print(f"  [+] Using all features without selection")

# ========== 5. SCALING ==========
print("\n[STEP 5] Scaling numerical features...")

X_num_final = X_selected.select_dtypes(include=[np.number])
X_cat_final = X_selected.select_dtypes(exclude=[np.number])

try:
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_num_final.fillna(X_num_final.median()).replace([np.inf, -np.inf], 0)),
        columns=X_num_final.columns,
        index=X_selected.index
    )
    
    X_scaled = pd.concat([X_num_scaled, X_cat_final], axis=1)
    print(f"  [+] Scaled {len(X_num_final.columns)} numerical features")
    
except Exception as e:
    print(f"  [!] Scaling error: {e}")
    X_scaled = X_selected

# ========== 6. FINAL DATASET ==========
print("\n[STEP 6] Building final dataset...")

df_final = pd.concat([X_scaled, y], axis=1)

# แก้ NaN และ Inf สุดท้าย
df_final = df_final.fillna(0).replace([np.inf, -np.inf], 0)

print(f"[STATUS] Final dataset shape: {df_final.shape}")
print(f"[STATUS] Total features (including target): {len(df_final.columns)}")

# ========== SAVE OUTPUT ==========
output_csv = os.path.join(OUTPUT_DIR, 'finn_output.csv')
df_final.to_csv(output_csv, index=False)
print(f"[STATUS] Saved: {output_csv}")

# ========== REPORT ==========
report = f"""Finn Feature Engineering Report
================================
Dataset: {os.path.basename(INPUT_PATH)}
Target Column: '{TARGET}'
Original Features: {df.shape[1] - 1}
New Features Created: {X_scaled.shape[1] - (df.shape[1] - 1)}
Final Features Selected: {X_scaled.shape[1]}

Features Created:
- Interaction features: {len(interaction_pairs)} pairs created
- Polynomial features (deg=2): added via PolynomialFeatures
- Binned features: quantile-based discretization

Encoding Used: None (all numeric data)
Scaling Used: StandardScaler (zero mean, unit variance)

Feature Selection:
- RFECV with RandomForestClassifier performed (if applicable)
- Final feature count: {X_scaled.shape[1]}
"""

report_path = os.path.join(OUTPUT_DIR, 'finn_feature_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"[STATUS] Report saved: {report_path}")

print("\n[STATUS] Finn pipeline completed successfully!")