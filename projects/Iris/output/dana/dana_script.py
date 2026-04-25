import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

_FALLBACK_INPUT  = r"C:\Users\Amorntep\DATA-Agent\projects\iris\input\iris.csv"
_FALLBACK_OUTPUT = r"C:\Users\Amorntep\DATA-Agent\projects\iris\output\dana"

OUTPUT_DIR = args.output_dir or _FALLBACK_OUTPUT
os.makedirs(OUTPUT_DIR, exist_ok=True)

_input = Path(args.input) if args.input else Path(_FALLBACK_INPUT)
INPUT_PATH = str(_input) if _input.is_file() else str(next((_input.glob("*.csv")), Path(_FALLBACK_INPUT)))

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dana_output.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"[STATUS] Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"[STATUS] Columns: {list(df.columns)}")

initial_rows = len(df)
initial_cols = len(df.columns)
missing_before = {k: v for k, v in df.isnull().sum().to_dict().items() if v > 0}

if missing_before:
    print(f"[STATUS] Missing values found: {missing_before}")
else:
    print(f"[STATUS] No missing values found")

# ── Clean ─────────────────────────────────────────────────────────────────────
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count == 0:
        continue
    pct = missing_count / len(df) * 100
    print(f"[STATUS] '{col}': {missing_count} missing ({pct:.1f}%) → fill median")
    df[col] = df[col].fillna(df[col].median())

# Outlier detection (IQR)
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c.lower() not in ["id", "index"]]
outlier_log = {}
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    if mask.sum() > 0:
        outlier_log[col] = int(mask.sum())
        print(f"[STATUS] '{col}': {mask.sum()} outliers (IQR)")

# Type conversion
for col in df.select_dtypes(include="object").columns:
    try:
        df[col] = pd.to_datetime(df[col])
        print(f"[STATUS] '{col}' → datetime")
    except Exception:
        df[col] = df[col].astype("category")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
final_rows, final_cols = len(df), len(df.columns)
print(f"[STATUS] === CLEANING COMPLETE ===")
print(f"[STATUS] Rows: {initial_rows} -> {final_rows} | Cols: {initial_cols} -> {final_cols}")
print(f"[STATUS] Outliers found: {sum(outlier_log.values())}")
print(f"[STATUS] Saved to: {OUTPUT_PATH}")
df_check = pd.read_csv(OUTPUT_PATH)
print(f"[STATUS] Verified: {df_check.shape[0]} rows x {df_check.shape[1]} cols")
