import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent.parent.parent  # projects/2026-04-23_thailand_employment/
RAW_PATH   = BASE / "input" / "thailand_employment_worldbank_2000_2024.csv"
CLEAN_PATH = BASE / "input" / "thailand_employment_clean.csv"

# ── Load raw ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(RAW_PATH)

print("=" * 60)
print("DANA — DATA CLEANING REPORT")
print("=" * 60)

# ── 1. Overview ───────────────────────────────────────────────────────────────
print("\n[1] DATASET OVERVIEW")
print(f"  Rows    : {df_raw.shape[0]}")
print(f"  Columns : {df_raw.shape[1]}")
print(f"  Year range: {df_raw['year'].min()} – {df_raw['year'].max()}")

# ── 2. Missing Values ─────────────────────────────────────────────────────────
print("\n[2] MISSING VALUES")
missing = df_raw.isnull().sum()
print(missing.to_string())
print(f"\n  Total missing: {missing.sum()}")

# ── 3. Data Types ─────────────────────────────────────────────────────────────
print("\n[3] DATA TYPES")
print(df_raw.dtypes.to_string())

# ── 4. Duplicates ─────────────────────────────────────────────────────────────
print("\n[4] DUPLICATES")
print(f"  Duplicate rows: {df_raw.duplicated().sum()}")

# ── 5. Consistency Check ──────────────────────────────────────────────────────
print("\n[5] SECTOR EMPLOYMENT CONSISTENCY (Agri + Industry + Services ~= 100%)")
df_work = df_raw.copy()
df_work["sector_sum"] = (
    df_work["employment_agriculture_pct"]
    + df_work["employment_industry_pct"]
    + df_work["employment_services_pct"]
)
df_work["sector_ok"] = df_work["sector_sum"].between(99.5, 100.5)
fail_rows = df_work[~df_work["sector_ok"]][["year", "sector_sum"]]
if fail_rows.empty:
    print("  OK All rows pass (sum within 99.5-100.5%)")
else:
    print("  FAIL rows:")
    print(fail_rows.to_string(index=False))

# ── 6. Outlier Detection — IQR ───────────────────────────────────────────────
print("\n[6] OUTLIER DETECTION (IQR method, threshold = 2.0 x IQR)")
numeric_cols = [c for c in df_raw.columns if c not in ("year",)]
outlier_log = []
for col in numeric_cols:
    q1, q3 = df_raw[col].quantile(0.25), df_raw[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 2.0 * iqr, q3 + 2.0 * iqr
    hits = df_raw[(df_raw[col] < lower) | (df_raw[col] > upper)][["year", col]]
    for _, row in hits.iterrows():
        outlier_log.append({"year": int(row["year"]), "column": col,
                             "value": row[col], "lower": round(lower, 3), "upper": round(upper, 3)})
if outlier_log:
    print(pd.DataFrame(outlier_log).to_string(index=False))
else:
    print("  OK No outliers detected")

# ── 7. Structural Break ───────────────────────────────────────────────────────
print("\n[7] STRUCTURAL BREAK — YoY Change > 4pp")
df_sorted = df_raw.sort_values("year").reset_index(drop=True)
flag_cols = ["employment_agriculture_pct", "employment_industry_pct",
             "employment_services_pct", "vulnerable_employment_pct"]
for col in flag_cols:
    yoy = df_sorted[col].diff()
    breaks = df_sorted[yoy.abs() > 4][["year", col]].copy()
    breaks["yoy_change"] = yoy[breaks.index].values
    if not breaks.empty:
        print(f"  WARNING  {col}:")
        print(breaks.to_string(index=False))

# ── 8. Build Clean Dataset ────────────────────────────────────────────────────
df_clean = df_sorted.copy()
df_clean["structural_shock"] = df_clean["year"].isin([2009, 2020, 2021]).astype(int)
df_clean["anomaly_flag"]     = df_clean["year"].isin([2013, 2014]).astype(int)

# ── 9. BEFORE / AFTER ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BEFORE / AFTER COMPARISON")
print("=" * 60)

key_cols = ["unemployment_rate_pct", "employment_agriculture_pct",
            "employment_services_pct", "vulnerable_employment_pct", "gdp_per_capita_usd"]

before = df_raw[key_cols].describe().loc[["mean", "std", "min", "max"]].round(2)
after  = df_clean[key_cols].describe().loc[["mean", "std", "min", "max"]].round(2)

print("\n  -- BEFORE (raw) --")
print(before.to_string())
print("\n  -- AFTER (clean) --")
print(after.to_string())

print("\n  -- COLUMNS ADDED --")
print("  + structural_shock  (1 = ปี 2009, 2020, 2021 / 0 = ปกติ)")
print("  + anomaly_flag      (1 = ปี 2013, 2014 / 0 = ปกติ)")

print("\n  -- ROWS/COLUMNS --")
print(f"  Before : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(f"  After  : {df_clean.shape[0]} rows x {df_clean.shape[1]} cols")
print(f"  Changed: +2 flag columns, 0 rows dropped")

# ── 10. Save ──────────────────────────────────────────────────────────────────
print("\n[10] SAVING")
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"  Saved: {CLEAN_PATH}")

print("\n" + "=" * 60)
print("Dana: DONE — passing to Vera (visualization) + Quinn (QC)")
print("=" * 60)
