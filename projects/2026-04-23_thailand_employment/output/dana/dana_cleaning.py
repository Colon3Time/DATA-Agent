import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="", help="path to raw CSV")
parser.add_argument("--output-dir", default="", help="dir to save outputs")
args = parser.parse_args()

# fallback: ถ้าไม่ได้รับ arg ใช้ path เดิม
if args.input:
    RAW_PATH = Path(args.input)
else:
    BASE = Path(__file__).parent.parent.parent
    RAW_PATH = BASE / "input" / "thailand_employment_worldbank_2000_2024.csv"

if args.output_dir:
    OUT_DIR = Path(args.output_dir)
else:
    OUT_DIR = Path(__file__).parent

OUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_PATH = OUT_DIR / "dana_clean.csv"
REPORT_PATH = OUT_DIR / "dana_cleaning_report.md"

# ── Load raw ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(RAW_PATH)

print("=" * 60)
print("DANA — DATA CLEANING REPORT")
print("=" * 60)

# ── 1. Overview ───────────────────────────────────────────────────────────────
print("\n[1] DATASET OVERVIEW")
print(f"  Input   : {RAW_PATH}")
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
numeric_cols = [c for c in df_raw.columns if c != "year"]
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
print("  + structural_shock  (1 = ปี 2009, 2020, 2021)")
print("  + anomaly_flag      (1 = ปี 2013, 2014)")
print(f"\n  Before : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(f"  After  : {df_clean.shape[0]} rows x {df_clean.shape[1]} cols")

# ── 10. Save ──────────────────────────────────────────────────────────────────
print("\n[10] SAVING")
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"  CSV    : {CLEAN_PATH}")
print(f"  Report : {REPORT_PATH}")

# บันทึก report
report_lines = [
    "# Dana — Cleaning Report",
    f"Date: {__import__('datetime').date.today()}",
    f"Input : {RAW_PATH}",
    f"Output: {CLEAN_PATH}",
    "",
    f"- Rows: {df_clean.shape[0]}",
    f"- Columns: {df_clean.shape[1]} (+2 flags)",
    f"- Missing values: {missing.sum()}",
    f"- Duplicates: {df_raw.duplicated().sum()}",
    f"- Outliers flagged: {len(outlier_log)}",
    f"- Structural breaks (>4pp YoY): see log above",
    "",
    "## Output path",
    str(CLEAN_PATH),
]
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print("\n" + "=" * 60)
print("Dana: DONE")
print(f"OUTPUT_CSV={CLEAN_PATH}")  # orchestrator จับ line นี้
print("=" * 60)
