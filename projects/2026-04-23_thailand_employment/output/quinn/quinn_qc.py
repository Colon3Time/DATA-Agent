import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).parent.parent.parent
RAW_PATH  = BASE / "input" / "thailand_employment_worldbank_2000_2024.csv"
CLEAN_PATH = BASE / "input" / "thailand_employment_clean.csv"
VERA_OUT  = BASE / "output" / "vera"

print("=" * 60)
print("QUINN — QUALITY CHECK REPORT")
print("=" * 60)

passed, failed = [], []

def check(name, result, detail=""):
    if result:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")

# ── 1. Files exist ────────────────────────────────────────────────────────────
print("\n[1] FILE EXISTENCE")
check("raw CSV exists",   RAW_PATH.exists())
check("clean CSV exists", CLEAN_PATH.exists())
for chart in ["before_after_flags.png", "sector_shift.png",
              "gdp_vs_vulnerable.png", "unemployment_total_vs_youth.png"]:
    check(f"vera/{chart}", (VERA_OUT / chart).exists())

# ── 2. Clean CSV integrity ────────────────────────────────────────────────────
print("\n[2] CLEAN CSV INTEGRITY")
df_raw   = pd.read_csv(RAW_PATH)
df_clean = pd.read_csv(CLEAN_PATH)

check("row count unchanged",     df_clean.shape[0] == df_raw.shape[0],
      f"raw={df_raw.shape[0]} clean={df_clean.shape[0]}")
check("columns increased",       df_clean.shape[1] > df_raw.shape[1],
      f"raw={df_raw.shape[1]} clean={df_clean.shape[1]}")
check("no missing values",       df_clean.isnull().sum().sum() == 0)
check("no duplicates",           df_clean.duplicated().sum() == 0)
check("year sorted ascending",   df_clean["year"].is_monotonic_increasing)
check("year range 2000-2024",    (df_clean["year"].min() == 2000) and (df_clean["year"].max() == 2024))

# ── 3. Flag columns ───────────────────────────────────────────────────────────
print("\n[3] FLAG COLUMNS")
check("structural_shock column exists", "structural_shock" in df_clean.columns)
check("anomaly_flag column exists",     "anomaly_flag" in df_clean.columns)
check("structural_shock only 0/1",     set(df_clean["structural_shock"].unique()).issubset({0, 1}))
check("anomaly_flag only 0/1",         set(df_clean["anomaly_flag"].unique()).issubset({0, 1}))
check("shock years correct (2009,2020,2021)",
      set(df_clean[df_clean["structural_shock"] == 1]["year"]) == {2009, 2020, 2021})
check("anomaly years correct (2013,2014)",
      set(df_clean[df_clean["anomaly_flag"] == 1]["year"]) == {2013, 2014})

# ── 4. Sector sum ─────────────────────────────────────────────────────────────
print("\n[4] SECTOR CONSISTENCY")
sector_sum = (df_clean["employment_agriculture_pct"]
              + df_clean["employment_industry_pct"]
              + df_clean["employment_services_pct"])
check("all sector sums within 99.5-100.5",
      bool((sector_sum >= 99.5).all() and (sector_sum <= 100.5).all()))

# ── 5. Value sanity ───────────────────────────────────────────────────────────
print("\n[5] VALUE SANITY")
check("unemployment_rate_pct in [0,10]",
      bool(df_clean["unemployment_rate_pct"].between(0, 10).all()))
check("labor_force_participation_pct in [0,100]",
      bool(df_clean["labor_force_participation_pct"].between(0, 100).all()))
check("gdp_per_capita_usd > 0",
      bool((df_clean["gdp_per_capita_usd"] > 0).all()))
check("labor_force_total > 0",
      bool((df_clean["labor_force_total"] > 0).all()))

# ── 6. Raw vs Clean value match (original cols unchanged) ─────────────────────
print("\n[6] RAW vs CLEAN — ORIGINAL VALUES UNCHANGED")
original_cols = list(df_raw.columns)
match = df_raw[original_cols].reset_index(drop=True).equals(
        df_clean[original_cols].reset_index(drop=True))
check("original columns values identical", match)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"QUINN SUMMARY: {len(passed)} passed / {len(failed)} failed")
if failed:
    print("FAILED CHECKS:")
    for f in failed:
        print(f"  - {f}")
else:
    print("ALL CHECKS PASSED — Dataset approved for downstream pipeline")
print("=" * 60)