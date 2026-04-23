"""
Dana + Vera + Quinn — Cleaning, Visualization & QC Report
Thailand Labour Market 2000-2024
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding="utf-8")

BASE       = Path(__file__).parent.parent          # projects/2026-04-23_thailand_employment/
RAW_PATH   = BASE / "input" / "thailand_employment_worldbank_2000_2024.csv"
CLEAN_PATH = BASE / "input" / "thailand_employment_clean.csv"
OUT_DANA   = BASE / "output" / "dana"
OUT_VERA   = BASE / "output" / "vera"
OUT_QUINN  = BASE / "output" / "quinn"

# ═══════════════════════════════════════════════════════════════════
#  DANA — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════
def run_dana():
    print("\n" + "═" * 60)
    print("  DANA — DATA CLEANING")
    print("═" * 60)

    df_raw = pd.read_csv(RAW_PATH)

    # ── Overview ──────────────────────────────────────────────────
    print(f"\n  Rows: {df_raw.shape[0]}  |  Cols: {df_raw.shape[1]}"
          f"  |  Year: {df_raw['year'].min()}–{df_raw['year'].max()}")

    # ── Missing & Duplicates ──────────────────────────────────────
    missing = df_raw.isnull().sum().sum()
    dupes   = df_raw.duplicated().sum()
    print(f"  Missing values : {missing}")
    print(f"  Duplicate rows : {dupes}")

    # ── Sector Consistency ────────────────────────────────────────
    df_work = df_raw.copy()
    sector_sum = (df_work["employment_agriculture_pct"]
                  + df_work["employment_industry_pct"]
                  + df_work["employment_services_pct"])
    sector_ok  = sector_sum.between(99.5, 100.5).all()
    print(f"  Sector sum check : {'OK all within 99.5-100.5%' if sector_ok else 'FAIL'}")

    # ── Outlier Detection (IQR 2.0x) ─────────────────────────────
    numeric_cols = [c for c in df_raw.columns if c != "year"]
    outliers = []
    for col in numeric_cols:
        q1, q3 = df_raw[col].quantile(0.25), df_raw[col].quantile(0.75)
        iqr = q3 - q1
        hits = df_raw[(df_raw[col] < q1 - 2*iqr) | (df_raw[col] > q3 + 2*iqr)]
        for _, row in hits.iterrows():
            outliers.append(f"    {col} @ {int(row['year'])}: {row[col]}")
    print(f"  Outliers detected: {len(outliers)}")
    for o in outliers:
        print(o)

    # ── Structural Breaks (YoY > 4pp) ────────────────────────────
    df_s = df_raw.sort_values("year").reset_index(drop=True)
    break_cols = ["employment_agriculture_pct", "employment_services_pct",
                  "vulnerable_employment_pct"]
    print("  Structural breaks (YoY > 4pp):")
    for col in break_cols:
        yoy   = df_s[col].diff()
        hits  = df_s[yoy.abs() > 4]
        for _, row in hits.iterrows():
            print(f"    {col} @ {int(row['year'])}: {yoy[row.name]:+.2f}pp")

    # ── Build Clean Dataset ───────────────────────────────────────
    df_clean = df_s.copy()
    df_clean["structural_shock"] = df_clean["year"].isin([2009, 2020, 2021]).astype(int)
    df_clean["anomaly_flag"]     = df_clean["year"].isin([2013, 2014]).astype(int)

    # ── Before / After ────────────────────────────────────────────
    key_cols = ["unemployment_rate_pct", "employment_agriculture_pct",
                "employment_services_pct", "vulnerable_employment_pct",
                "gdp_per_capita_usd"]
    stats_before = df_raw[key_cols].agg(["mean", "std", "min", "max"]).round(2)
    stats_after  = df_clean[key_cols].agg(["mean", "std", "min", "max"]).round(2)

    print("\n  BEFORE / AFTER (key columns):")
    print("  -- BEFORE --")
    print(stats_before.to_string())
    print("\n  -- AFTER --")
    print(stats_after.to_string())
    print(f"\n  Columns added : structural_shock, anomaly_flag")
    print(f"  Rows dropped  : 0  |  Final shape: {df_clean.shape}")

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\n  Saved: {CLEAN_PATH}")
    print("  Dana: DONE")
    return df_clean, numeric_cols


# ═══════════════════════════════════════════════════════════════════
#  VERA — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def run_vera(df):
    print("\n" + "═" * 60)
    print("  VERA — VISUALIZATION")
    print("═" * 60)

    shock_years   = df[df["structural_shock"] == 1]["year"]
    anomaly_years = df[df["anomaly_flag"] == 1]["year"]

    # ── Fig 1: Before/After Flags — 4 subplots ───────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Dana — Before / After Cleaning\nThailand Labour Market 2000–2024",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    plots = [
        (gs[0, 0], "unemployment_rate_pct",      "Unemployment Rate (%)",      "steelblue"),
        (gs[0, 1], "employment_agriculture_pct",  "Employment: Agriculture (%)", "forestgreen"),
        (gs[1, 0], "employment_services_pct",     "Employment: Services (%)",    "darkorange"),
        (gs[1, 1], "vulnerable_employment_pct",   "Vulnerable Employment (%)",   "crimson"),
    ]
    for g, col, title, color in plots:
        ax = fig.add_subplot(g)
        ax.plot(df["year"], df[col], color=color, linewidth=2, marker="o", markersize=3)
        for y in shock_years:
            ax.axvline(x=y, color="red",  linestyle="--", alpha=0.4, linewidth=1)
        for y in anomaly_years:
            ax.axvline(x=y, color="gold", linestyle=":",  alpha=0.8, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Year", fontsize=8)
        ax.grid(True, alpha=0.3)

    legend_els = [
        Line2D([0],[0], color="red",  linestyle="--", label="Structural Shock (2009,2020,2021)"),
        Line2D([0],[0], color="gold", linestyle=":",  label="Anomaly Flag (2013,2014)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=9)
    p1 = OUT_VERA / "01_before_after_flags.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p1.name}")

    # ── Fig 2: Sector Shift (Stacked Area) ───────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(df["year"],
                 df["employment_agriculture_pct"],
                 df["employment_industry_pct"],
                 df["employment_services_pct"],
                 labels=["Agriculture", "Industry", "Services"],
                 colors=["#4CAF50", "#FF9800", "#2196F3"], alpha=0.8)
    ax.set_title("Thailand Employment Sector Shift 2000–2024",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Total Employment")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    p2 = OUT_VERA / "02_sector_shift.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p2.name}")

    # ── Fig 3: GDP vs Vulnerable Employment ──────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(df["year"], df["gdp_per_capita_usd"],
             color="navy", linewidth=2, label="GDP per Capita (USD)")
    ax2.plot(df["year"], df["vulnerable_employment_pct"],
             color="crimson", linewidth=2, linestyle="--", label="Vulnerable Employment (%)")
    ax1.set_ylabel("GDP per Capita (USD)", color="navy")
    ax2.set_ylabel("Vulnerable Employment (%)", color="crimson")
    ax1.set_title("GDP per Capita vs Vulnerable Employment\n(Low-Unemployment, High-Vulnerability Trap)",
                  fontsize=11, fontweight="bold")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="lower left")
    ax1.grid(True, alpha=0.3)
    p3 = OUT_VERA / "03_gdp_vs_vulnerable.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p3.name}")

    # ── Fig 4: Unemployment Total vs Youth ───────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["year"], df["unemployment_rate_pct"],
            color="steelblue", linewidth=2, label="Total Unemployment (%)")
    ax.plot(df["year"], df["youth_unemployment_pct"],
            color="tomato", linewidth=2, linestyle="--", label="Youth Unemployment (%)")
    for y in [2009, 2020]:
        ax.axvspan(y - 0.4, y + 0.4, color="red", alpha=0.12)
    ax.set_title("Unemployment Rate: Total vs Youth 2000–2024",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Labor Force")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p4 = OUT_VERA / "04_unemployment_total_vs_youth.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p4.name}")

    print("  Vera: DONE")
    return [p1, p2, p3, p4]


# ═══════════════════════════════════════════════════════════════════
#  QUINN — QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════
def run_quinn(df_raw, df_clean, chart_paths):
    print("\n" + "═" * 60)
    print("  QUINN — QUALITY CHECK")
    print("═" * 60)

    passed, failed = [], []

    def check(name, result, detail=""):
        status = "PASS" if result else "FAIL"
        line   = f"  [{status}] {name}"
        if not result and detail:
            line += f"  ({detail})"
        print(line)
        (passed if result else failed).append(name)

    # File existence
    print("\n  -- Files --")
    check("raw CSV exists",   RAW_PATH.exists())
    check("clean CSV exists", CLEAN_PATH.exists())
    for p in chart_paths:
        check(f"chart: {p.name}", p.exists())

    # Shape
    print("\n  -- Shape --")
    check("row count unchanged",
          df_clean.shape[0] == df_raw.shape[0],
          f"raw={df_raw.shape[0]} clean={df_clean.shape[0]}")
    check("2 flag columns added",
          df_clean.shape[1] == df_raw.shape[1] + 2,
          f"raw={df_raw.shape[1]} clean={df_clean.shape[1]}")

    # Integrity
    print("\n  -- Integrity --")
    check("no missing values",        df_clean.isnull().sum().sum() == 0)
    check("no duplicates",            df_clean.duplicated().sum() == 0)
    check("year sorted ascending",    df_clean["year"].is_monotonic_increasing)
    check("year range 2000-2024",     df_clean["year"].min() == 2000 and df_clean["year"].max() == 2024)

    # Flags
    print("\n  -- Flag Columns --")
    check("structural_shock is 0/1",  set(df_clean["structural_shock"].unique()).issubset({0,1}))
    check("anomaly_flag is 0/1",      set(df_clean["anomaly_flag"].unique()).issubset({0,1}))
    check("shock years = {2009,2020,2021}",
          set(df_clean[df_clean["structural_shock"]==1]["year"]) == {2009,2020,2021})
    check("anomaly years = {2013,2014}",
          set(df_clean[df_clean["anomaly_flag"]==1]["year"]) == {2013,2014})

    # Sector sum
    print("\n  -- Sector Consistency --")
    s = (df_clean["employment_agriculture_pct"]
         + df_clean["employment_industry_pct"]
         + df_clean["employment_services_pct"])
    check("sector sums within 99.5-100.5", bool(s.between(99.5,100.5).all()))

    # Value sanity
    print("\n  -- Value Sanity --")
    check("unemployment_rate in [0,10]",
          bool(df_clean["unemployment_rate_pct"].between(0,10).all()))
    check("gdp_per_capita > 0",
          bool((df_clean["gdp_per_capita_usd"] > 0).all()))

    # Raw values unchanged
    print("\n  -- Raw Values Unchanged --")
    orig_cols = list(df_raw.columns)
    match = df_raw[orig_cols].reset_index(drop=True).equals(
            df_clean[orig_cols].reset_index(drop=True))
    check("original column values identical", match)

    # Summary
    print("\n" + "═" * 60)
    total = len(passed) + len(failed)
    print(f"  QUINN RESULT: {len(passed)}/{total} passed", end="  ")
    if failed:
        print(f"| FAILED: {', '.join(failed)}")
    else:
        print("| ALL CHECKS PASSED")
    print("═" * 60)
    return len(failed) == 0


# ═══════════════════════════════════════════════════════════════════
#  MAIN — ANNA ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ANNA — PIPELINE: Dana → Vera → Quinn")
    print("=" * 60)

    df_clean, numeric_cols = run_dana()
    df_raw   = pd.read_csv(RAW_PATH)
    charts   = run_vera(df_clean)
    ok       = run_quinn(df_raw, df_clean, charts)

    print("\n" + "=" * 60)
    if ok:
        print("  ANNA: Pipeline complete. Ready for Eddie (EDA).")
    else:
        print("  ANNA: Quinn found issues. Please review before proceeding.")
    print("=" * 60)
