import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(__file__).parent.parent.parent
df     = pd.read_csv(BASE / "input" / "thailand_employment_clean.csv")
OUT    = Path(__file__).parent

print("Vera — generating visualizations...")

# ── Chart 1: Before/After — Raw vs Clean flag overlay ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Dana — Before / After Cleaning\nThailand Labour Market 2000–2024", fontsize=14, fontweight="bold")

shock_years   = df[df["structural_shock"] == 1]["year"]
anomaly_years = df[df["anomaly_flag"] == 1]["year"]

plots = [
    ("unemployment_rate_pct",       "Unemployment Rate (%)",         "steelblue"),
    ("employment_agriculture_pct",  "Employment: Agriculture (%)",   "forestgreen"),
    ("employment_services_pct",     "Employment: Services (%)",      "darkorange"),
    ("vulnerable_employment_pct",   "Vulnerable Employment (%)",     "crimson"),
]

for ax, (col, title, color) in zip(axes.flat, plots):
    ax.plot(df["year"], df[col], color=color, linewidth=2, marker="o", markersize=3)
    for y in shock_years:
        ax.axvline(x=y, color="red", linestyle="--", alpha=0.4, linewidth=1)
    for y in anomaly_years:
        ax.axvline(x=y, color="gold", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)

# legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color="red",  linestyle="--", label="Structural Shock (2009,2020,2021)"),
    Line2D([0], [0], color="gold", linestyle=":",  label="Anomaly Flag (2013,2014)"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=9)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / "before_after_flags.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: before_after_flags.png")

# ── Chart 2: Sector Shift (Stacked Area) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.stackplot(df["year"],
             df["employment_agriculture_pct"],
             df["employment_industry_pct"],
             df["employment_services_pct"],
             labels=["Agriculture", "Industry", "Services"],
             colors=["#4CAF50", "#FF9800", "#2196F3"], alpha=0.8)
ax.set_title("Thailand Employment Sector Shift 2000–2024", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("% of Total Employment")
ax.legend(loc="upper left")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "sector_shift.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: sector_shift.png")

# ── Chart 3: GDP vs Vulnerable Employment ────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(df["year"], df["gdp_per_capita_usd"],   color="navy",  linewidth=2, label="GDP per Capita (USD)")
ax2.plot(df["year"], df["vulnerable_employment_pct"], color="crimson", linewidth=2, linestyle="--", label="Vulnerable Employment (%)")
ax1.set_xlabel("Year")
ax1.set_ylabel("GDP per Capita (USD)", color="navy")
ax2.set_ylabel("Vulnerable Employment (%)", color="crimson")
ax1.set_title("GDP per Capita vs Vulnerable Employment — Low-Unemployment Trap", fontsize=12, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "gdp_vs_vulnerable.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gdp_vs_vulnerable.png")

# ── Chart 4: Unemployment — Total vs Youth ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["year"], df["unemployment_rate_pct"],  color="steelblue", linewidth=2, label="Total Unemployment (%)")
ax.plot(df["year"], df["youth_unemployment_pct"], color="tomato",    linewidth=2, linestyle="--", label="Youth Unemployment (%)")
for y in [2009, 2020]:
    ax.axvspan(y - 0.3, y + 0.3, color="red", alpha=0.15)
ax.set_title("Unemployment Rate: Total vs Youth 2000–2024", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("% of Labor Force")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "unemployment_total_vs_youth.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: unemployment_total_vs_youth.png")

print("\nVera: ALL CHARTS DONE — passing to Quinn (QC)")
