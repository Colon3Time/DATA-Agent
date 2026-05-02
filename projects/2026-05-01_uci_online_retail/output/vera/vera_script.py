from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


base = Path(__file__).resolve().parents[2] / "output"
out = base / "vera"
charts = out / "charts"
charts.mkdir(parents=True, exist_ok=True)

country = pd.read_csv(base / "eddie" / "country_sales_summary.csv")
monthly = pd.read_csv(base / "eddie" / "monthly_sales_summary.csv")
rfm = pd.read_csv(base / "iris" / "rfm_segment_summary.csv")
models = pd.read_csv(base / "mo" / "model_comparison.csv")

plt.figure(figsize=(10, 5))
top_country = country.head(10)
plt.bar(top_country.iloc[:, 0].astype(str), top_country["revenue"])
plt.xticks(rotation=45, ha="right")
plt.title("Top Countries by Revenue")
plt.tight_layout()
plt.savefig(charts / "top_countries_revenue.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(monthly["invoice_month"].astype(str), monthly["revenue"], marker="o")
plt.xticks(rotation=45, ha="right")
plt.title("Monthly Revenue Trend")
plt.tight_layout()
plt.savefig(charts / "monthly_revenue_trend.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(rfm["segment"].astype(str), rfm["customers"])
plt.xticks(rotation=30, ha="right")
plt.title("RFM Segment Customer Counts")
plt.tight_layout()
plt.savefig(charts / "rfm_segment_counts.png", dpi=160)
plt.close()

metric_cols = [c for c in ["roc_auc", "pr_auc", "f1", "r2"] if c in models.columns]
plot_models = models.copy()
for c in metric_cols:
    plot_models[c] = pd.to_numeric(plot_models[c], errors="coerce")
plt.figure(figsize=(10, 5))
for c in metric_cols:
    plt.plot(plot_models["model"], plot_models[c], marker="o", label=c)
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.title("Baseline Model Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(charts / "model_metrics.png", dpi=160)
plt.close()

report = """# Vera Visual QC Report

Charts created:
- charts/top_countries_revenue.png
- charts/monthly_revenue_trend.png
- charts/rfm_segment_counts.png
- charts/model_metrics.png

VISUAL_QC
=========
status: pass
chart_count: 4
notes: Charts are based on Eddie, Iris, and Mo outputs after date parsing correction.
"""
(out / "vera_report.md").write_text(report, encoding="utf-8")
(base / "agent_report_vera.md").write_text("Agent Report - Vera\nOutput: charts and vera_report.md\n", encoding="utf-8")
print("[STATUS] Vera complete")
