import argparse
import os
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base = OUTPUT_DIR.parent.parent  # projects/2026-05-01_uci_online_retail
out = OUTPUT_DIR

country = pd.read_csv(base / "output" / "eddie" / "country_sales_summary.csv")
monthly = pd.read_csv(base / "output" / "eddie" / "monthly_sales_summary.csv")
rfm = pd.read_csv(base / "output" / "iris" / "rfm_segment_summary.csv")
models = pd.read_csv(base / "output" / "mo" / "model_comparison.csv")
finn = pd.read_csv(base / "output" / "finn" / "engineered_data.csv")

total_revenue = country["revenue"].sum()
top_country = country.iloc[0, 0]
top_country_share = country.iloc[0]["revenue_share_pct"]
peak_month = monthly.sort_values("revenue", ascending=False).iloc[0]["invoice_month"]
champions = rfm[rfm["segment"].astype(str).eq("Champions")]["customers"].sum() if "Champions" in set(rfm["segment"].astype(str)) else 0
churn_rate = finn["is_churned_180d"].mean() * 100 if "is_churned_180d" in finn.columns else float("nan")

report = f"""# Executive Report - UCI Online Retail Analytics

## Executive Summary

The project now has a full working analytics pipeline from Scout through Rex using the UCI Online Retail transaction data. The key correction was enforcing grain-specific analytics: transactions for descriptive sales, customers for RFM/churn/CLV, invoices for basket analysis, and product-month for inventory.

## Key Findings

- Total valid-sales revenue: {total_revenue:,.2f}
- Top revenue country: {top_country} ({top_country_share:.1f}% share)
- Peak revenue month: {peak_month}
- Customer-level model table: {len(finn):,} known customers
- Champions segment size: {int(champions):,} customers
- Baseline churn label rate: {churn_rate:.1f}%

## Model Results

{models.to_markdown(index=False)}

## Business Recommendations

1. Prioritize UK retention campaigns because revenue concentration is high.
2. Use RFM segments for marketing actions: Champions/Loyal for retention, Potential for cross-sell, At Risk/Lost for win-back.
3. Use product-month demand tables for inventory planning rather than transaction rows.
4. Treat current Mo results as baseline validation only; final churn/CLV claims require time-cutoff labels.

## Deliverables

- Dana clean table: `output/dana/dana_output.csv`
- Eddie EDA: `output/eddie/eddie_report.md`
- Finn features: `output/finn/engineered_data.csv`
- Iris segmentation: `output/iris/rfm_segments.csv`
- Mo model results: `output/mo/model_results.md`
- Vera charts: `output/vera/charts/`

FINAL_STATUS
============
status: complete_with_modeling_caveat
main_caveat: production-grade churn/CLV requires time-cutoff validation
"""
(out / "executive_summary.md").write_text(report, encoding="utf-8")
(out / "rex_report.md").write_text(report, encoding="utf-8")
(base / "output" / "agent_report_rex.md").write_text("Agent Report - Rex\nOutput: executive_summary.md, rex_report.md\n", encoding="utf-8")

# Save CSV to satisfy orchestrator requirement
summary_df = pd.DataFrame({
    "metric": ["total_revenue", "top_country", "top_country_share_pct", "peak_month", "champions_customers", "churn_rate_pct"],
    "value": [total_revenue, top_country, top_country_share, peak_month, int(champions), churn_rate]
})
summary_df.to_csv(OUTPUT_DIR / "output.csv", index=False)

print("[STATUS] Rex complete")