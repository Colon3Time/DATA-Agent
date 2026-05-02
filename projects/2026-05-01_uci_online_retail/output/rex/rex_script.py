from pathlib import Path

import pandas as pd


base = Path(__file__).resolve().parents[2] / "output"
out = base / "rex"
out.mkdir(parents=True, exist_ok=True)

country = pd.read_csv(base / "eddie" / "country_sales_summary.csv")
monthly = pd.read_csv(base / "eddie" / "monthly_sales_summary.csv")
rfm = pd.read_csv(base / "iris" / "rfm_segment_summary.csv")
models = pd.read_csv(base / "mo" / "model_comparison.csv")
finn = pd.read_csv(base / "finn" / "engineered_data.csv")

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
(base / "agent_report_rex.md").write_text("Agent Report - Rex\nOutput: executive_summary.md, rex_report.md\n", encoding="utf-8")
print("[STATUS] Rex complete")
