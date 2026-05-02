import argparse
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

project_output = Path(args.output_dir).parent
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

customer_path = Path(args.input) if args.input else project_output / "finn" / "engineered_data.csv"
invoice_path = project_output / "finn" / "invoice_basket_features.csv"

customers = pd.read_csv(customer_path)
invoices = pd.read_csv(invoice_path) if invoice_path.exists() else pd.DataFrame()

rfm = customers.copy()
rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

def segment(score):
    if score >= 13:
        return "Champions"
    if score >= 10:
        return "Loyal"
    if score >= 7:
        return "Potential"
    if score >= 5:
        return "At Risk"
    return "Lost"

rfm["segment"] = rfm["rfm_score"].map(segment)
rfm.to_csv(output_dir / "rfm_segments.csv", index=False)

segment_summary = (
    rfm.groupby("segment")
    .agg(
        customers=("segment", "size"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        revenue=("monetary", "sum"),
    )
    .sort_values("revenue", ascending=False)
)
segment_summary.to_csv(output_dir / "rfm_segment_summary.csv")

basket_summary = pd.DataFrame()
if not invoices.empty:
    basket_summary = invoices[["basket_value", "basket_quantity", "unique_items", "item_diversity"]].describe().T
    basket_summary.to_csv(output_dir / "basket_summary.csv")

report = f"""# Iris Segmentation and Basket Report

Input customer table: {customer_path}

## RFM Segmentation

Customers segmented: {len(rfm):,}

{segment_summary.to_markdown()}

## Basket Readiness

Invoice basket table: {invoice_path}
Invoices available: {len(invoices):,}

Market basket analysis is ready at invoice grain. For association rules, use valid sales line items from Eddie/Finn and exclude returns.

## Business Use

- Champions/Loyal: retention and premium bundles
- Potential: cross-sell and replenishment campaigns
- At Risk/Lost: win-back offers with cost cap

IRIS_DECISION_FRAME
===================
rfm_output: rfm_segments.csv
basket_output: basket_summary.csv
next_agent: Vera for charts, Rex for final executive story
"""
(output_dir / "iris_report.md").write_text(report, encoding="utf-8")
(output_dir.parent / "agent_report_iris.md").write_text("Agent Report - Iris\nOutput: rfm_segments.csv, iris_report.md\n", encoding="utf-8")
print("[STATUS] Iris complete")
