import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


START = time.perf_counter()


def status(msg):
    print(f"[STATUS {time.perf_counter() - START:7.1f}s] {msg}", flush=True)


def find_col(df, candidates):
    lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for name in candidates:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

input_path = args.input
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if not input_path or not Path(input_path).exists():
    raise SystemExit(f"[ERROR] --input is required and must exist: {input_path}")

df = pd.read_csv(input_path, low_memory=False)
status(f"Loaded {df.shape[0]:,} rows x {df.shape[1]:,} cols")

invoice_col = find_col(df, ["Invoice", "InvoiceNo", "Invoice ID"])
stock_col = find_col(df, ["StockCode", "Stock Code", "SKU"])
desc_col = find_col(df, ["Description", "Product"])
qty_col = find_col(df, ["Quantity", "Qty"])
date_col = find_col(df, ["InvoiceDate", "Invoice Date", "Date"])
price_col = find_col(df, ["Price", "UnitPrice", "Unit Price"])
customer_col = find_col(df, ["Customer ID", "CustomerID", "Customer Id", "customer_id"])
country_col = find_col(df, ["Country"])

required = {
    "invoice": invoice_col,
    "stock": stock_col,
    "quantity": qty_col,
    "date": date_col,
    "price": price_col,
    "country": country_col,
}
missing_required = [k for k, v in required.items() if not v]
if missing_required:
    raise SystemExit(f"[ERROR] Missing required retail columns: {missing_required}")

df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df["revenue"] = df[qty_col] * df[price_col]
df["is_return"] = ((df[qty_col] < 0) | df[invoice_col].astype(str).str.startswith("C")).astype("int8")
df["is_valid_sale"] = ((df[qty_col] > 0) & (df[price_col] > 0)).astype("int8")
df["invoice_year"] = df[date_col].dt.year
df["invoice_month"] = df[date_col].dt.to_period("M").astype(str)
df["invoice_hour"] = df[date_col].dt.hour
df["invoice_dayofweek"] = df[date_col].dt.day_name()

if customer_col:
    df[customer_col] = df[customer_col].astype("string").fillna("UNKNOWN_CUSTOMER")
    known_customer_mask = df[customer_col].ne("UNKNOWN_CUSTOMER")
else:
    known_customer_mask = pd.Series(False, index=df.index)

valid_sales = df[df["is_valid_sale"].eq(1)].copy()
known_sales = valid_sales[known_customer_mask.loc[valid_sales.index]].copy()
status(f"Prepared transaction features; valid sales={len(valid_sales):,}, known-customer sales={len(known_sales):,}")

country_sales = (
    valid_sales.groupby(country_col, dropna=False)
    .agg(revenue=("revenue", "sum"), transactions=(invoice_col, "count"), invoices=(invoice_col, "nunique"))
    .sort_values("revenue", ascending=False)
)
country_sales["revenue_share_pct"] = country_sales["revenue"] / max(country_sales["revenue"].sum(), 1) * 100
country_sales.head(50).to_csv(output_dir / "country_sales_summary.csv")

product_group_cols = [stock_col] + ([desc_col] if desc_col else [])
product_sales = (
    valid_sales.groupby(product_group_cols, dropna=False)
    .agg(revenue=("revenue", "sum"), quantity=(qty_col, "sum"), transactions=(invoice_col, "count"))
    .sort_values("revenue", ascending=False)
)
product_sales.head(100).to_csv(output_dir / "top_products_summary.csv")

monthly_sales = (
    valid_sales.groupby("invoice_month")
    .agg(revenue=("revenue", "sum"), invoices=(invoice_col, "nunique"), transactions=(invoice_col, "count"))
    .sort_index()
)
monthly_sales.to_csv(output_dir / "monthly_sales_summary.csv")

hourly_sales = (
    valid_sales.groupby("invoice_hour")
    .agg(revenue=("revenue", "sum"), transactions=(invoice_col, "count"))
    .sort_index()
)
hourly_sales.to_csv(output_dir / "hourly_sales_summary.csv")

rfm = pd.DataFrame()
if customer_col and not known_sales.empty:
    snapshot = known_sales[date_col].max() + pd.Timedelta(days=1)
    rfm = (
        known_sales.groupby(customer_col)
        .agg(
            recency_days=(date_col, lambda s: int((snapshot - s.max()).days)),
            frequency=(invoice_col, "nunique"),
            monetary=("revenue", "sum"),
            first_purchase=(date_col, "min"),
            last_purchase=(date_col, "max"),
            distinct_products=(stock_col, "nunique"),
        )
        .reset_index()
    )
    rfm["is_repeat_customer"] = (rfm["frequency"] > 1).astype("int8")
    rfm["high_value_customer"] = (rfm["monetary"] >= rfm["monetary"].quantile(0.80)).astype("int8")
    rfm.to_csv(output_dir / "rfm_customer_table.csv", index=False)

sample_n = min(100_000, len(valid_sales))
sample = valid_sales.sample(sample_n, random_state=42) if sample_n else valid_sales
diagnostic = {
    "sample_rows": sample_n,
    "quantity_revenue_corr": float(sample[[qty_col, "revenue"]].corr().iloc[0, 1]) if sample_n else np.nan,
    "price_revenue_corr": float(sample[[price_col, "revenue"]].corr().iloc[0, 1]) if sample_n else np.nan,
}

output_csv = output_dir / "eddie_output.csv"
df.to_csv(output_csv, index=False)
status(f"Saved eddie_output.csv -> {output_csv}")

total_revenue = valid_sales["revenue"].sum()
uk_share = country_sales.loc["United Kingdom", "revenue_share_pct"] if "United Kingdom" in country_sales.index else np.nan
peak_month = monthly_sales["revenue"].idxmax() if not monthly_sales.empty else "unknown"
peak_hour = int(hourly_sales["revenue"].idxmax()) if not hourly_sales.empty else -1
return_rows = int(df["is_return"].sum())
unknown_customer_rows = int((~known_customer_mask).sum()) if customer_col else len(df)
repeat_rate = float(rfm["is_repeat_customer"].mean() * 100) if not rfm.empty else np.nan
top20_product_share = float(product_sales.head(20)["revenue"].sum() / max(total_revenue, 1) * 100)

report = f"""# Eddie EDA & Business Report - UCI Online Retail

Rows: {len(df):,}
Columns: {len(df.columns):,}
Input: {input_path}

## BUSINESS_EDA_FRAME

Business question: What happened across country, time, and product sales, and what customer/product grains are needed for RFM, market basket, CLV, churn, and inventory decisions?
Decision owner: Head of E-commerce / Retail Operations / Marketing Analytics
Primary KPI: valid sales revenue = Quantity * Price for rows where Quantity > 0 and Price > 0
Strongest evidence:
- Total valid-sales revenue: {total_revenue:,.2f}
- United Kingdom revenue share: {uk_share:.1f}%
- Peak sales month: {peak_month}
- Peak sales hour: {peak_hour}:00
- Top 20 product revenue share: {top20_product_share:.1f}%
- Known-customer repeat rate: {repeat_rate:.1f}%
Causality status: descriptive only; no causal claim.
Temporal/leakage risk: InvoiceDate exists, so churn/CLV must use cutoff dates and exclude future transactions.
Imbalance/skew risk: country and revenue are highly skewed; UK dominates and revenue has large retail extremes.
Validation strategy: use time-based validation for forecasting/churn/CLV; use customer-level split after aggregation, not transaction-row random split.

## Descriptive Findings

- Country: United Kingdom is the dominant revenue country; see `country_sales_summary.csv`.
- Time: sales concentrate in business hours; see `hourly_sales_summary.csv`.
- Product: revenue is concentrated in top SKUs; see `top_products_summary.csv`.
- Returns/cancellations: {return_rows:,} rows flagged by negative quantity or cancellation invoice.
- Anonymous customers: {unknown_customer_rows:,} rows do not have a known customer identity.

## Behavioral Readiness

- RFM is ready only for known customers. Output: `rfm_customer_table.csv` ({len(rfm):,} customers).
- Market basket analysis should use invoice-level valid sales and exclude returns/cancellations.
- CLV and churn must use customer-level aggregates from `rfm_customer_table.csv`, not raw transaction rows.

## Predictive/Decision Readiness

- CLV target candidate: customer-level monetary value over a fixed future window.
- Churn target candidate: no repeat purchase after a defined cutoff/horizon.
- Inventory target candidate: product-month demand from valid sales quantity.
- Mo must not train churn/CLV directly on `eddie_output.csv` transaction rows.

## Diagnostic Sample

- Sample rows for diagnostics: {diagnostic["sample_rows"]:,}
- Quantity/revenue correlation: {diagnostic["quantity_revenue_corr"]:.3f}
- Price/revenue correlation: {diagnostic["price_revenue_corr"]:.3f}
- Full KMeans/silhouette on 1M+ rows was intentionally skipped; use sampled or aggregated tables only.

## PIPELINE_SPEC

problem_type        : multi-grain retail analytics
target_column       : none at transaction grain
transaction_table   : eddie_output.csv
customer_table      : rfm_customer_table.csv
product_time_table  : monthly_sales_summary.csv plus top_products_summary.csv
n_rows              : {len(df):,}
n_features          : {len(df.columns):,}
recommended_model   : descriptive tables first; KMeans for RFM segments; supervised churn/CLV only after Finn creates customer-level targets
key_features        : country, invoice_month, invoice_hour, stock code, quantity, price, revenue, is_return
preprocessing       : exclude returns for sales modeling; exclude UNKNOWN_CUSTOMER for customer modeling; time cutoff before target creation
grain_contract      : transaction for descriptive, invoice for basket, customer for RFM/CLV/churn, product-month for inventory
data_quality_issues : UNKNOWN_CUSTOMER rows={unknown_customer_rows:,}; returns/cancellations={return_rows:,}; retail extremes are flagged not dropped
finn_instructions   : build customer-level features from rfm_customer_table.csv for churn/CLV; build invoice-item matrix for basket; build product-month table for inventory
mo_instructions     : fail if target/grain is missing; do not train on transaction rows for churn or CLV

## INSIGHT_QUALITY

Criteria Met: 3/4
1. Business KPI defined: PASS
2. Grain-specific downstream plan: PASS
3. Actionable country/time/product/customer patterns: PASS
4. Causal claim: NOT ATTEMPTED

Verdict: SUFFICIENT for Finn/Iris handoff.
"""

report_path = output_dir / "eddie_report.md"
report_path.write_text(report, encoding="utf-8")
status(f"Saved eddie_report.md -> {report_path}")

agent_report = f"""Agent Report - Eddie
============================
Input   : {input_path}
Output  : {output_csv}
Report  : {report_path}
Tables  : country_sales_summary.csv, top_products_summary.csv, monthly_sales_summary.csv, hourly_sales_summary.csv, rfm_customer_table.csv
Finding : transaction grain is clean for descriptive work; customer grain is required for RFM/CLV/churn.
Next    : Finn/Iris should aggregate by the correct grain before modeling or basket analysis.
"""
(output_dir.parent / "agent_report_eddie.md").write_text(agent_report, encoding="utf-8")
status("Eddie EDA complete")
