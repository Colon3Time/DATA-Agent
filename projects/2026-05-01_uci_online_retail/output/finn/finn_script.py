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
    for c in candidates:
        key = c.lower().replace(" ", "").replace("_", "")
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
status(f"Loaded {len(df):,} transaction rows")

invoice_col = find_col(df, ["Invoice", "InvoiceNo", "Invoice ID"])
stock_col = find_col(df, ["StockCode", "Stock Code", "SKU"])
qty_col = find_col(df, ["Quantity", "Qty"])
date_col = find_col(df, ["InvoiceDate", "Invoice Date", "Date"])
price_col = find_col(df, ["Price", "UnitPrice", "Unit Price"])
customer_col = find_col(df, ["Customer ID", "CustomerID", "Customer Id", "customer_id"])
country_col = find_col(df, ["Country"])
desc_col = find_col(df, ["Description"])

for name, col in {
    "invoice": invoice_col,
    "stock": stock_col,
    "quantity": qty_col,
    "date": date_col,
    "price": price_col,
    "customer": customer_col,
}.items():
    if not col:
        raise SystemExit(f"[ERROR] Missing required {name} column")

df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[customer_col] = df[customer_col].astype("string").fillna("UNKNOWN_CUSTOMER")

if "revenue" not in df.columns:
    df["revenue"] = df[qty_col] * df[price_col]
if "is_return" not in df.columns:
    df["is_return"] = ((df[qty_col] < 0) | df[invoice_col].astype(str).str.startswith("C")).astype("int8")
if "is_valid_sale" not in df.columns:
    df["is_valid_sale"] = ((df[qty_col] > 0) & (df[price_col] > 0)).astype("int8")

valid = df[df["is_valid_sale"].eq(1) & df[date_col].notna()].copy()
known = valid[df.loc[valid.index, customer_col].ne("UNKNOWN_CUSTOMER")].copy()
status(f"Valid sales={len(valid):,}; known-customer sales={len(known):,}")

snapshot = known[date_col].max() + pd.Timedelta(days=1)
customer = (
    known.groupby(customer_col)
    .agg(
        recency_days=(date_col, lambda s: int((snapshot - s.max()).days)),
        frequency=(invoice_col, "nunique"),
        monetary=("revenue", "sum"),
        total_quantity=(qty_col, "sum"),
        avg_unit_price=(price_col, "mean"),
        first_purchase=(date_col, "min"),
        last_purchase=(date_col, "max"),
        distinct_products=(stock_col, "nunique"),
        return_rows=("is_return", "sum"),
    )
    .reset_index()
)
customer["tenure_days"] = (customer["last_purchase"] - customer["first_purchase"]).dt.days.clip(lower=0)
customer["avg_order_value"] = customer["monetary"] / customer["frequency"].replace(0, np.nan)
customer["purchase_freq_per_30d"] = customer["frequency"] / (customer["tenure_days"] / 30 + 1)
customer["clv_proxy"] = customer["monetary"]
customer["is_high_value"] = (customer["monetary"] >= customer["monetary"].quantile(0.80)).astype("int8")
customer["is_churned_180d"] = (customer["recency_days"] > 180).astype("int8")
customer["grain"] = "customer"
customer = customer.replace([np.inf, -np.inf], np.nan).fillna(0)

engineered_path = output_dir / "engineered_data.csv"
finn_output_path = output_dir / "finn_output.csv"
customer.to_csv(engineered_path, index=False)
customer.to_csv(finn_output_path, index=False)
status(f"Saved customer-level engineered data: {len(customer):,} rows")

invoice = (
    valid.groupby(invoice_col)
    .agg(
        basket_value=("revenue", "sum"),
        basket_quantity=(qty_col, "sum"),
        unique_items=(stock_col, "nunique"),
        line_items=(stock_col, "count"),
        invoice_date=(date_col, "min"),
        customer_id=(customer_col, "first"),
        country=(country_col, "first") if country_col else (customer_col, "first"),
    )
    .reset_index()
)
invoice["avg_price_per_line"] = invoice["basket_value"] / invoice["line_items"].replace(0, np.nan)
invoice["item_diversity"] = invoice["unique_items"] / invoice["line_items"].replace(0, np.nan)
invoice["grain"] = "invoice"
invoice.to_csv(output_dir / "invoice_basket_features.csv", index=False)

valid["product_month"] = valid[date_col].dt.to_period("M").astype(str)
product_month = (
    valid.groupby([stock_col, "product_month"])
    .agg(
        monthly_quantity=(qty_col, "sum"),
        monthly_revenue=("revenue", "sum"),
        monthly_invoices=(invoice_col, "nunique"),
        monthly_customers=(customer_col, "nunique"),
        avg_price=(price_col, "mean"),
    )
    .reset_index()
)
product_month["grain"] = "product_month"
product_month.to_csv(output_dir / "product_month_features.csv", index=False)

if desc_col:
    top_desc = valid.groupby(stock_col)[desc_col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    top_desc.to_csv(output_dir / "product_descriptions.csv")

report = f"""# Finn Feature Engineering Report - UCI Online Retail

Input: {input_path}

## Outputs

- `engineered_data.csv`: customer-level model table, {len(customer):,} rows
- `finn_output.csv`: same customer-level table for pipeline compatibility
- `invoice_basket_features.csv`: invoice-level basket table, {len(invoice):,} rows
- `product_month_features.csv`: product-month inventory table, {len(product_month):,} rows

## Grain Contract

- Customer grain: RFM, CLV proxy, churn target candidates
- Invoice grain: market basket readiness
- Product-month grain: inventory/demand optimization
- Transaction grain is not used for churn/CLV modeling

## Target Columns Created

- `is_churned_180d`: 1 if recency > 180 days at snapshot {snapshot.date()}
- `clv_proxy`: historical customer monetary value
- `is_high_value`: top 20% monetary customers

## Leakage Notes

These are analytical targets from the full historical dataset. For production modeling, Mo must use a time cutoff and rebuild features only from transactions before the cutoff. This file is acceptable for baseline experimentation and pipeline validation, not final deployment claims.

## Feature Governance

- UNKNOWN_CUSTOMER excluded from customer-level table
- Returns/cancellations excluded from valid sales features
- InvoiceDate parsed as real calendar dates, not Excel serial nanoseconds
- No row-level transaction table is handed to Mo for churn/CLV

## Suggested Next Steps

- Iris: run RFM segmentation from `engineered_data.csv` and basket analysis from `invoice_basket_features.csv`
- Mo: train baseline churn/high-value classifiers on `engineered_data.csv`, with leakage caveat clearly reported

FEATURE_GOVERNANCE
==================
selected_feature_count: {len(customer.columns)}
primary_grain: customer
target_columns: is_churned_180d, is_high_value, clv_proxy
train_only_transforms: required in Mo for production-grade validation
"""
(output_dir / "finn_report.md").write_text(report, encoding="utf-8")

agent_report = f"""Agent Report - Finn
============================
Input  : {input_path}
Output : {engineered_path}
Done   : built customer, invoice, and product-month feature tables
Rows   : customer={len(customer):,}, invoice={len(invoice):,}, product_month={len(product_month):,}
Next   : Iris for RFM/basket; Mo for baseline customer-level churn/high-value/CLV models
"""
(output_dir.parent / "agent_report_finn.md").write_text(agent_report, encoding="utf-8")
status("Finn feature engineering complete")
