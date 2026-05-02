import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


START = time.perf_counter()
TARGET_COLS = ["monetary", "is_churned_180d", "is_high_value", "clv_proxy"]
LEAKY_FROM_MONETARY = ["clv_proxy", "avg_order_value", "is_high_value"]


def status(msg):
    print(f"[STATUS {time.perf_counter() - START:7.1f}s] {msg}", flush=True)


def find_col(df, candidates):
    lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for c in candidates:
        key = c.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


def build_customer_table(work, invoice_col, stock_col, qty_col, date_col, price_col, customer_col, snapshot, churn_days):
    known = work[work[customer_col].ne("UNKNOWN_CUSTOMER") & work[date_col].notna()].copy()
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
    customer[f"is_churned_{churn_days}d"] = (customer["recency_days"] > churn_days).astype("int8")
    customer["grain"] = "customer"
    return customer.replace([np.inf, -np.inf], np.nan).fillna(0)


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
status(f"Valid sales={len(valid):,}")

snapshot = valid[date_col].max() + pd.Timedelta(days=1)
customer = build_customer_table(valid, invoice_col, stock_col, qty_col, date_col, price_col, customer_col, snapshot, 180)

engineered_path = output_dir / "engineered_data.csv"
finn_output_path = output_dir / "finn_output.csv"
customer.to_csv(engineered_path, index=False)
customer.to_csv(finn_output_path, index=False)
status(f"Saved customer-level engineered data: {len(customer):,} rows")

cutoff = pd.Timestamp("2011-09-30")
label_end = cutoff + pd.Timedelta(days=90)
pre_cutoff = valid[valid[date_col] <= cutoff].copy()
post_window = valid[(valid[date_col] > cutoff) & (valid[date_col] <= label_end)].copy()
oot = build_customer_table(pre_cutoff, invoice_col, stock_col, qty_col, date_col, price_col, customer_col, cutoff + pd.Timedelta(days=1), 90)
active_after = set(post_window[customer_col].dropna().astype(str))
oot["is_churned_90d"] = (~oot[customer_col].astype(str).isin(active_after)).astype("int8")
oot["oot_cutoff_date"] = cutoff.strftime("%Y-%m-%d")
oot_path = output_dir / "engineered_data_oot.csv"
oot.to_csv(oot_path, index=False)

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

manifest = {
    "targets": {
        "monetary": {
            "task": "regression",
            "exclude_features": ["clv_proxy", "avg_order_value", "is_high_value", "is_churned_180d", "recency_days", "total_quantity", "avg_unit_price"],
        },
        "is_churned_180d": {
            "task": "classification",
            "exclude_features": ["recency_days", "clv_proxy"],
        },
        "is_high_value": {
            "task": "classification",
            "exclude_features": ["monetary", "clv_proxy", "avg_order_value"],
        },
    },
    "id_cols": [customer_col, "grain"],
    "datetime_cols": ["first_purchase", "last_purchase"],
    "target_cols": TARGET_COLS,
    "leaky_from_monetary": LEAKY_FROM_MONETARY,
    "oot_split": {"cutoff_date": "2011-09-30", "label_window_days": 90, "table": "engineered_data_oot.csv"},
}
(output_dir / "finn_feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

report = f"""# Finn Feature Engineering Report - UCI Online Retail

Input: {input_path}

## Outputs

- `engineered_data.csv`: full-history customer table, {len(customer):,} rows
- `engineered_data_oot.csv`: time-cutoff table at 2011-09-30 with 90-day churn label, {len(oot):,} rows
- `finn_feature_manifest.json`: target-specific feature exclusions for Mo
- `invoice_basket_features.csv`: invoice-level basket table, {len(invoice):,} rows
- `product_month_features.csv`: product-month inventory table, {len(product_month):,} rows

## Leakage Controls

- `engineered_data.csv` keeps targets for reference, but Mo must follow `finn_feature_manifest.json`.
- Monetary-derived columns are explicitly listed: {", ".join(LEAKY_FROM_MONETARY)}.
- OOT validation table uses features before 2011-09-30 and labels activity in the next 90 days.

FEATURE_GOVERNANCE
==================
selected_feature_count: {len(customer.columns)}
primary_grain: customer
target_columns: {", ".join(TARGET_COLS)}
train_only_transforms: enforced downstream by Mo manifest
"""
(output_dir / "finn_report.md").write_text(report, encoding="utf-8")

agent_report = f"""Agent Report - Finn
============================
Input  : {input_path}
Output : {engineered_path}
Done   : built customer, OOT, invoice, and product-month feature tables
Rows   : customer={len(customer):,}, oot={len(oot):,}, invoice={len(invoice):,}, product_month={len(product_month):,}
Next   : Iris for RFM/basket; Mo for manifest-controlled models
"""
(output_dir.parent / "agent_report_finn.md").write_text(agent_report, encoding="utf-8")
status("Finn feature engineering complete")
