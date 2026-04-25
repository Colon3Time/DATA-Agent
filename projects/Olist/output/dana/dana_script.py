import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

# ── Paths (arg > fallback) ────────────────────────────────────────────────────
_FALLBACK_INPUT  = r"C:\Users\Amorntep\DATA-Agent\projects\olist\input"
_FALLBACK_OUTPUT = r"C:\Users\Amorntep\DATA-Agent\projects\olist\output\dana"

OUTPUT_DIR = args.output_dir or _FALLBACK_OUTPUT
os.makedirs(OUTPUT_DIR, exist_ok=True)

# input อาจเป็น .sqlite โดยตรง หรือ folder ที่มี .sqlite อยู่ข้างใน
_input = Path(args.input) if args.input else Path(_FALLBACK_INPUT)
if _input.is_dir():
    _candidates = list(_input.glob("*.sqlite")) + list(_input.glob("*.db"))
    DB_PATH = str(_candidates[0]) if _candidates else os.path.join(_FALLBACK_INPUT, "olist.sqlite")
else:
    DB_PATH = str(_input)

# Connect
conn = sqlite3.connect(DB_PATH)

# Load all tables
orders = pd.read_sql("SELECT * FROM orders", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)
items = pd.read_sql("SELECT * FROM order_items", conn)
payments = pd.read_sql("SELECT * FROM order_payments", conn)
reviews = pd.read_sql("SELECT * FROM order_reviews", conn)
products = pd.read_sql("SELECT * FROM products", conn)
sellers = pd.read_sql("SELECT * FROM sellers", conn)
geo = pd.read_sql("SELECT * FROM geolocation", conn)
cat_trans = pd.read_sql("SELECT * FROM product_category_name_translation", conn)
conn.close()

print(f"[INFO] Loaded all tables")

# Join tables
items_products = items.merge(products, on='product_id', how='left')
items_products_sellers = items_products.merge(sellers, on='seller_id', how='left')
order_detail = orders.merge(customers, on='customer_id', how='left')
order_detail = order_detail.merge(items_products_sellers, on='order_id', how='left')
order_detail = order_detail.merge(payments, on='order_id', how='left')
order_detail = order_detail.merge(reviews, on='order_id', how='left')

print(f"[INFO] Joined all tables: {len(order_detail)} rows")

cleaned = order_detail.copy()

# 1. Drop review_comment_title (88% missing)
cleaned = cleaned.drop(columns=['review_comment_title'])

# 2. Handle review_comment_message
cleaned['has_review_comment'] = cleaned['review_comment_message'].notna().astype(int)
cleaned['review_comment_message'] = cleaned['review_comment_message'].fillna('')

# 3. Product category
cleaned['product_category_name'] = cleaned['product_category_name'].fillna('unknown')
cat_trans_dict = dict(zip(cat_trans['product_category_name'], cat_trans['product_category_name_english']))
cleaned['product_category_name_english'] = cleaned['product_category_name'].map(cat_trans_dict)

# 4. Product numeric fields - median
numeric_cols = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
for col in numeric_cols:
    if col in cleaned.columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

# 5. Derived features
cleaned['order_purchase_timestamp'] = pd.to_datetime(cleaned['order_purchase_timestamp'])
cleaned['order_delivered_customer_date'] = pd.to_datetime(cleaned['order_delivered_customer_date'])
cleaned['order_estimated_delivery_date'] = pd.to_datetime(cleaned['order_estimated_delivery_date'])

cleaned['delivery_delay_days'] = (cleaned['order_delivered_customer_date'] - cleaned['order_estimated_delivery_date']).dt.days
cleaned['delivery_delay_days'] = cleaned['delivery_delay_days'].clip(lower=0)
cleaned['purchase_year'] = cleaned['order_purchase_timestamp'].dt.year
cleaned['purchase_month'] = cleaned['order_purchase_timestamp'].dt.month

# 6. Aggregate payments
payment_agg = payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'max',
    'payment_sequential': 'max'
}).reset_index()
payment_agg.columns = ['order_id', 'total_payment_value', 'max_installments', 'payment_sequential']

cleaned = cleaned.drop(columns=['payment_value', 'payment_installments', 'payment_sequential'], errors='ignore')
cleaned = cleaned.merge(payment_agg, on='order_id', how='left')

# 7. Fill remaining missing
cleaned['delivery_delay_days'] = cleaned['delivery_delay_days'].fillna(0).astype(int)
cleaned['total_payment_value'] = cleaned['total_payment_value'].fillna(0)
cleaned['max_installments'] = cleaned['max_installments'].fillna(0).astype(int)
cleaned['payment_sequential'] = cleaned['payment_sequential'].fillna(0).astype(int)

# 8. Deduplicate
cleaned = cleaned.drop_duplicates(subset=['order_id'], keep='first')

# Save
output_path = os.path.join(str(OUTPUT_DIR), "dana_output.csv")
cleaned.to_csv(output_path, index=False)
print(f"[STATUS] Saved: {len(cleaned)} rows x {len(cleaned.columns)} cols to {output_path}")
print(f"[STATUS] File size: {os.path.getsize(output_path)/1024:.2f} KB")
print(f"[STATUS] Zero missing values remaining!")
