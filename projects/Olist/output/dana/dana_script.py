
import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\input\\olist.sqlite')
parser.add_argument('--output-dir', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\output\\dana')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[STATUS] Connecting to SQLite database...")
conn = sqlite3.connect(INPUT_PATH)

# Check what tables actually exist
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(f"[STATUS] Tables found in database: {tables}")

# Build case-insensitive mapping directly
table_map = {}
table_lower_to_real = {}
for tbl in tables:
    tbl_lower = tbl.lower()
    table_lower_to_real[tbl_lower] = tbl

# Map expected table names (lowercase) to real table names
expected_tables = [
    'olist_customers', 'olist_geolocation', 'olist_order_items', 'olist_order_payments',
    'olist_order_reviews', 'olist_orders', 'olist_products', 'olist_sellers',
    'product_category_name_translation'
]

for expected in expected_tables:
    if expected in table_lower_to_real:
        table_map[expected] = table_lower_to_real[expected]
        print(f"[STATUS] Found '{expected}' -> '{table_map[expected]}'")
    else:
        for tbl_lower, tbl_real in table_lower_to_real.items():
            if expected.replace('_', '') in tbl_lower.replace('_', ''):
                table_map[expected] = tbl_real
                print(f"[STATUS] Matched '{expected}' -> '{tbl_real}'")
                break
        else:
            print(f"[WARN] Table '{expected}' not found in database")

print(f"[STATUS] Final table mapping: {table_map}")

# Check for missing tables
missing_tables = [t for t in expected_tables if t not in table_map]
if missing_tables:
    print(f"[ERROR] Missing tables: {missing_tables}")
    print(f"[ERROR] Cannot proceed without all required tables.")
    conn.close()
    exit(1)

print("[STATUS] Loading all tables...")

customers = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_customers']}]", conn)
print(f"[STATUS] Loaded customers: {customers.shape}")

geolocation = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_geolocation']}]", conn)
print(f"[STATUS] Loaded geolocation: {geolocation.shape}")

order_items = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_order_items']}]", conn)
print(f"[STATUS] Loaded order_items: {order_items.shape}")

order_payments = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_order_payments']}]", conn)
print(f"[STATUS] Loaded order_payments: {order_payments.shape}")

order_reviews = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_order_reviews']}]", conn)
print(f"[STATUS] Loaded order_reviews: {order_reviews.shape}")

orders = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_orders']}]", conn)
print(f"[STATUS] Loaded orders: {orders.shape}")

products = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_products']}]", conn)
print(f"[STATUS] Loaded products: {products.shape}")

sellers = pd.read_sql_query(f"SELECT * FROM [{table_map['olist_sellers']}]", conn)
print(f"[STATUS] Loaded sellers: {sellers.shape}")

category_translation = pd.read_sql_query(f"SELECT * FROM [{table_map['product_category_name_translation']}]", conn)
print(f"[STATUS] Loaded product_category_name_translation: {category_translation.shape}")

conn.close()
print("[STATUS] All tables loaded successfully")

# ─── JOIN ───
print("[STATUS] Joining tables...")

# Joining orders with order_items on order_id
df = orders.merge(order_items, on='order_id', how='left')
print(f"[STATUS] After order_items join: {df.shape}")

# Joining with order_payments on order_id
df = df.merge(order_payments, on='order_id', how='left')
print(f"[STATUS] After order_payments join: {df.shape}")

# Joining with order_reviews on order_id
df = df.merge(order_reviews, on='order_id', how='left')
print(f"[STATUS] After order_reviews join: {df.shape}")

# Joining with products on product_id
df = df.merge(products, on='product_id', how='left')
print(f"[STATUS] After products join: {df.shape}")

# Joining with category_translation on product_category_name
if 'product_category_name' in df.columns:
    df = df.merge(category_translation, on='product_category_name', how='left')
    print(f"[STATUS] After category_translation join: {df.shape}")

# Joining with sellers on seller_id
df = df.merge(sellers, on='seller_id', how='left')
print(f"[STATUS] After sellers join: {df.shape}")

# Joining with customers on customer_id
df = df.merge(customers, on='customer_id', how='left')
print(f"[STATUS] After customers join: {df.shape}")

# Remove duplicate columns from joins
df = df.loc[:, ~df.columns.duplicated()]
print(f"[STATUS] After removing duplicates: {df.shape}")

# ─── FEATURE ENGINEERING ───
print("[STATUS] Starting feature engineering...")

# Timestamp features
if 'order_purchase_timestamp' in df.columns:
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    print(f"[STATUS] Added timestamp features from purchase_timestamp")

if 'order_delivered_customer_date' in df.columns:
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')
if 'order_estimated_delivery_date' in df.columns:
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'], errors='coerce')
if 'order_purchase_timestamp' in df.columns and 'order_delivered_customer_date' in df.columns:
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    print(f"[STATUS] Added delivery_time_days feature")

# Geolocation features (if available, aggregate by zip code prefix)
if 'geolocation_zip_code_prefix' in geolocation.columns:
    geo_agg = geolocation.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()
    geo_agg.columns = ['zip_code_prefix', 'geo_lat_mean', 'geo_lng_mean']

    # Merge seller geolocation
    if 'seller_zip_code_prefix' in df.columns:
        seller_geo = geo_agg.rename(columns={'geo_lat_mean': 'seller_lat', 'geo_lng_mean': 'seller_lng'})
        df = df.merge(seller_geo, left_on='seller_zip_code_prefix', right_on='zip_code_prefix', how='left')
        df = df.drop(columns=['zip_code_prefix'])
        print(f"[STATUS] Added seller geolocation features")

    # Merge customer geolocation
    if 'customer_zip_code_prefix' in df.columns:
        # Reset index to avoid column name conflict
        geo_agg_cust = geo_agg.copy()
        geo_agg_cust.columns = ['zip_code_prefix', 'customer_lat', 'customer_lng']
        df = df.merge(geo_agg_cust, left_on='customer_zip_code_prefix', right_on='zip_code_prefix', how='left')
        df = df.drop(columns=['zip_code_prefix'], errors='ignore')
        print(f"[STATUS] Added customer geolocation features")

# Text features from review
if 'review_comment_message' in df.columns:
    df['review_comment_length'] = df['review_comment_message'].fillna('').str.len()
    print(f"[STATUS] Added review_comment_length feature")

if 'review_comment_title' in df.columns:
    df['review_comment_title_length'] = df['review_comment_title'].fillna('').str.len()
    print(f"[STATUS] Added review_comment_title_length feature")

# ─── CLEANING ───
print("[STATUS] Starting data cleaning...")

# Drop columns that are all NaN or single value
cols_before = df.shape[1]
df = df.dropna(axis=1, how='all')
cols_after = df.shape[1]
if cols_before != cols_after:
    print(f"[STATUS] Dropped {cols_before - cols_after} all-NaN columns")

# Fill numerical NaN with median for key monetary columns
if 'payment_value' in df.columns:
    med_val = df['payment_value'].median()
    df['payment_value'] = df['payment_value'].fillna(med_val)
    print(f"[STATUS] Filled payment_value NaN with median: {med_val}")

if 'price' in df.columns:
    med_price = df['price'].median()
    df['price'] = df['price'].fillna(med_price)
    print(f"[STATUS] Filled price NaN with median: {med_price}")

if 'freight_value' in df.columns:
    med_freight = df['freight_value'].median()
    df['freight_value'] = df['freight_value'].fillna(med_freight)
    print(f"[STATUS] Filled freight_value NaN with median: {med_freight}")

if 'product_weight_g' in df.columns:
    med_weight = df['product_weight_g'].median()
    df['product_weight_g'] = df['product_weight_g'].fillna(med_weight)
    print(f"[STATUS] Filled product_weight_g NaN with median: {med_weight}")

if 'product_length_cm' in df.columns:
    med_len = df['product_length_cm'].median()
    df['product_length_cm'] = df['product_length_cm'].fillna(med_len)
    print(f"[STATUS] Filled product_length_cm NaN with median: {med_len}")

if 'product_height_cm' in df.columns:
    med_hei = df['product_height_cm'].median()
    df['product_height_cm'] = df['product_height_cm'].fillna(med_hei)
    print(f"[STATUS] Filled product_height_cm NaN with median: {med_hei}")

if 'product_width_cm' in df.columns:
    med_wid = df['product_width_cm'].median()
    df['product_width_cm'] = df['product_width_cm'].fillna(med_wid)
    print(f"[STATUS] Filled product_width_cm NaN with median: {med_wid}")

# Fill categorical NaN with 'Unknown'
cat_cols = ['product_category_name', 'product_category_name_english',
            'review_comment_message', 'review_comment_title', 'review_score']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        print(f"[STATUS] Filled {col} NaN with 'Unknown'")

# Fill delivery_time_days for undelivered orders with a placeholder
if 'delivery_time_days' in df.columns:
    df['delivery_time_days'] = df['delivery_time_days'].fillna(-1)
    print(f"[STATUS] Filled delivery_time_days NaN with -1 (undelivered)")

# Drop geolocation columns if they exist and have too many NaN
for col in ['geolocation_lat', 'geolocation_lng', 'seller_lat', 'seller_lng',
            'customer_lat', 'customer_lng']:
    if col in df.columns and df[col].isna().sum() > 0.5 * len(df):
        df = df.drop(columns=[col])
        print(f"[STATUS] Dropped {col} due to >50% missing")

# Remove duplicate rows
dups_before = len(df)
df = df.drop_duplicates()
dups_after = len(df)
if dups_before != dups_after:
    print(f"[STATUS] Removed {dups_before - dups_after} duplicate rows")

# Final stats
print(f"[STATUS] Final shape: {df.shape}")
print(f"[STATUS] Columns: {list(df.columns)}")
print(f"[STATUS] Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"[STATUS] Missing data summary:\n{df.isna().sum().to_string()}")

# ─── EXPORT ───
output_csv_path = os.path.join(OUTPUT_DIR, 'dana_output.csv')
df.to_csv(output_csv_path, index=False)
print(f"[STATUS] Saved dana_output.csv to {output_csv_path}")
print(f"[STATUS] CSV file size: {os.path.getsize(output_csv_path) / 1024**2:.2f} MB")
print(f"[STATUS] Data cleaning complete ✓")
