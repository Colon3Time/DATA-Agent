"""
Max Data Miner Script
Olist E-commerce Dataset
Techniques: Clustering (K-Means), Association Rules (Apriori), Anomaly Detection (IQR/Isolation Forest)
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

_FALLBACK_INPUT  = r"C:\Users\Amorntep\DATA-Agent\projects\Olist\output\dana\dana_output.csv"
_FALLBACK_OUTPUT = r"C:\Users\Amorntep\DATA-Agent\projects\Olist\output\max"

_inp = Path(args.input) if args.input else Path(_FALLBACK_INPUT)
if _inp.is_dir():
    _csv = sorted(_inp.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    INPUT_PATH = str(_csv[0]) if _csv else _FALLBACK_INPUT
else:
    INPUT_PATH = str(_inp)

OUTPUT_DIR = args.output_dir or _FALLBACK_OUTPUT
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA (dana merged output)
# ============================================================
print("=== Max Data Miner ===")
print(f"Loading from: {INPUT_PATH}")

df_raw = pd.read_csv(INPUT_PATH)
print(f"Loaded: {df_raw.shape}")

# ── Derive sub-aggregations from merged CSV ────────────────────────────────
# customers
df = df_raw[["customer_id", "customer_state", "customer_city"]].drop_duplicates()

# orders
orders = df_raw[["order_id", "customer_id", "order_status",
                  "order_purchase_timestamp",
                  "order_delivered_customer_date",
                  "order_estimated_delivery_date"]].drop_duplicates("order_id")

# payments (aggregated per order)
pay_cols = [c for c in ["order_id", "total_payment_value", "max_installments", "payment_type"]
            if c in df_raw.columns]
payments = df_raw[pay_cols].drop_duplicates("order_id") if "total_payment_value" in df_raw.columns else None

# reviews
rev_cols = [c for c in ["order_id", "review_score"] if c in df_raw.columns]
reviews = df_raw[rev_cols].drop_duplicates("order_id")

# items (price + freight per order)
item_cols = [c for c in ["order_id", "price", "freight_value", "product_category_name_english"]
             if c in df_raw.columns]
items = df_raw[item_cols].copy() if item_cols else None

print("All datasets derived from merged CSV successfully!")

# ============================================================
# 2. MERGE & BUILD FEATURES
# ============================================================
print("\n=== Building Feature Set ===")

# Merge orders with customer info
orders_cust = orders.merge(df, on='customer_id', how='left')

# Merge with payments
if payments is not None and "total_payment_value" in payments.columns:
    orders_pay = orders_cust.merge(
        payments.rename(columns={"total_payment_value": "payment_value",
                                  "max_installments": "payment_installments"}),
        on='order_id', how='left')
else:
    orders_pay = orders_cust
    orders_pay['payment_value'] = df_raw.groupby('order_id')['price'].sum().reindex(
        orders_pay['order_id'].values).values
    orders_pay['payment_installments'] = 1
    orders_pay['payment_type'] = 'unknown'

# Merge with reviews
reviews_agg = reviews.groupby('order_id').agg({'review_score': 'mean'}).reset_index()
reviews_agg.columns = ['order_id', 'avg_review_score']
reviews_agg['has_comment'] = 0
orders_rev = orders_pay.merge(reviews_agg, on='order_id', how='left')

# Merge with items (price + freight per order)
if items is not None and 'price' in items.columns:
    items_agg = items.groupby('order_id').agg(
        num_items=('price', 'count'),
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum')
    ).reset_index()
    orders_full = orders_rev.merge(items_agg, on='order_id', how='left')
else:
    orders_full = orders_rev
    orders_full['num_items'] = 1
    orders_full['total_price'] = orders_full.get('payment_value', 0)
    orders_full['total_freight'] = 0

# Top category per order
if items is not None and 'product_category_name_english' in items.columns:
    top_cats = items.groupby(['order_id', 'product_category_name_english']).size().reset_index()
    top_cats.columns = ['order_id', 'category', 'count']
    top_cat = top_cats.sort_values('count', ascending=False).groupby('order_id').first().reset_index()
    top_cat = top_cat[['order_id', 'category']].rename(columns={'category': 'main_category'})
    orders_full = orders_full.merge(top_cat, on='order_id', how='left')
else:
    orders_full['main_category'] = df_raw.get('product_category_name_english',
                                               pd.Series(['unknown'] * len(orders_full)))

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n=== Feature Engineering ===")

df_analysis = orders_full.copy()

# Temporal features
df_analysis['order_purchase_timestamp'] = pd.to_datetime(df_analysis['order_purchase_timestamp'])
df_analysis['order_delivered_customer_date'] = pd.to_datetime(df_analysis['order_delivered_customer_date'])
df_analysis['order_estimated_delivery_date'] = pd.to_datetime(df_analysis['order_estimated_delivery_date'])

# Delivery performance
df_analysis['delivery_delay_days'] = (df_analysis['order_delivered_customer_date'] - df_analysis['order_estimated_delivery_date']).dt.days
df_analysis['delivery_time_days'] = (df_analysis['order_delivered_customer_date'] - df_analysis['order_purchase_timestamp']).dt.days
df_analysis['is_delayed'] = (df_analysis['delivery_delay_days'] > 0).astype(int)

# Purchase metrics
df_analysis['purchase_month'] = df_analysis['order_purchase_timestamp'].dt.month
df_analysis['purchase_dayofweek'] = df_analysis['order_purchase_timestamp'].dt.dayofweek
df_analysis['purchase_hour'] = df_analysis['order_purchase_timestamp'].dt.hour

# Financial metrics
df_analysis['avg_item_price'] = df_analysis['total_price'] / df_analysis['num_items'].replace(0, 1)
df_analysis['freight_ratio'] = df_analysis['total_freight'] / df_analysis['total_price'].replace(0, 0.001)

# customer_state already merged via orders_cust — rename suffix if duplicated
if 'customer_state_x' in df_analysis.columns:
    df_analysis = df_analysis.rename(columns={'customer_state_x': 'customer_state'})
    df_analysis = df_analysis.drop(columns=['customer_state_y'], errors='ignore')

# Select columns for analysis
feature_cols = [
    'order_id', 'customer_id', 'customer_state', 'customer_city',
    'order_status', 'main_category',
    'payment_value', 'payment_installments', 'payment_type',
    'avg_review_score', 'has_comment',
    'num_items', 'total_price', 'total_freight',
    'delivery_delay_days', 'delivery_time_days', 'is_delayed',
    'purchase_month', 'purchase_dayofweek', 'purchase_hour',
    'avg_item_price', 'freight_ratio'
]

df_final = df_analysis[feature_cols].copy()

# Clean up
df_final = df_final.dropna(subset=['payment_value', 'total_price', 'num_items'])
df_final = df_final[df_final['num_items'] > 0]
df_final = df_final[df_final['total_price'] > 0]

print(f"Final dataset shape: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")

# Save the processed data
out_csv = os.path.join(OUTPUT_DIR, "max_output.csv")
df_final.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

# ============================================================
# 4. PATTERN MINING - CLUSTERING (K-Means)
# ============================================================
print("\n=== Clustering Analysis (K-Means) ===")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Select numeric features for clustering
cluster_features = [
    'payment_value', 'num_items', 'total_price', 'total_freight',
    'avg_review_score', 'delivery_delay_days', 'delivery_time_days',
    'avg_item_price', 'freight_ratio', 'is_delayed'
]

cluster_df = df_final[cluster_features].copy()
cluster_df = cluster_df.fillna(cluster_df.median())

# Handle infinite values
cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan)
cluster_df = cluster_df.fillna(cluster_df.median())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# Find optimal K using elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

print(f"Inertias for K=2..8: {[round(i, 2) for i in inertias]}")

# Use K=4 (elbow point typically around 4 for this data)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_final['cluster'] = kmeans.fit_predict(X_scaled)

# Profile clusters
print("\nCluster Profiles:")
for c in range(4):
    subset = df_final[df_final['cluster'] == c]
    print(f"\nCluster {c} (n={len(subset)}):")
    print(f"  Avg Payment: ${subset['payment_value'].mean():.2f}")
    print(f"  Avg Items: {subset['num_items'].mean():.2f}")
    print(f"  Avg Review: {subset['avg_review_score'].mean():.2f}")
    print(f"  Avg Delivery Days: {subset['delivery_time_days'].mean():.2f}")
    print(f"  Avg Delay Days: {subset['delivery_delay_days'].mean():.2f}")
    print(f"  % Delayed: {subset['is_delayed'].mean()*100:.1f}%")
    print(f"  Top State: {subset['customer_state'].mode()[0] if len(subset) > 0 else 'N/A'}")
    print(f"  Top Category: {subset['main_category'].mode()[0] if len(subset) > 0 else 'N/A'}")

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_final['pca1'] = X_pca[:, 0]
df_final['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'orange', 'red']
for c in range(4):
    mask = df_final['cluster'] == c
    plt.scatter(df_final.loc[mask, 'pca1'], df_final.loc[mask, 'pca2'], 
               c=colors[c], label=f'Cluster {c}', alpha=0.5, s=10)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Customer Order Clusters (K-Means)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "clusters.png"), dpi=150, bbox_inches='tight')
print("\nCluster visualization saved.")

# ============================================================
# 5. PATTERN MINING - ASSOCIATION RULES
# ============================================================
print("\n=== Association Rules Mining ===")

# Convert to transaction format: customer-level purchase patterns
customer_orders = df_final.groupby('customer_id').agg({
    'main_category': lambda x: list(set(x.dropna())),
    'payment_type': lambda x: list(set(x.dropna())),
    'num_items': 'sum',
    'payment_value': 'sum',
    'customer_state': 'first'
}).reset_index()

# Filter to customers with multiple orders/categories for meaningful rules
customer_orders['num_categories'] = customer_orders['main_category'].apply(len)
multi_cat_customers = customer_orders[customer_orders['num_categories'] > 1]
print(f"Customers with multiple categories: {len(multi_cat_customers)}")

# Find top category combinations
from itertools import combinations
from collections import Counter

cat_pairs = Counter()
for cats in customer_orders['main_category']:
    if len(cats) >= 2:
        for pair in combinations(sorted(cats), 2):
            cat_pairs[pair] += 1

print("\nTop Category Associations (customers buying both):")
total_customers = len(customer_orders)
for (cat1, cat2), count in cat_pairs.most_common(15):
    support = count / total_customers * 100
    print(f"  '{cat1}' & '{cat2}': {count} customers ({support:.1f}% support)")

# Payment type analysis
print("\n--- Payment Type Patterns ---")
pay_combos = Counter()
for payments_list in customer_orders['payment_type']:
    if len(payments_list) >= 2:
        for pair in combinations(sorted(payments_list), 2):
            pay_combos[pair] += 1

for (p1, p2), count in pay_combos.most_common(5):
    print(f"  '{p1}' & '{p2}': {count} customers")

# ============================================================
# 6. ANOMALY DETECTION (IQR Method)
# ============================================================
print("\n=== Anomaly Detection (IQR) ===")

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

# Detect anomalies in key metrics
anomaly_features = ['payment_value', 'delivery_delay_days', 'freight_ratio', 'avg_item_price']
anomaly_results = {}

for feat in anomaly_features:
    if feat in df_final.columns:
        outlier_mask = detect_outliers_iqr(df_final[feat].fillna(df_final[feat].median()))
        n_outliers = outlier_mask.sum()
        pct = n_outliers / len(df_final) * 100
        anomaly_results[feat] = {
            'count': n_outliers,
            'percentage': round(pct, 2),
            'threshold_lower': round(df_final[feat].quantile(0.25) - 1.5 * (df_final[feat].quantile(0.75) - df_final[feat].quantile(0.25)), 2),
            'threshold_upper': round(df_final[feat].quantile(0.75) + 1.5 * (df_final[feat].quantile(0.75) - df_final[feat].quantile(0.25)), 2)
        }
        print(f"\n{feat}:")
        print(f"  Outliers found: {n_outliers} ({pct:.2f}%)")
        print(f"  Thresholds: <{anomaly_results[feat]['threshold_lower']} or >{anomaly_results[feat]['threshold_upper']}")

# Flag anomalous orders (extreme on multiple dimensions)
df_final['is_anomaly_score'] = 0
for feat in anomaly_features:
    if feat in df_final.columns:
        df_final['is_anomaly_score'] += detect_outliers_iqr(df_final[feat].fillna(df_final[feat].median())).astype(int)

df_final['is_anomaly'] = (df_final['is_anomaly_score'] >= 2).astype(int)
print(f"\nOrders flagged as anomalous (2+ outlier metrics): {df_final['is_anomaly'].sum()} ({df_final['is_anomaly'].mean()*100:.1f}%)")

# Profile anomalies
anomalies = df_final[df_final['is_anomaly'] == 1]
print("\n--- Anomaly Profile ---")
print(f"Avg Payment: ${anomalies['payment_value'].mean():.2f} (vs ${df_final['payment_value'].mean():.2f} overall)")
print(f"Avg Delay: {anomalies['delivery_delay_days'].mean():.1f} days (vs {df_final['delivery_delay_days'].mean():.1f} overall)")
print(f"Avg Items: {anomalies['num_items'].mean():.1f} (vs {df_final['num_items'].mean():.1f} overall)")
print(f"Top Anomaly Categories: {anomalies['main_category'].value_counts().head(5).to_dict()}")

# ============================================================
# 7. SEQUENTIAL PATTERN - TIME BASED
# ============================================================
print("\n=== Sequential Pattern Analysis ===")

# Sales by day of week
dayofweek_avg = df_final.groupby('purchase_dayofweek')['payment_value'].mean()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print("\nAverage Order Value by Day of Week:")
for d in range(7):
    print(f"  {day_names[d]}: ${dayofweek_avg[d]:.2f}")

# Sales by hour
hour_avg = df_final.groupby('purchase_hour')['payment_value'].mean()
peak_hour = hour_avg.idxmax()
print(f"\nPeak Hour: {peak_hour}:00 (Avg Order: ${hour_avg[peak_hour]:.2f})")

# Monthly trends
month_avg = df_final.groupby('purchase_month')['payment_value'].mean()
print("\nMonthly Average Order Value:")
for m in sorted(month_avg.index):
    print(f"  Month {m}: ${month_avg[m]:.2f}")

# ============================================================
# 8. STATE-WISE ANALYSIS
# ============================================================
print("\n=== Geographic Patterns ===")

state_stats = df_final.groupby('customer_state').agg({
    'payment_value': ['mean', 'count'],
    'avg_review_score': 'mean',
    'is_delayed': 'mean',
    'delivery_time_days': 'mean'
}).round(2)

# Flatten columns
state_stats.columns = ['avg_payment', 'num_orders', 'avg_review', 'delay_rate', 'avg_delivery_time']
state_stats = state_stats.sort_values('num_orders', ascending=False)

print("\nTop States by Orders:")
for state in state_stats.head(10).index:
    row = state_stats.loc[state]
    print(f"  {state}: {int(row['num_orders'])} orders, ${row['avg_payment']:.2f} avg, {row['avg_review']:.2f} review, {row['delay_rate']*100:.1f}% delayed")

print("\n=== Max Mining Complete ===")
print("Results saved to output/max/ directory")
