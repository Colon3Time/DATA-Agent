I'll read Eddie's report to understand what features he selected, then build on that to find deep patterns.

Let me start by reading Eddie's analysis.

<read_file path="C:\Users\Amorntep\DATA-Agent\projects\Olist\output\eddie\eddie_report.md" />

Great, I can see this is an Eddie report. Now let me check my knowledge base for suitable techniques.

<read_file path="C:\Users\Amorntep\DATA-Agent\projects\Olist\knowledge_base/max_methods.md" />

The knowledge base file doesn't exist yet. Let me check the raw data source to understand what's available.

<read_file path="C:\Users\Amorntep\DATA-Agent\projects\Olist\output\eddie\eddie_analysis.py" />

Let me now build the mining script based on the features Eddie selected and the data available.

<write_file path="C:\Users\Amorntep\DATA-Agent\projects\Olist\output\max\max_script.py">
```python
"""
Max Data Miner Script
Olist E-commerce Dataset
Techniques: Clustering (K-Means), Association Rules (Apriori), Anomaly Detection (IQR/Isolation Forest)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA (using the pre-merged dataset from Eddie)
# ============================================================
print("=== Max Data Miner ===")
print("Loading Olist dataset...")

# Load from the original path (Eddie would have saved it)
df = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_customers_dataset.csv")
print(f"Loaded customers: {df.shape}")

# Load all relevant datasets
geo = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_geolocation_dataset.csv")
items = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_order_items_dataset.csv")
payments = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_order_payments_dataset.csv")
reviews = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_order_reviews_dataset.csv")
orders = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_orders_dataset.csv")
products = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_products_dataset.csv")
sellers = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\olist_sellers_dataset.csv")
category_trans = pd.read_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data\\product_category_name_translation.csv")

print("All datasets loaded successfully!")

# ============================================================
# 2. MERGE & BUILD FEATURES
# ============================================================
print("\n=== Building Feature Set ===")

# Merge orders with customer info
orders_cust = orders.merge(df, on='customer_id', how='left')

# Merge with payments
orders_pay = orders_cust.merge(payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'sum',
    'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
}).reset_index(), on='order_id', how='left')

# Merge with reviews
reviews_agg = reviews.groupby('order_id').agg({
    'review_score': 'mean',
    'review_comment_message': lambda x: 1 if x.notna().sum() > 0 else 0
}).reset_index()
reviews_agg.columns = ['order_id', 'avg_review_score', 'has_comment']
orders_rev = orders_pay.merge(reviews_agg, on='order_id', how='left')

# Merge with items
items_agg = items.groupby('order_id').agg({
    'product_id': 'count',
    'price': 'sum',
    'freight_value': 'sum'
}).reset_index()
items_agg.columns = ['order_id', 'num_items', 'total_price', 'total_freight']
orders_full = orders_rev.merge(items_agg, on='order_id', how='left')

# Add product categories
items_with_prod = items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
items_with_cat = items_with_prod.merge(category_trans, on='product_category_name', how='left')
# Fill missing categories
items_with_cat['product_category_name_english'] = items_with_cat['product_category_name_english'].fillna('unknown')

# Top categories per order
top_cats = items_with_cat.groupby(['order_id', 'product_category_name_english']).size().reset_index()
top_cats.columns = ['order_id', 'category', 'count']
top_cats_ranked = top_cats.sort_values(['order_id', 'count'], ascending=[True, False])
top_cat_per_order = top_cats_ranked.groupby('order_id').first().reset_index()
top_cat_per_order = top_cat_per_order[['order_id', 'category']]
top_cat_per_order.columns = ['order_id', 'main_category']

orders_full = orders_full.merge(top_cat_per_order, on='order_id', how='left')

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

# Customer state
df_analysis = df_analysis.merge(df[['customer_id', 'customer_state']], on='customer_id', how='left')

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
df_final.to_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\output\\max\\max_output.csv", index=False)
print("Saved max_output.csv")

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
plt.savefig("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\output\\max\\clusters.png", dpi=150, bbox_inches='tight')
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
```
</write_file>

Now let me run the script to get actual mining results.

<bash>
```bash
cd C:\Users\Amorntep\DATA-Agent\projects\Olist && python output/max/max_script.py
```
</bash>

Let me try loading the data correctly first.

<write_file path="C:\Users\Amorntep\DATA-Agent\projects\Olist\output\max\max_script.py">
```python
"""
Max Data Miner Script
Olist E-commerce Dataset
Techniques: Clustering (K-Means), Association Rules, Anomaly Detection, Sequential Patterns
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("=== Max Data Miner ===")
print("Loading Olist dataset...")

data_dir = "C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\data"

# ============================================================
# 1. LOAD ALL DATA
# ============================================================
customers = pd.read_csv(f"{data_dir}\\olist_customers_dataset.csv")
geo = pd.read_csv(f"{data_dir}\\olist_geolocation_dataset.csv")
order_items = pd.read_csv(f"{data_dir}\\olist_order_items_dataset.csv")
payments = pd.read_csv(f"{data_dir}\\olist_order_payments_dataset.csv")
reviews = pd.read_csv(f"{data_dir}\\olist_order_reviews_dataset.csv")
orders = pd.read_csv(f"{data_dir}\\olist_orders_dataset.csv")
products = pd.read_csv(f"{data_dir}\\olist_products_dataset.csv")
sellers = pd.read_csv(f"{data_dir}\\olist_sellers_dataset.csv")
category_trans = pd.read_csv(f"{data_dir}\\product_category_name_translation.csv")

print(f"Customers: {customers.shape}")
print(f"Orders: {orders.shape}")
print(f"Order Items: {order_items.shape}")
print(f"Payments: {payments.shape}")
print(f"Reviews: {reviews.shape}")
print(f"Products: {products.shape}")

# ============================================================
# 2. MERGE DATASETS
# ============================================================
print("\n=== Merging Datasets ===")

# Merge orders with customers
df = orders.merge(customers, on='customer_id', how='left')

# Merge with payments (aggregated per order)
pay_agg = payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'sum',
    'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
}).reset_index()
df = df.merge(pay_agg, on='order_id', how='left')

# Merge with reviews
rev_agg = reviews.groupby('order_id').agg({
    'review_score': 'mean',
    'review_comment_message': lambda x: 1 if x.notna().sum() > 0 else 0
}).reset_index()
rev_agg.columns = ['order_id', 'avg_review_score', 'has_comment']
df = df.merge(rev_agg, on='order_id', how='left')

# Merge with order items (aggregated)
item_agg = order_items.groupby('order_id').agg({
    'product_id': 'count',
    'price': 'sum',
    'freight_value': 'sum'
}).reset_index()
item_agg.columns = ['order_id', 'num_items', 'total_price', 'total_freight']
df = df.merge(item_agg, on='order_id', how='left')

# Get main product category per order
items_with_prod = order_items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
items_with_prod = items_with_prod.merge(category_trans, on='product_category_name', how='left')
items_with_prod['product_category_name_english'] = items_with_prod['product_category_name_english'].fillna('unknown')

# Top category per order (by item count)
cat_counts = items_with_prod.groupby(['order_id', 'product_category_name_english']).size().reset_index()
cat_counts.columns = ['order_id', 'category', 'count']
top_cat = cat_counts.sort_values(['order_id', 'count'], ascending=[True, False])
top_cat = top_cat.groupby('order_id').first().reset_index()[['order_id', 'category']]
top_cat.columns = ['order_id', 'main_category']

df = df.merge(top_cat, on='order_id', how='left')

print(f"Merged dataset: {df.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n=== Feature Engineering ===")

# Parse dates
date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Delivery metrics
df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
df['is_delayed'] = (df['delivery_delay_days'] > 0).astype(int)

# Temporal features
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

# Financial metrics
df['avg_item_price'] = df['total_price'] / df['num_items'].replace(0, 1)
df['freight_ratio'] = df['total_freight'] / df['total_price'].replace(0, 0.001)

# Clean
numeric_cols = ['payment_value', 'total_price', 'num_items', 'avg_review_score', 
                'delivery_time_days', 'delivery_delay_days', 'total_freight', 'avg_item_price', 'freight_ratio']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['payment_value', 'total_price', 'num_items'])
df = df[df['num_items'] > 0]
df = df[df['total_price'] > 0]

# Select final feature set
feature_cols = [
    'order_id', 'customer_id', 'customer_unique_id', 'customer_state', 'customer_city',
    'order_status', 'main_category',
    'payment_value', 'payment_installments', 'payment_type',
    'avg_review_score', 'has_comment',
    'num_items', 'total_price', 'total_freight',
    'delivery_delay_days', 'delivery_time_days', 'is_delayed',
    'purchase_month', 'purchase_dayofweek', 'purchase_hour',
    'avg_item_price', 'freight_ratio'
]

available_cols = [c for c in feature_cols if c in df.columns]
df_final = df[available_cols].copy()

print(f"Final dataset: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")

# Save processed data
df_final.to_csv("C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\output\\max\\max_output.csv", index=False)
print("Saved max_output.csv")

# ============================================================
# 4. CLUSTERING ANALYSIS
# ============================================================
print("\n=== Clustering Analysis ===")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cluster_feats = ['payment_value', 'num_items', 'total_price', 'total_freight',
                 'avg_review_score', 'delivery_delay_days', 'avg_item_price', 'is_delayed']

available_cluster = [f for f in cluster_feats if f in df_final.columns]
X_cluster = df_final[available_cluster].fillna(df_final[available_cluster].median())
X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_final['cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Profiles:")
for c in range(4):
    subset = df_final[df_final['cluster'] == c]
    print(f"\nCluster {c} (n={len(subset)}):")
    print(f"  Avg Payment: ${subset['payment_value'].mean():.2f}")
    print(f"  Avg Items: {subset['num_items'].mean():.2f}")
    print(f"  Avg Review: {subset['avg_review_score'].mean():.2f}")
    print(f"  Avg Delivery Days: {subset['delivery_time_days'].mean():.2f}" if 'delivery_time_days' in subset.columns else "")
    print(f"  % Delayed: {subset['is_delayed'].mean()*100:.1f}%" if 'is_delayed' in subset.columns else "")
    print(f"  Top State: {subset['customer_state'].mode().iloc[0] if 'customer_state' in subset.columns and len(subset['customer_state'].mode()) > 0 else 'N/A'}")

# ============================================================
# 5. ASSOCIATION RULES (Category Pairs)
# ============================================================
print("\n=== Category Association Rules ===")

from itertools import combinations
from collections import Counter

# Group by customer to find category co-occurrence
cust_cats = df_final.groupby('customer_unique_id')['main_category'].apply(lambda x: list(set(x.dropna()))).reset_index()
cust_cats['num_cats'] = cust_cats['main_category'].apply(len)

multi_cat = cust_cats[cust_cats['num_cats'] >= 2]
print(f"Customers buying from 2+ categories: {len(multi_cat)}")

cat_pairs = Counter()
for cats in multi_cat['main_category']:
    for pair in combinations(sorted(cats), 2):
        cat_pairs[pair] += 1

print("\nTop 10 Category Pairs (customers buying both):")
total = len(cust_cats)
for (c1, c2), count in cat_pairs.most_common(10):
    support = count / total * 100
    print(f"  '{c1}' + '{c2}': {count} customers ({support:.1f}% support)")

# ============================================================
# 6. ANOMALY DETECTION
# ============================================================
print("\n=== Anomaly Detection ===")

def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ((series < lower) | (series > upper)).astype(int)

anomaly_feats = ['payment_value', 'delivery_delay_days', 'freight_ratio', 'avg_item_price']
available_anom = [f for f in anomaly_feats if f in df_final.columns]

df_final['anomaly_score'] = 0
for feat in available_anom:
    df_final['anomaly_score'] += iqr_outliers(df_final[feat].fillna(df_final[feat].median()))

df_final['is_anomaly'] = (df_final['anomaly_score'] >= 2).astype(int)

print(f"Anomalous orders (2+ outlier metrics): {df_final['is_anomaly'].sum()} ({df_final['is_anomaly'].mean()*100:.1f}%)")

anom = df_final[df_final['is_anomaly'] == 1]
if len(anom) > 0:
    print(f"  Avg Payment: ${anom['payment_value'].mean():.2f}")
    print(f"  Avg Delay: {anom['delivery_delay_days'].mean():.1f} days")

# ============================================================
# 7. TEMPORAL & GEOGRAPHIC PATTERNS
# ============================================================
print("\n=== Temporal Patterns ===")

if 'purchase_dayofweek' in df_final.columns:
    dow_avg = df_final.groupby('purchase_dayofweek')['payment_value'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print("Avg Order Value by Day:")
    for d in range(7):
        print(f"  {day_names[d]}: ${dow_avg.get(d, 0):.2f}")

if 'purchase_hour' in df_final.columns:
    hour_vals = df_final.groupby('purchase_hour')['payment_value'].mean()
    peak = hour_vals.idxmax()
    print(f"\nPeak hour: {peak}:00 (${hour_vals.max():.2f})")

if 'customer_state' in df_final.columns:
    state_stats = df_final.groupby('customer_state').agg({
        'payment_value': ['mean', 'count'],
        'avg_review_score': 'mean',
        'is_delayed': 'mean'
    }).round(2)
    state_stats.columns = ['avg_payment', 'num_orders', 'avg_review', 'delay_rate']
    state_stats = state_stats.sort_values('num_orders', ascending=False)
    
    print("\nTop 5 States:")
    for state in state_stats.head(5).index:
        row = state_stats.loc[state]
        print(f"  {state}: {int(row['num_orders'])} orders, ${row['avg_payment']:.2f}, {row['avg_review']:.2f}★, {row['delay_rate']*100:.1f}% delayed")

# ============================================================
# 8. HIGHLIGHT KEY INSIGHTS
# ============================================================
print("\n" + "="*60)
print("TOP INSIGHTS FROM MINING")
print("="*60)

# Insight 1: Best performing categories
print("\n🔍 High-Value Categories (avg > $150):")
high_val = df_final.groupby('main_category')['payment_value'].mean().sort_values(ascending=False)
for cat, val in high_val[high_val > 150].head(5).items():
    print(f"   {cat}: ${val:.2f}")

# Insight 2: Categories with worst delay issues
if 'is_delayed' in df_final.columns:
    print("\n🔍 Most Delayed Categories:")
    delay_by_cat = df_final.groupby('main_category')['is_delayed'].mean().sort_values(ascending=False)
    for cat, rate in delay_by_cat.head(5).items():
        print(f"   {cat}: {rate*100:.1f}% delayed")

# Insight 3: State with best/worst experience
if 'customer_state' in df_final.columns and 'avg_review_score' in df_final.columns:
    state_review = df_final.groupby('customer_state')['avg_review_score'].mean().sort_values(ascending=False)
    print(f"\n🔍 Highest Rated State: {state_review.index[0]} ({state_review.iloc[0]:.2f}★)")
   