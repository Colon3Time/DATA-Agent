Dana Cleaning Report
============================================================

Before: 119143 rows, 44 columns
After: 119143 rows, 44 columns

Missing Values:
----------------------------------------
- review_comment_title: Dropped (>85% missing)
- order_approved_at: 177 missing -> 177 (KNN Imputation)
- order_delivered_carrier_date: 2086 missing -> 2086 (KNN Imputation)
- order_delivered_customer_date: 3421 missing -> 3421 (KNN Imputation)
- order_item_id: 833 missing -> 0 (KNN Imputation)
- product_id: 833 missing -> 833 (KNN Imputation)
- seller_id: 833 missing -> 833 (KNN Imputation)
- shipping_limit_date: 833 missing -> 833 (KNN Imputation)
- price: 833 missing -> 0 (KNN Imputation)
- freight_value: 833 missing -> 0 (KNN Imputation)
- payment_sequential: 3 missing -> 0 (KNN Imputation)
- payment_type: 3 missing -> 3 (KNN Imputation)
- payment_installments: 3 missing -> 0 (KNN Imputation)
- payment_value: 3 missing -> 0 (KNN Imputation)
- review_id: 997 missing -> 997 (KNN Imputation)
- review_score: 997 missing -> 0 (KNN Imputation)
- review_comment_message: 68898 missing -> 68898 (KNN Imputation)
- review_creation_date: 997 missing -> 997 (KNN Imputation)
- review_answer_timestamp: 997 missing -> 997 (KNN Imputation)
- product_category_name: 2542 missing -> 2542 (KNN Imputation)
- product_name_lenght: 2542 missing -> 0 (KNN Imputation)
- product_description_lenght: 2542 missing -> 0 (KNN Imputation)
- product_photos_qty: 2542 missing -> 0 (KNN Imputation)
- product_weight_g: 853 missing -> 0 (KNN Imputation)
- product_length_cm: 853 missing -> 0 (KNN Imputation)
- product_height_cm: 853 missing -> 0 (KNN Imputation)
- product_width_cm: 853 missing -> 0 (KNN Imputation)
- seller_zip_code_prefix: 833 missing -> 0 (KNN Imputation)
- seller_city: 833 missing -> 833 (KNN Imputation)
- seller_state: 833 missing -> 833 (KNN Imputation)
- customer_avg_lat: 322 missing -> 0 (KNN Imputation)
- customer_avg_lng: 322 missing -> 0 (KNN Imputation)
- seller_avg_lat: 1098 missing -> 0 (KNN Imputation)
- seller_avg_lng: 1098 missing -> 0 (KNN Imputation)
- product_category_name_english: 2567 missing -> 2567 (KNN Imputation)

Outlier Detection:
----------------------------------------
Method: Isolation Forest (contamination=0.05) + IQR (1.5x)
Total outliers found: 121068
- Likely Error (flagged): 10451 rows
  * row 81: order_item_id=4.00 (IQR outlier: order_item_id=4.0000 (bounds: 1.0000, 1.0000))
  * row 91: order_item_id=4.00 (IQR outlier: order_item_id=4.0000 (bounds: 1.0000, 1.0000))
  * row 92: order_item_id=5.00 (IQR outlier: order_item_id=5.0000 (bounds: 1.0000, 1.0000))
  * row 182: order_item_id=4.00 (IQR outlier: order_item_id=4.0000 (bounds: 1.0000, 1.0000))
  * row 183: order_item_id=5.00 (IQR outlier: order_item_id=5.0000 (bounds: 1.0000, 1.0000))
  * ... and 10446 more
- Likely Real (flagged): 110617 rows
  * row 14: order_item_id=2.00 (IQR outlier: order_item_id=2.0000 (bounds: 1.0000, 1.0000))
  * row 33: order_item_id=2.00 (IQR outlier: order_item_id=2.0000 (bounds: 1.0000, 1.0000))
  * row 60: order_item_id=2.00 (IQR outlier: order_item_id=2.0000 (bounds: 1.0000, 1.0000))
  * row 66: order_item_id=2.00 (IQR outlier: order_item_id=2.0000 (bounds: 1.0000, 1.0000))
  * row 74: order_item_id=2.00 (IQR outlier: order_item_id=2.0000 (bounds: 1.0000, 1.0000))
  * ... and 110612 more
- Uncertain: 0 rows

outlier_flags.csv: 121068 rows

Data Quality Score:
----------------------------------------
Completeness: Before 96.0% -> After 98.3%
Validity: Before 100.0% -> After 91.2%
Overall: Before 98.0% -> After 94.8%

Column Stats (Before -> After):
----------------------------------------
- order_item_id: mean 1.20 -> 1.20, std 0.70 -> 0.70
- price: mean 120.65 -> 120.67, std 184.11 -> 184.08
- freight_value: mean 20.03 -> 20.02, std 15.84 -> 15.82
- payment_sequential: mean 1.09 -> 1.09, std 0.73 -> 0.73
- payment_installments: mean 2.94 -> 2.94, std 2.78 -> 2.78
- payment_value: mean 172.74 -> 172.73, std 267.78 -> 267.77
- review_score: mean 4.02 -> 4.00, std 1.40 -> 1.41
- product_name_lenght: mean 48.77 -> 48.71, std 10.03 -> 10.01
- product_description_lenght: mean 785.97 -> 785.14, std 652.58 -> 648.59
- product_photos_qty: mean 2.21 -> 2.21, std 1.72 -> 1.71

New Method Found: None