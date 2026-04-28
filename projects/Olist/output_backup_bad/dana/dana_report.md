Dana Cleaning Report
==================================================
Date: 2026-04-28 02:20

Before: 32951 rows, 10 columns
After: 32951 rows, 11 columns

Missing Values:
------------------------------
  - product_category_name: 610 (1.85%) remaining
  - product_category_name_english: 623 (1.89%) remaining

Outlier Detection:
------------------------------
  Method: Isolation Forest + IQR (1.5x)
  Likely Error (fixed): None
  Likely Real (flagged): 11073 rows
    - row 279: product_name_lenght=19.0 (is_outlier=1)
    - row 397: product_name_lenght=18.0 (is_outlier=1)
    - row 447: product_name_lenght=17.0 (is_outlier=1)
    - row 774: product_name_lenght=18.0 (is_outlier=1)
    - row 811: product_name_lenght=15.0 (is_outlier=1)
    ... and 11068 more
  Uncertain (flagged): 3 rows
    - row 487: Isolation Forest anomaly (multivariate)
    - row 24374: Isolation Forest anomaly (multivariate)
    - row 29654: Isolation Forest anomaly (multivariate)

Data Quality Score:
------------------------------
  Completeness: 99.07% -> 99.66%
  Validity: 90.68% -> 100.00%
  Overall: 94.87% -> 99.83%

Column Stats (Before -> After):
------------------------------
  product_name_lenght: mean 48.48->48.49, std 10.25->10.17
  product_description_lenght: mean 771.50->770.63, std 635.12->630.68
  product_photos_qty: mean 2.19->2.19, std 1.74->1.73
  product_weight_g: mean 2276.47->2276.67, std 4282.04->4281.84
  product_length_cm: mean 30.82->30.82, std 16.91->16.91

New Method Found: None