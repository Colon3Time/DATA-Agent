# Dana Cleaning Report
====================
Date: 2026-04-25 19:38
Input: C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test3\input\retail_sales_600.csv

Before: 600 rows, 12 columns
After:  600 rows, 12 columns
Total Missing: 72 -> 36

## Data Types Changes
Changes: 1
- date: converted to datetime

## Missing Values Handling
Handled: 2 columns
- unit_price: missing 3.0% -> median fill (low missing)
- quantity: missing 3.0% -> median fill (low missing)

## Outliers Handling
Handled: 2 columns
- unit_price: IQR capped 70 outliers (IQR*1.5)
- total_amount: IQR capped 51 outliers (IQR*1.5)

## Column Summary
- date: datetime64[us], 345 unique, 18 missing
- product: str, 10 unique, 0 missing
- category: str, 4 unique, 0 missing
- unit_price: float64, 478 unique, 0 missing
- quantity: float64, 14 unique, 0 missing
- discount_pct: float64, 5 unique, 0 missing
- total_amount: float64, 541 unique, 0 missing
- region: str, 5 unique, 18 missing
- sales_channel: str, 3 unique, 0 missing
- sales_rep: str, 5 unique, 0 missing
- customer_segment: str, 3 unique, 0 missing
- return_flag: int64, 2 unique, 0 missing

## Data Quality Score
Before: 99.0%
After:  99.5%