DATASET_PROFILE
===============
dataset_name  : Online Retail II
source        : UCI Machine Learning Repository
source_url    : https://archive.ics.uci.edu/dataset/502/online+retail+ii
doi           : 10.24432/C5CG6D
license       : CC BY 4.0
local_file    : projects/2026-04-30_marketing_dl_dataset/input/online_retail_II.xlsx
download_note : Download verified from UCI archive URL, but shell write to project input is blocked by Windows ACL DENY in this workspace. Verified file copy is staged at C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_II.xlsx.

rows          : 1,067,371
cols          : 8
sheets        : Year 2009-2010, Year 2010-2011
date_range    : 2009-12-01 to 2011-12-09

dtypes        : numeric=2, categorical=5, datetime=1
columns       : Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
missing       : {"Description": 4382, "Customer ID": 243007}

target_column : none
problem_type  : unsupervised / time-series / regression-ready
recommended_targets:
- customer_value_regression: future revenue or customer lifetime value derived from Quantity * Price
- churn_classification: inactive customer flag derived from recency windows
- basket_clustering: customer or invoice segmentation from RFM and product mix

key_quality_notes:
- Customer ID is missing in 243,007 rows, so customer-level modeling must either exclude anonymous rows or keep them only for product/invoice aggregates.
- Description is missing in 4,382 rows, but StockCode remains present and can preserve item-level analysis.
- Cancellation rows are present: 19,494 Invoice values start with C.
- Country has 43 distinct values; the dataset is dominated by UK retail transactions.
- InvoiceDate is stored in Excel serial-date form in the raw workbook and should be parsed to datetime by pandas/openpyxl or equivalent Excel reader.

scout_gate:
- rows_gt_1000: PASS
- real_dataset_not_shortlist: PASS
- input_folder_checked: PASS
- xlsx_reader_required: pandas.read_excel with openpyxl recommended
- dataset_risk_register: PASS

DATASET_RISK_REGISTER
=====================
1. Excel dependency risk: pandas requires openpyxl or another Excel engine to read the .xlsx directly.
2. Memory/runtime risk: full workbook has 1.07M rows across two sheets; downstream agents should stream, sample, or cache to CSV/parquet after first read.
3. Target leakage risk: generated CLV/churn targets must use chronological splits so future purchases do not leak into training features.
4. Missing customer risk: Customer ID missingness affects customer-level tasks and must be handled before RFM/CLV/churn modeling.
5. Return/cancellation risk: cancellation invoices and negative quantities/prices need explicit business rules before revenue modeling.
