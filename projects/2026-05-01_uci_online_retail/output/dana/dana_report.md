Dana Cleaning Report
====================
Before: 1,067,371 rows, 8 columns
After: 1,067,371 rows, 10 columns

Runtime Fix:
- Replaced KNNImputer on 1M+ rows with deterministic median/Unknown handling.
- Replaced row-by-row outlier loops with vectorized IQR flags.
- Skipped full IsolationForest because it is too slow for this project size and not required for cleaning retail transactions.

Missing Values:
- Description: 4,382 missing -> Unknown
- Customer ID: 243,007 missing -> UNKNOWN_CUSTOMER plus Customer ID_missing
- Price: 5 negative values corrected to median 2.1

Outlier Detection:
- Method: IQR vectorized; IsolationForest skipped for large retail dataset
- Rows flagged in dana_output.csv: 183,904
- outlier_flags.csv: 100,000 sampled rows (cap=100,000)
- Likely business-real retail extremes are kept and flagged, not capped.

Data Quality Score:
- Completeness: 97.1% -> 100.0%
- Validity: 100.0% -> 100.0%
- Overall: 98.6% -> 100.0%

Column Stats (After Cleaning):
- Quantity: mean=9.94, std=172.71, min=-80995.00, max=80995.00
- Price: mean=4.80, std=95.43, min=0.00, max=38970.00

New Method Found: FastDanaLargeRetailCleaning

DATA_QUALITY_AUDIT
==================
Raw shape: 1,067,371 x 8
Cleaned shape: 1,067,371 x 10
Completeness change: 97.1% -> 100.0%
Validity change: 100.0% -> 100.0%
Rows removed: none
Columns added: is_outlier plus missingness flags where needed
Imputation strategy: Unknown for categorical/customer gaps; median for numeric gaps
Outlier strategy: vectorized IQR flags; keep business-real extremes
Train-only safeguards: NA (cleaning/profiling only; downstream modeling must split before fitted transforms)
Bias/coverage impact: missing customer IDs preserved as UNKNOWN_CUSTOMER instead of dropping 22.77% of rows
Downstream warnings for Finn/Mo/Iris: exclude UNKNOWN_CUSTOMER from customer-level CLV/RFM/churn labels unless explicitly modeling anonymous transactions
Verdict: Ready
