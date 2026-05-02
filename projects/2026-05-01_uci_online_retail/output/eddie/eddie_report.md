# Eddie EDA & Business Report - UCI Online Retail

Rows: 1,067,371
Columns: 17
Input: C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\dana\dana_output.csv

## BUSINESS_EDA_FRAME

Business question: What happened across country, time, and product sales, and what customer/product grains are needed for RFM, market basket, CLV, churn, and inventory decisions?
Decision owner: Head of E-commerce / Retail Operations / Marketing Analytics
Primary KPI: valid sales revenue = Quantity * Price for rows where Quantity > 0 and Price > 0
Strongest evidence:
- Total valid-sales revenue: 20,972,978.64
- United Kingdom revenue share: 85.2%
- Peak sales month: 2011-11
- Peak sales hour: 12:00
- Top 20 product revenue share: 13.4%
- Known-customer repeat rate: 72.4%
Causality status: descriptive only; no causal claim.
Temporal/leakage risk: InvoiceDate exists, so churn/CLV must use cutoff dates and exclude future transactions.
Imbalance/skew risk: country and revenue are highly skewed; UK dominates and revenue has large retail extremes.
Validation strategy: use time-based validation for forecasting/churn/CLV; use customer-level split after aggregation, not transaction-row random split.

## Descriptive Findings

- Country: United Kingdom is the dominant revenue country; see `country_sales_summary.csv`.
- Time: sales concentrate in business hours; see `hourly_sales_summary.csv`.
- Product: revenue is concentrated in top SKUs; see `top_products_summary.csv`.
- Returns/cancellations: 22,951 rows flagged by negative quantity or cancellation invoice.
- Anonymous customers: 243,007 rows do not have a known customer identity.

## Behavioral Readiness

- RFM is ready only for known customers. Output: `rfm_customer_table.csv` (5,878 customers).
- Market basket analysis should use invoice-level valid sales and exclude returns/cancellations.
- CLV and churn must use customer-level aggregates from `rfm_customer_table.csv`, not raw transaction rows.

## Predictive/Decision Readiness

- CLV target candidate: customer-level monetary value over a fixed future window.
- Churn target candidate: no repeat purchase after a defined cutoff/horizon.
- Inventory target candidate: product-month demand from valid sales quantity.
- Mo must not train churn/CLV directly on `eddie_output.csv` transaction rows.

## Diagnostic Sample

- Sample rows for diagnostics: 100,000
- Quantity/revenue correlation: 0.942
- Price/revenue correlation: 0.219
- Full KMeans/silhouette on 1M+ rows was intentionally skipped; use sampled or aggregated tables only.

## PIPELINE_SPEC

problem_type        : multi-grain retail analytics
target_column       : none at transaction grain
transaction_table   : eddie_output.csv
customer_table      : rfm_customer_table.csv
product_time_table  : monthly_sales_summary.csv plus top_products_summary.csv
n_rows              : 1,067,371
n_features          : 17
recommended_model   : descriptive tables first; KMeans for RFM segments; supervised churn/CLV only after Finn creates customer-level targets
key_features        : country, invoice_month, invoice_hour, stock code, quantity, price, revenue, is_return
preprocessing       : exclude returns for sales modeling; exclude UNKNOWN_CUSTOMER for customer modeling; time cutoff before target creation
grain_contract      : transaction for descriptive, invoice for basket, customer for RFM/CLV/churn, product-month for inventory
data_quality_issues : UNKNOWN_CUSTOMER rows=243,007; returns/cancellations=22,951; retail extremes are flagged not dropped
finn_instructions   : build customer-level features from rfm_customer_table.csv for churn/CLV; build invoice-item matrix for basket; build product-month table for inventory
mo_instructions     : fail if target/grain is missing; do not train on transaction rows for churn or CLV

## INSIGHT_QUALITY

Criteria Met: 3/4
1. Business KPI defined: PASS
2. Grain-specific downstream plan: PASS
3. Actionable country/time/product/customer patterns: PASS
4. Causal claim: NOT ATTEMPTED

Verdict: SUFFICIENT for Finn/Iris handoff.
