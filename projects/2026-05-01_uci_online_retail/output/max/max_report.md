# Max Inventory Optimization Report

Generated: 2026-05-02 22:27:21

## Inputs
- Top products: `C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\eddie\top_products_summary.csv`
- Monthly sales summary checked: `C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\eddie\monthly_sales_summary.csv`
- Product-level monthly quantity source: `C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\eddie\eddie_output.csv`

## Method
- Top 10 products are keyed by `Description` from `top_products_summary.csv`.
- Product-month demand is aggregated from `eddie_output.csv` using `Description`, `InvoiceDate` or `invoice_month`, and `Quantity`.
- Forecast horizon is 6 monthly periods per product.
- Forecast model is Prophet when it fits successfully; otherwise the script records a moving-average fallback for that product.
- Safety stock formula: `historical monthly mean + 1.5 * historical monthly std`.

## Forecast Accuracy
| Description | Prophet MAPE | Naive MAPE | Beats Naive |
|---|---:|---:|---|
| WHITE HANGING HEART T-LIGHT HOLDER | 124.53 | 282.72 | True |
| ASSORTED COLOUR BIRD ORNAMENT | 136.82 | 128.73 | False |
| PAPER CRAFT , LITTLE BIRDIE | 100.0 | 100.0 | False |
| JUMBO BAG RED RETROSPOT | 320.11 | 83.26 | False |
| MEDIUM CERAMIC TOP STORAGE JAR | 12580.77 | 100.0 | False |
| SMALL POPCORN HOLDER | N/A | N/A | False |
| ASSORTED COLOURS SILK FAN | 291.6 | 195.01 | False |
| VICTORIAN GLASS HANGING T-LIGHT | 178.02 | 94.68 | False |
| RED  HARMONICA IN BOX | 290.11 | 104.19 | False |
| STRAWBERRY CERAMIC TRINKET BOX | 151.67 | 524.65 | True |
- Beats naive count: 2 of 10

## Output Summary
- Products forecasted: 10
- Forecast rows: 60
- Next-month forecast quantity: 148,665
- Next-month recommended stock: 237,464

## Top Products
| Rank | Description | StockCode | Historical Mean | Historical Std | Safety Stock | Model |
|---:|---|---|---:|---:|---:|---|
| 1 | WHITE HANGING HEART T-LIGHT HOLDER | 85123A | 3,867.32 | 1,568.82 | 6,220.55 | prophet |
| 2 | ASSORTED COLOUR BIRD ORNAMENT | 84879 | 3,272.36 | 1,849.08 | 6,045.97 | prophet |
| 3 | PAPER CRAFT , LITTLE BIRDIE | 23843 | 3,239.80 | 15,871.71 | 27,047.37 | prophet |
| 4 | JUMBO BAG RED RETROSPOT | 85099B | 3,171.16 | 2,147.88 | 6,392.98 | prophet |
| 5 | MEDIUM CERAMIC TOP STORAGE JAR | 23166 | 3,121.32 | 14,514.10 | 24,892.47 | prophet |
| 6 | SMALL POPCORN HOLDER | 22197 | 1,997.92 | 2,246.89 | 5,368.26 | prophet |
| 7 | ASSORTED COLOURS SILK FAN | 15036 | 1,774.60 | 1,388.74 | 3,857.71 | prophet |
| 8 | VICTORIAN GLASS HANGING T-LIGHT | 22178 | 1,629.92 | 924.99 | 3,017.41 | prophet |
| 9 | RED  HARMONICA IN BOX | 21915 | 1,527.64 | 1,105.97 | 3,186.60 | prophet |
| 10 | STRAWBERRY CERAMIC TRINKET BOX | 21232 | 1,504.40 | 843.28 | 2,769.32 | prophet |

## Next-Month Inventory Recommendation
| Description | Forecast Quantity | Safety Stock | Recommended Stock |
|---|---:|---:|---:|
| MEDIUM CERAMIC TOP STORAGE JAR | 129,218 | 24,892 | 154,111 |
| PAPER CRAFT , LITTLE BIRDIE | 0 | 27,047 | 27,047 |
| JUMBO BAG RED RETROSPOT | 3,974 | 6,393 | 10,367 |
| WHITE HANGING HEART T-LIGHT HOLDER | 3,721 | 6,221 | 9,942 |
| ASSORTED COLOUR BIRD ORNAMENT | 3,411 | 6,046 | 9,457 |
| SMALL POPCORN HOLDER | 3,464 | 5,368 | 8,832 |
| VICTORIAN GLASS HANGING T-LIGHT | 2,807 | 3,017 | 5,825 |
| ASSORTED COLOURS SILK FAN | 1,368 | 3,858 | 5,226 |
| RED  HARMONICA IN BOX | 702 | 3,187 | 3,888 |
| STRAWBERRY CERAMIC TRINKET BOX | 0 | 2,769 | 2,769 |

## Data Notes
- Aggregated product-month rows: 198
- Historical range: 2009-12-01 to 2011-12-01
- Negative quantities and rows marked as returns are excluded for inventory demand forecasting.