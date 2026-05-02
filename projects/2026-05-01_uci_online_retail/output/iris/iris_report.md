# Iris Segmentation and Basket Report

Input customer table: C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\finn\engineered_data.csv

## RFM Segmentation

Customers segmented: 5,878

| segment   |   customers |   avg_recency |   avg_frequency |   avg_monetary |          revenue |
|:----------|------------:|--------------:|----------------:|---------------:|-----------------:|
| Champions |        1294 |       25.7836 |        17.966   |       9894.42  |      1.28034e+07 |
| Loyal     |        1354 |       95.9446 |         5.5901  |       2315.83  |      3.13563e+06 |
| Potential |        1453 |      201.773  |         2.70406 |        868.803 |      1.26237e+06 |
| At Risk   |         972 |      319.404  |         1.41255 |        393.382 | 382367           |
| Lost      |         805 |      517.415  |         1.0559  |        198.365 | 159684           |

## Basket Readiness

Invoice basket table: C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\finn\invoice_basket_features.csv
Invoices available: 40,083

Market basket analysis is ready at invoice grain. For association rules, use valid sales line items from Eddie/Finn and exclude returns.

## Business Use

- Champions/Loyal: retention and premium bundles
- Potential: cross-sell and replenishment campaigns
- At Risk/Lost: win-back offers with cost cap

IRIS_DECISION_FRAME
===================
rfm_output: rfm_segments.csv
basket_output: basket_summary.csv
next_agent: Vera for charts, Rex for final executive story
