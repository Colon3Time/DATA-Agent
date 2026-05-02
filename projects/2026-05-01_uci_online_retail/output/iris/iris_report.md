# Iris Segmentation and Basket Report

Input: projects\2026-05-01_uci_online_retail\output\finn\engineered_data.csv

## RFM Segment Recommendations

Customers segmented: 5,878

| segment   |   customers |   avg_recency |   avg_frequency |   avg_monetary |          revenue |
|:----------|------------:|--------------:|----------------:|---------------:|-----------------:|
| Champions |        1292 |       19.726  |        17.13    |        9355.6  |      1.20874e+07 |
| Loyal     |         978 |      170.54   |         7.9182  |        3083.89 |      3.01604e+06 |
| Potential |        1097 |       52.9225 |         2.78304 |        1197.45 |      1.3136e+06  |
| Lost      |        2013 |      367.718  |         1.2469  |         400.76 | 806730           |
| At Risk   |         498 |      387.315  |         3.07229 |        1043.42 | 519623           |

## Segment Actions

- Champions: protect with early access, premium bundles, and service recovery priority.
- Loyal: replenish and cross-sell adjacent categories.
- Potential: move to second and third purchase with limited-time bundles.
- At Risk: win-back only where expected margin covers incentive cost.
- Lost: suppress broad discounts unless reactivation economics are validated.

## Association Rules

Association rules generated from 33,937 invoices x 200 products.

Top rules:

| antecedents                                                      | consequents                                                      |   support |   confidence |    lift |
|:-----------------------------------------------------------------|:-----------------------------------------------------------------|----------:|-------------:|--------:|
| PINK REGENCY TEACUP AND SAUCER                                   | GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER | 0.0213926 |     0.708984 | 23.5198 |
| GREEN REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER | PINK REGENCY TEACUP AND SAUCER                                   | 0.0213926 |     0.709677 | 23.5198 |
| PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER  | GREEN REGENCY TEACUP AND SAUCER                                  | 0.0213926 |     0.902985 | 22.8691 |
| GREEN REGENCY TEACUP AND SAUCER                                  | PINK REGENCY TEACUP AND SAUCER, ROSES REGENCY TEACUP AND SAUCER  | 0.0213926 |     0.541791 | 22.8691 |
| PINK REGENCY TEACUP AND SAUCER                                   | GREEN REGENCY TEACUP AND SAUCER                                  | 0.0251643 |     0.833984 | 21.1216 |

BUSINESS_DECISION_BRIEF
=======================
business_lever: segmented CRM and basket bundling
kpi: repeat purchase rate, margin per customer, basket size
owner: CRM / merchandising
validation_plan: holdout campaign test by RFM segment and bundle rule
confidence: medium, depends on campaign economics and OOT validation
