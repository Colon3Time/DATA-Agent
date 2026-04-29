# Self-Improvement Report — Iris
**Date:** 2026-04-29 22:12

## Methods Used
- **Framework:** Business Insight Engine with SHAP interpretation patterns
- **Statistical Testing:** Chi-square for categorical feature significance
- **Customer Segmentation:** Propensity-based binning using model scores

## Key Observations
- Input data had 1 columns: ['age;"job";"marital";"education";"default";"housing";"loan";"contact";"month";"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"']
- Detected true columns: ['age;"job";"marital";"education";"default";"housing";"loan";"contact";"month";"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"']
- Detected prediction columns: []
- Detected probability columns: []
- Detected feature importance columns: []
- Detected cluster columns: []

## Improvements for Next Time
- Ensure input data includes y_true and y_pred columns for accuracy metrics
- Request probability scores for customer propensity segmentation
- Include feature importance for actionable marketing insights

## Knowledge Base Updates
- Add pattern: When input lacks y columns, fallback to descriptive statistics
- Add pattern: Always check for missing columns and adapt insights accordingly