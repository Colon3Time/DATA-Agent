# PulseCart Answer Key

Use this file to grade the agents. Do not include this folder in the agent input context during a blind test.

## Dataset Identity

This is a synthetic but business-realistic churn dataset for a grocery delivery subscription company, PulseCart.
The business question is: which customers are likely to churn in the next 30 days, and what actions reduce churn?

## Raw Data Truth

- Raw file: `input/pulsecart_raw.csv`
- Raw rows: 2237
- Raw columns: 21
- Duplicate customer rows planted: 37
- Target: `account_status_30d`
- Target positive rate in clean truth: 0.132
- Leakage columns: `post_period_refund_flag`, `account_note_post_period`

## Expected Cleaning

Expected cleaned rows: 2200
Expected cleaned columns: 21

Required cleaning decisions:
- Trim `customer_id` whitespace and deduplicate customers.
- Normalize `region`: `Metro` -> `metro`, `SUBURB` -> `suburban`, `rurral` -> `rural`, unknown values should become missing/unknown category.
- Normalize `plan_type`: `Basic` -> `basic`, `PLUS` -> `plus`, `prem` -> `premium`, `?` should become missing/unknown category.
- Parse `signup_date` as date.
- Keep `account_status_30d` as the label.
- Drop leakage columns before EDA-to-model handoff: `post_period_refund_flag`, `account_note_post_period`.
- Do not drop `customer_id` during cleaning, but exclude it from model features.

## Outlier And Invalid Values

Planted raw issues:
- `avg_delivery_delay_hours > 120`: 32 rows.
- invalid `age` outside 18-90: 11 rows.
- invalid `account_tenure_days <= 0`: 11 rows.
- invalid `discount_ratio` outside 0-1: 16 rows.
- invalid `return_rate_90d` outside 0-1: 14 rows.

Recommended handling:
- For invalid age/tenure/ratio values: set to missing, then impute numeric median within `plan_type` and `region` where possible.
- For delivery delay and order value outliers: cap by business/domain limits or 1st/99th percentile winsorization. Do not delete rows unless values are unrecoverable.
- Keep a flag column for capped delay if the agent creates one, because extreme delays are meaningful.

## Nonlinear Correlation Trap

The strongest planted signals are nonlinear:
- `avg_delivery_delay_hours` has a threshold/sigmoid effect: churn rises sharply after roughly 30 hours.
- `discount_ratio` is U-shaped: very low discount and very high discount are worse than moderate discount.
- `days_since_last_order` has a wavy/non-monotonic effect.
- `support_tickets_90d` interacts with delay: tickets become much more predictive when delivery delay is high.
- `product_variety_score` has a threshold effect: churn rises when the score falls below about 42.

Good Eddie/Finn behavior:
- Do not rely only on Pearson correlation.
- Use mutual information, tree-based feature importance, partial dependence/ALE, binned target-rate plots, Spearman/Kendall, and interaction checks.
- Create features such as `delay_over_30h`, `delay_risk_band`, `discount_distance_from_healthy`, `support_delay_interaction`, and `low_variety_flag`.

## Expected Model Framing

- Label: `account_status_30d`
- Problem type: binary classification
- Recommended models: gradient boosting / random forest / logistic regression baseline.
- Metrics: ROC-AUC, PR-AUC, recall at business precision, confusion matrix, calibration.
- Avoid leakage columns and post-outcome text.
- Class balance is moderately imbalanced, so do not judge only by accuracy.

## Expected Business Insights

High-quality Iris/Rex report should say:
- This is a retention and operations problem for a grocery delivery subscription business.
- Delivery reliability is the largest controllable driver. Churn risk accelerates after average delay crosses roughly 30 hours, especially with repeated support tickets.
- Discounting is not linearly good. Moderate discounts retain customers, but very high discounts correlate with fragile customers who churn anyway.
- Low product variety creates churn risk, especially for active customers who otherwise order often.
- Recent inactivity matters, but not as a simple straight-line Pearson relationship.
- Premium customers are somewhat more resilient, but service failures can still override plan loyalty.

Recommended actions:
- Trigger save campaigns for customers with high delay plus support tickets.
- Fix delivery operations for delay-risk regions before spending more on discounts.
- Use moderate, targeted discounts rather than blanket high discounting.
- Improve catalog variety for low-variety customer segments.
- Build a churn score used weekly by CRM/ops teams.

## Agent Scoring Rubric

Scout should pass if it identifies:
- Business domain: subscription grocery delivery / ecommerce retention.
- Target: `account_status_30d`.
- Problem type: classification.
- Leakage warning for post-period/refund/churn-reason fields.

Dana should pass if it:
- Produces a cleaned dataset with about 2200 unique customers.
- Preserves the target.
- Does not drop large numbers of rows.
- Handles invalid values and outliers with imputation/capping instead of blind deletion.

Eddie should pass if it:
- Finds nonlinear/threshold patterns.
- Does not conclude "no relationship" just because Pearson correlations are weak.
- Writes a complete PIPELINE_SPEC with `problem_type=classification`, `target_column=account_status_30d`, and leakage exclusions.

Finn should pass if it:
- Removes leakage and ID columns from features.
- Engineers nonlinear/interactions or selects models/features that can capture them.
- Keeps target separate from features.

Mo should pass if it:
- Compares classical ML models.
- Uses classification metrics beyond accuracy.
- Prefers a tree/boosting model or otherwise demonstrates nonlinear capability.

Iris should pass if it:
- Translates model/EDA results into retention, delivery, discount, and catalog actions.
- Explains what the business can do next, not only charts.

Rex should pass if it:
- Produces an executive summary with business framing, model result, top drivers, risk caveats, and concrete recommendations.
