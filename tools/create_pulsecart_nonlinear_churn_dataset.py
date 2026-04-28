from __future__ import annotations

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd


SEED = 4282026
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "2026-04-28_pulsecart_customer_behavior"
INPUT = PROJECT / "input"
ANSWER = PROJECT / "answer_key"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def add_missing(series: pd.Series, pct: float, rng: np.random.Generator) -> pd.Series:
    out = series.copy()
    n = int(round(len(out) * pct))
    idx = rng.choice(out.index.to_numpy(), n, replace=False)
    out.loc[idx] = np.nan
    return out


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    n_base = 2200

    customer_id = [f"PC{100000+i}" for i in range(n_base)]
    region = rng.choice(["metro", "suburban", "rural"], n_base, p=[0.48, 0.34, 0.18])
    plan_type = rng.choice(["basic", "plus", "premium"], n_base, p=[0.47, 0.36, 0.17])
    acquisition_channel = rng.choice(
        ["organic", "paid_search", "social", "referral", "affiliate"],
        n_base,
        p=[0.28, 0.26, 0.19, 0.17, 0.10],
    )
    age = np.clip(rng.normal(38, 11, n_base).round(), 18, 78).astype(int)
    account_tenure_days = rng.gamma(2.3, 120, n_base).round().astype(int) + 1
    avg_order_value = np.exp(rng.normal(3.95, 0.42, n_base)).round(2)
    orders_last_90d = np.clip(rng.poisson(7.5, n_base) + (plan_type == "premium") * 2, 0, 34)
    app_sessions_last_30d = np.clip(
        rng.poisson(9, n_base) + (orders_last_90d / 3).round().astype(int), 0, 55
    )
    support_tickets_90d = rng.poisson(0.8, n_base) + (rng.random(n_base) < 0.08).astype(int)
    late_delivery_rate = np.clip(
        rng.beta(1.5, 8.0, n_base)
        + (region == "rural") * 0.05
        + (support_tickets_90d > 2) * 0.08,
        0,
        0.92,
    )
    avg_delivery_delay_hours = np.clip(
        rng.gamma(2.0, 6.5, n_base) + late_delivery_rate * 38 + (region == "rural") * 4,
        0,
        96,
    ).round(2)
    discount_ratio = np.clip(
        rng.beta(2.3, 7.5, n_base) + (acquisition_channel == "affiliate") * 0.08,
        0,
        0.72,
    ).round(3)
    product_variety_score = np.clip(
        rng.normal(55, 18, n_base) + (orders_last_90d * 1.2) - (support_tickets_90d * 2.0),
        1,
        100,
    ).round(1)
    return_rate_90d = np.clip(
        rng.beta(1.2, 16, n_base) + support_tickets_90d * 0.018,
        0,
        0.65,
    ).round(3)
    competitor_price_index = np.clip(rng.normal(1.0, 0.12, n_base), 0.68, 1.48).round(3)
    days_since_last_order = np.clip(
        rng.gamma(2.2, 10.0, n_base) + np.maximum(0, 10 - orders_last_90d) * 3.2,
        0,
        120,
    ).round(1)

    # Planted nonlinear relationships:
    # 1. Delay effect is weak until around 30 hours, then rises sharply.
    # 2. Discount has a U shape: no discount and very high discount both raise churn.
    # 3. Activity is seasonal/wavy, so Pearson correlation understates signal.
    delay_risk = sigmoid((avg_delivery_delay_hours - 31) / 6.2)
    discount_u = ((discount_ratio - 0.24) ** 2) * 5.2
    inactivity_wave = 0.55 * np.sin((days_since_last_order - 12) / 7.0)
    support_delay_interaction = (support_tickets_90d >= 2) * sigmoid((avg_delivery_delay_hours - 22) / 5.5)
    low_variety_risk = sigmoid((42 - product_variety_score) / 7.5)
    tenure_protection = sigmoid((account_tenure_days - 260) / 90)

    score = (
        -2.35
        + 1.45 * delay_risk
        + 0.85 * support_delay_interaction
        + 0.75 * low_variety_risk
        + 0.70 * discount_u
        + 0.55 * inactivity_wave
        + 0.45 * return_rate_90d
        + 0.38 * (competitor_price_index < 0.90)
        - 0.85 * tenure_protection
        - 0.25 * (plan_type == "premium")
        + 0.22 * (acquisition_channel == "paid_search")
    )
    churn_prob = sigmoid(score)
    account_status_30d = (rng.random(n_base) < churn_prob).astype(int)

    clean = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": pd.Timestamp("2023-01-01")
            + pd.to_timedelta(730 - np.minimum(account_tenure_days, 730), unit="D"),
            "region": region,
            "plan_type": plan_type,
            "acquisition_channel": acquisition_channel,
            "age": age,
            "account_tenure_days": account_tenure_days,
            "avg_order_value": avg_order_value,
            "orders_last_90d": orders_last_90d,
            "app_sessions_last_30d": app_sessions_last_30d,
            "support_tickets_90d": support_tickets_90d,
            "late_delivery_rate": late_delivery_rate.round(3),
            "avg_delivery_delay_hours": avg_delivery_delay_hours,
            "discount_ratio": discount_ratio,
            "product_variety_score": product_variety_score,
            "return_rate_90d": return_rate_90d,
            "competitor_price_index": competitor_price_index,
            "days_since_last_order": days_since_last_order,
            "account_status_30d": account_status_30d,
        }
    )
    clean["signup_date"] = clean["signup_date"].dt.strftime("%Y-%m-%d")

    raw = clean.copy()
    raw["signup_date"] = raw["signup_date"].astype(str)

    # Post-outcome leakage columns that must be dropped before modeling.
    raw["post_period_refund_flag"] = np.where(
        raw["account_status_30d"].eq(1),
        rng.choice(["Y", "N"], n_base, p=[0.64, 0.36]),
        rng.choice(["Y", "N"], n_base, p=[0.08, 0.92]),
    )
    raw["account_note_post_period"] = np.where(
        raw["account_status_30d"].eq(1),
        rng.choice(["late delivery", "price", "support", "variety", ""], n_base),
        "",
    )

    # Dirty categorical values.
    raw.loc[rng.choice(raw.index, 64, replace=False), "region"] = rng.choice(
        ["Metro", "SUBURB", "rurral", "unknown"], 64
    )
    raw.loc[rng.choice(raw.index, 52, replace=False), "plan_type"] = rng.choice(
        ["Basic", "PLUS", "prem", "?"], 52
    )

    # Missing values.
    for col, pct in [
        ("age", 0.035),
        ("avg_order_value", 0.025),
        ("late_delivery_rate", 0.030),
        ("product_variety_score", 0.020),
        ("competitor_price_index", 0.018),
    ]:
        raw[col] = add_missing(raw[col], pct, rng)

    # Invalid and outlier values.
    idx = rng.choice(raw.index, 22, replace=False)
    raw.loc[idx[:11], "age"] = rng.choice([8, 12, 97, 104], 11)
    raw.loc[idx[11:], "account_tenure_days"] = rng.choice([-15, -1, 0], 11)
    idx_delay = rng.choice(raw.index, 31, replace=False)
    raw.loc[idx_delay, "avg_delivery_delay_hours"] = rng.choice([145, 180, 220, 360], 31)
    idx_aov = rng.choice(raw.index, 18, replace=False)
    raw.loc[idx_aov, "avg_order_value"] = raw.loc[idx_aov, "avg_order_value"] * rng.choice([8, 12, 20], 18)
    idx_disc = rng.choice(raw.index, 16, replace=False)
    raw.loc[idx_disc, "discount_ratio"] = rng.choice([1.15, 1.4, -0.2], 16)
    idx_returns = rng.choice(raw.index, 14, replace=False)
    raw.loc[idx_returns, "return_rate_90d"] = rng.choice([1.2, 1.6, -0.15], 14)

    # Duplicate rows.
    duplicate_rows = raw.sample(37, random_state=SEED + 1)
    raw = pd.concat([raw, duplicate_rows], ignore_index=True)

    # Shuffle rows and add a few whitespace issues.
    raw = raw.sample(frac=1.0, random_state=SEED + 2).reset_index(drop=True)
    whitespace_idx = rng.choice(raw.index, 48, replace=False)
    raw.loc[whitespace_idx, "customer_id"] = raw.loc[whitespace_idx, "customer_id"].astype(str) + " "

    expected_clean = clean.copy()
    expected_clean["delay_risk_band"] = pd.cut(
        expected_clean["avg_delivery_delay_hours"],
        bins=[-0.1, 18, 30, 48, 999],
        labels=["low", "watch", "high", "critical"],
    ).astype(str)
    expected_clean["discount_band"] = pd.cut(
        expected_clean["discount_ratio"],
        bins=[-0.1, 0.10, 0.35, 0.999],
        labels=["low", "healthy", "high"],
    ).astype(str)

    stats = {
        "seed": SEED,
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "expected_clean_rows": int(expected_clean.shape[0]),
        "expected_clean_columns": int(expected_clean.shape[1]),
        "duplicates": int(raw["customer_id"].astype(str).str.strip().duplicated(keep="first").sum()),
        "target_rate": float(expected_clean["account_status_30d"].mean()),
        "outlier_delay_count": int((raw["avg_delivery_delay_hours"] > 120).sum()),
        "invalid_age_count": int(((raw["age"] < 18) | (raw["age"] > 90)).sum(skipna=True)),
        "invalid_tenure_count": int((raw["account_tenure_days"] <= 0).sum()),
        "invalid_discount_count": int(((raw["discount_ratio"] < 0) | (raw["discount_ratio"] > 1)).sum()),
        "invalid_return_rate_count": int(((raw["return_rate_90d"] < 0) | (raw["return_rate_90d"] > 1)).sum()),
    }
    return raw, expected_clean, stats


def write_docs(stats: dict[str, object]) -> None:
    data_dictionary = f"""# PulseCart Dataset

Business context: PulseCart is a fictional grocery delivery subscription business.
The dataset contains customer activity, service experience, pricing, product usage, and follow-up status fields.
Use the data to decide what business question is most useful, what column should be modeled if any, and what preparation is required.

Files:
- `pulsecart_raw.csv`: raw customer-level dataset.
- `data_dictionary.md`: schema notes.

Columns:
- `customer_id`: customer key.
- `signup_date`: customer signup date.
- `region`: customer region.
- `plan_type`: subscription plan.
- `acquisition_channel`: acquisition source.
- `age`: customer age.
- `account_tenure_days`: days since signup.
- `avg_order_value`: average basket value.
- `orders_last_90d`: order count in last 90 days.
- `app_sessions_last_30d`: recent app usage.
- `support_tickets_90d`: support tickets in last 90 days.
- `late_delivery_rate`: share of recent deliveries that were late.
- `avg_delivery_delay_hours`: average delay hours.
- `discount_ratio`: discount / gross value.
- `product_variety_score`: score from 1 to 100.
- `return_rate_90d`: returned items / ordered items.
- `competitor_price_index`: competitor price divided by PulseCart price.
- `days_since_last_order`: recency.
- `account_status_30d`: 30-day account follow-up status flag.
- `post_period_refund_flag`: refund flag from the CRM extract.
- `account_note_post_period`: account note from the CRM extract.

Raw shape: {stats['raw_rows']} rows x {stats['raw_columns']} columns.
"""
    answer_key = f"""# PulseCart Answer Key

Use this file to grade the agents. Do not include this folder in the agent input context during a blind test.

## Dataset Identity

This is a synthetic but business-realistic churn dataset for a grocery delivery subscription company, PulseCart.
The business question is: which customers are likely to churn in the next 30 days, and what actions reduce churn?

## Raw Data Truth

- Raw file: `input/pulsecart_raw.csv`
- Raw rows: {stats['raw_rows']}
- Raw columns: {stats['raw_columns']}
- Duplicate customer rows planted: {stats['duplicates']}
- Target: `account_status_30d`
- Target positive rate in clean truth: {stats['target_rate']:.3f}
- Leakage columns: `post_period_refund_flag`, `account_note_post_period`

## Expected Cleaning

Expected cleaned rows: {stats['expected_clean_rows']}
Expected cleaned columns: {stats['expected_clean_columns']}

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
- `avg_delivery_delay_hours > 120`: {stats['outlier_delay_count']} rows.
- invalid `age` outside 18-90: {stats['invalid_age_count']} rows.
- invalid `account_tenure_days <= 0`: {stats['invalid_tenure_count']} rows.
- invalid `discount_ratio` outside 0-1: {stats['invalid_discount_count']} rows.
- invalid `return_rate_90d` outside 0-1: {stats['invalid_return_rate_count']} rows.

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
- Produces a cleaned dataset with about {stats['expected_clean_rows']} unique customers.
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
"""
    iris_expected = """# Expected Iris Report

Iris should frame this as a customer retention problem for PulseCart, a grocery delivery subscription business.

Minimum expected insight quality:
- The dataset supports predicting `account_status_30d`, a binary churn-risk follow-up label.
- Delivery delay is not merely correlated linearly with churn; risk jumps after a delay threshold around 30 hours.
- Customers with both support tickets and delivery delay are a priority intervention group.
- Discounting has a nonlinear/U-shaped pattern; high discount users are not automatically loyal.
- Product variety below a practical threshold is a churn risk.
- The strongest business action is operational reliability plus targeted retention, not blanket discounts.

A strong final recommendation:
Build a weekly churn-risk workflow. Route high-delay/high-ticket customers to service recovery, offer moderate targeted incentives only after operational issues are fixed, and improve product variety for exposed segments.
"""

    INPUT.mkdir(parents=True, exist_ok=True)
    ANSWER.mkdir(parents=True, exist_ok=True)
    (INPUT / "data_dictionary.md").write_text(data_dictionary, encoding="utf-8")
    (ANSWER / "answer_key.md").write_text(answer_key, encoding="utf-8")
    (ANSWER / "iris_expected_report.md").write_text(iris_expected, encoding="utf-8")


def main() -> None:
    raw, expected_clean, stats = build_dataset()
    INPUT.mkdir(parents=True, exist_ok=True)
    ANSWER.mkdir(parents=True, exist_ok=True)
    raw.to_csv(INPUT / "pulsecart_raw.csv", index=False)
    expected_clean.to_csv(ANSWER / "expected_clean_output.csv", index=False)
    write_docs(stats)
    print(f"project={PROJECT}")
    print(f"raw_rows={stats['raw_rows']} raw_cols={stats['raw_columns']}")
    print(f"clean_rows={stats['expected_clean_rows']} clean_cols={stats['expected_clean_columns']}")
    print(f"target_rate={stats['target_rate']:.3f}")


if __name__ == "__main__":
    main()
