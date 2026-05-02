import argparse
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
project_dir = output_dir.parents[1]

input_path = Path(args.input) if args.input else project_dir / "output" / "finn" / "engineered_data.csv"
if not input_path.exists():
    input_path = project_dir / "output" / "finn" / "finn_output.csv"
if not input_path.exists():
    raise FileNotFoundError(f"Iris requires Finn customer data, missing: {input_path}")

df = pd.read_csv(input_path)
print(f"[STATUS] Loaded input: {input_path} shape={df.shape}")

for col in ["recency_days", "frequency", "monetary"]:
    if col not in df.columns:
        raise ValueError(f"Iris expected Finn RFM column `{col}` in {input_path}")

rfm = df.copy()
rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]


def segment(row):
    if row["r_score"] >= 4 and row["f_score"] >= 4 and row["m_score"] >= 4:
        return "Champions"
    if row["f_score"] >= 4 and row["m_score"] >= 3:
        return "Loyal"
    if row["r_score"] >= 3 and row["rfm_score"] >= 8:
        return "Potential"
    if row["r_score"] <= 2 and row["f_score"] >= 3:
        return "At Risk"
    return "Lost"


rfm["segment"] = rfm.apply(segment, axis=1)
rfm.to_csv(output_dir / "rfm_segments.csv", index=False)
segment_summary = (
    rfm.groupby("segment")
    .agg(
        customers=("segment", "size"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        revenue=("monetary", "sum"),
    )
    .sort_values("revenue", ascending=False)
)
segment_summary.to_csv(output_dir / "rfm_segment_summary.csv")

association_note = ""
top_rules_md = "_No association rules produced._"
rules_path = output_dir / "association_rules.csv"
eddie_path = project_dir / "output" / "eddie" / "eddie_output.csv"
if eddie_path.exists():
    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        eddie_df = pd.read_csv(eddie_path, low_memory=False)
        required = ["Invoice", "Description", "Quantity", "Price"]
        missing = [c for c in required if c not in eddie_df.columns]
        if missing:
            raise ValueError(f"Missing market-basket columns: {missing}")
        work = eddie_df.copy()
        work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
        work["Price"] = pd.to_numeric(work["Price"], errors="coerce")
        work = work[(work["Quantity"] > 0) & (work["Price"] > 0)]
        top_skus = work["Description"].value_counts().head(200).index
        work = work[work["Description"].isin(top_skus)]
        basket = (
            work.groupby(["Invoice", "Description"])["Quantity"]
            .sum()
            .unstack(fill_value=0)
        )
        basket = (basket > 0).astype(int)
        freq = apriori(basket, min_support=0.02, use_colnames=True, max_len=3)
        rules = association_rules(freq, metric="lift", min_threshold=2.0)
        rules = rules.sort_values("lift", ascending=False).head(50).copy()
        for col in ["antecedents", "consequents"]:
            rules[col] = rules[col].apply(lambda x: ", ".join(sorted(map(str, x))))
        rules.to_csv(rules_path, index=False)
        top_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(5)
        top_rules_md = top_rules.to_markdown(index=False)
        association_note = f"Association rules generated from {basket.shape[0]:,} invoices x {basket.shape[1]:,} products."
    except Exception as exc:
        pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"]).to_csv(rules_path, index=False)
        association_note = f"Association rules could not be generated: {exc}"
else:
    pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"]).to_csv(rules_path, index=False)
    association_note = f"Missing Eddie transaction table: {eddie_path}"

report_segment = segment_summary.to_markdown()
report = f"""# Iris Segmentation and Basket Report

Input: {input_path}

## RFM Segment Recommendations

Customers segmented: {len(rfm):,}

{report_segment}

## Segment Actions

- Champions: protect with early access, premium bundles, and service recovery priority.
- Loyal: replenish and cross-sell adjacent categories.
- Potential: move to second and third purchase with limited-time bundles.
- At Risk: win-back only where expected margin covers incentive cost.
- Lost: suppress broad discounts unless reactivation economics are validated.

## Association Rules

{association_note}

Top rules:

{top_rules_md}

BUSINESS_DECISION_BRIEF
=======================
business_lever: segmented CRM and basket bundling
kpi: repeat purchase rate, margin per customer, basket size
owner: CRM / merchandising
validation_plan: holdout campaign test by RFM segment and bundle rule
confidence: medium, depends on campaign economics and OOT validation
"""
(output_dir / "iris_report.md").write_text(report, encoding="utf-8")

iris_output = segment_summary.reset_index()
iris_output.to_csv(output_dir / "iris_output.csv", index=False)

agent_report = f"""Agent Report - Iris
====================
Input: {input_path}
Output: iris_report.md, rfm_segments.csv, association_rules.csv
Notes: RFM segmentation and market-basket analysis completed.
"""
(output_dir.parent / "agent_report_iris.md").write_text(agent_report, encoding="utf-8")

print(f"[STATUS] Iris complete - output in {output_dir}")
