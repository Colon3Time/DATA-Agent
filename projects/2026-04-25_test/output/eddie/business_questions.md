# Business Questions from EDA

Generated: 2026-04-25 18:56:40

## Questions for Business Stakeholders

- Q: unit_price is highly right-skewed — are there premium segments causing the skew? Should we segment customers by value?
- Q: unit_price has heavy tails — are there outlier events we should investigate separately?
- Q: total_amount is highly right-skewed — are there premium segments causing the skew? Should we segment customers by value?
- Q: total_amount has heavy tails — are there outlier events we should investigate separately?

## Potential Opportunities

- Top region sum dominates — consider focused marketing campaigns and localized strategies
- Strong correlation between unit_price and total_amount — potential for predictive modeling or cross-selling

## Risk Flags

- ⚠️ 42 outliers (8.4%) in unit_price — investigate if these are errors or genuine extreme values
- ⚠️ 56 outliers (11.2%) in total_amount — investigate if these are errors or genuine extreme values

## Suggested Roadmap

- 🟢 Phase 1 (Immediate): Investigate outliers and data quality issues before making strategic decisions
- 🟡 Phase 2 (Short-term): Segment customers based on key numeric features to identify high-value clusters
- 🟡 Phase 2 (Short-term): Develop predictive model using unit_price and total_amount
- 🔵 Phase 3 (Medium-term): Implement geo-targeted marketing campaigns based on regional performance data
- 🔵 Phase 3 (Medium-term): Set up automated data quality monitoring with alerts for >5% missing rates
- 🔴 Phase 4 (Long-term): Build customer retention model to reduce single-purchase rate and increase CLV
