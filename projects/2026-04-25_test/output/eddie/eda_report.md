Eddie EDA & Business Report
==============================
Generated: 2026-04-25 18:56:40
Dataset: 500 rows × 10 columns

---
### 📊 Data Quality Summary
- **Shape**: [500, 10]
- **Total Missing**: 0 (0.0%)
- **Duplicates**: 0

---
### 📈 Univariate Analysis (Numerical Columns)

**unit_price**
- Count: 500 | Mean: 6965.0 | Median: 1800.0 | Std: 12088.46
- Min: 150 | Q25: 600.0 | Q75: 8500.0 | Max: 45000
- Skewness=2.564030153635525: Highly skewed (right-tailed: many low values, few high values/potential outliers). Kurtosis=5.371437299462315: Strongly leptokurtic (significant outlier presence).

**quantity**
- Count: 500 | Mean: 10.61 | Median: 10.5 | Std: 5.67
- Min: 1 | Q25: 6.0 | Q75: 16.0 | Max: 20
- Skewness=0.03091721368780985: Near-symmetric distribution. Kurtosis=-1.2003382381104757: Moderately platykurtic (fewer outliers than normal).

**discount_pct**
- Count: 500 | Mean: 0.05 | Median: 0.05 | Std: 0.06
- Min: 0.0 | Q25: 0.0 | Q75: 0.1 | Max: 0.15
- Skewness=0.535372035517115: Moderately skewed (right-tailed: most values below mean). Kurtosis=-1.334688784085087: Moderately platykurtic (fewer outliers than normal).

**total_amount**
- Count: 500 | Mean: 67411.16 | Median: 17000.0 | Std: 142566.08
- Min: 300.0 | Q25: 5745.0 | Q75: 60800.0 | Max: 900000.0
- Skewness=3.888650685646937: Highly skewed (right-tailed: many low values, few high values/potential outliers). Kurtosis=16.418405326754794: Strongly leptokurtic (significant outlier presence).

---
### 🔗 Correlation Analysis
**High Correlations Found:**
- unit_price ↔ total_amount: 0.818 (strong)

---
### ⚠️ Outlier Analysis
- **unit_price**: 42 outliers (8.4%), bounds=[-11250.0, 20350.0]
- **quantity**: 0 outliers (0.0%), bounds=[-9.0, 31.0]
- **discount_pct**: 0 outliers (0.0%), bounds=[-0.15, 0.25]
- **total_amount**: 56 outliers (11.2%), bounds=[-76837.5, 143382.5]

---
### 🗺️ Geographic Insights
- Geography column: **region**
- Value column: **unit_price**
- Top 5 regions by share:
  1. sum
  2. mean
  3. count
  4. share_pct

---
### 📉 Time Series Analysis
- No date column detected for time series analysis

---
### 📝 Executive Business Insights
• unit_price: mean=6965.0, median=1800.0, range=[150–45000] — 8.4% outliers detected
• quantity: mean=10.61, median=10.5, range=[1–20]
• discount_pct: mean=0.05, median=0.05, range=[0.0–0.15]
• Key correlation: unit_price vs total_amount = 0.818 (strong)
• date: 264 unique values — Top 10 categories cover 9.6% of data
• customer_id: 478 unique values — Top 10 categories cover 4.2% of data

---
### ❓ Actionable Business Questions
- Q: unit_price is highly right-skewed — are there premium segments causing the skew? Should we segment customers by value?
- Q: unit_price has heavy tails — are there outlier events we should investigate separately?
- Q: total_amount is highly right-skewed — are there premium segments causing the skew? Should we segment customers by value?
- Q: total_amount has heavy tails — are there outlier events we should investigate separately?

---
### 💡 Opportunities Found
- ✅ Top region sum dominates — consider focused marketing campaigns and localized strategies
- ✅ Strong correlation between unit_price and total_amount — potential for predictive modeling or cross-selling

---
### 🚨 Risk Signals
- ⚠️ 42 outliers (8.4%) in unit_price — investigate if these are errors or genuine extreme values
- ⚠️ 56 outliers (11.2%) in total_amount — investigate if these are errors or genuine extreme values

---
### 🗺️ Actionable Roadmap
- 🟢 Phase 1 (Immediate): Investigate outliers and data quality issues before making strategic decisions
- 🟡 Phase 2 (Short-term): Segment customers based on key numeric features to identify high-value clusters
- 🟡 Phase 2 (Short-term): Develop predictive model using unit_price and total_amount
- 🔵 Phase 3 (Medium-term): Implement geo-targeted marketing campaigns based on regional performance data
- 🔵 Phase 3 (Medium-term): Set up automated data quality monitoring with alerts for >5% missing rates
- 🔴 Phase 4 (Long-term): Build customer retention model to reduce single-purchase rate and increase CLV

---
### 🔬 Statistical Tests
- 2025-06-08 vs 2025-07-22: t=0.1861, p=0.8561 — Significant: No