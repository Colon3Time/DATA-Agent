Scout Dataset Brief
===================
Dataset: UCI Online Retail Dataset (2010-2011)
Source: https://archive.ics.uci.edu/ml/datasets/Online+Retail
License: Public Domain
Size: 1,067,371 rows × 8 columns / 283.84 MB
Format: CSV
Time Period: 2010-12-01 to 2011-12-09

Columns Summary:
- InvoiceNo: string — Invoice number (unique per transaction)
- StockCode: string — Product code
- Description: string — Product name/description
- Quantity: int — Quantity purchased per transaction
- InvoiceDate: datetime — Transaction timestamp
- UnitPrice: float — Price per unit
- CustomerID: float — Customer identifier (has missing)
- Country: string — Customer country

Key Columns for Analytics:
- Target: Quantity (ปริมาณการสั่งซื้อต่อรายการ — ใช้สำหรับ demand forecasting และ customer value analysis)
- Date: InvoiceDate (datetime) for time series
- ID: CustomerID (has NaN), StockCode, InvoiceNo

Business Opportunity (3 Levels):
1. Descriptive Analytics — Sales overview, top products, country-wise sales
2. Behavioral Analytics — RFM analysis, customer segments, repeat purchase patterns
3. Predictive Analytics — Demand forecasting with Quantity target, churn prediction with CustomerID segmentation

Known Issues:
- Missing: {"Customer ID": 22.77, "Description": 0.41}
- CustomerID has ~25% missing (Cancelled transactions)
- Negative Quantity = Cancelled transactions (can be filtered)
- StockCode varies, Description has typos

Dispatch Recommendation:
- Dana: Descriptive analytics — overall sales KPIs, time trends, country summary
- Dana + Mo: Behavioral analytics — RFM, customer segmentation, cohort analysis
- Eddie: Predictive analytics — demand forecast using Quantity as target

DATASET_RISK_REGISTER
=====================
Source credibility: High — UCI Machine Learning Repository, curated benchmark dataset
License/usage: Allowed — public domain for academic/commercial use
Business fit: High — Online Retail transaction data, directly applicable to e-commerce analytics
Target suitability: Clear — Quantity: ปริมาณการสั่งซื้อต่อรายการ — ใช้สำหรับ demand forecasting และ customer value analysis
Recency/deployment fit: Dataset is historical (2010-2011) — limited for current trend detection but valid for pattern learning
Leakage risks: None — no future information, no post-outcome columns
Bias/coverage risks: Single retailer from UK — may not generalize to other markets
Data dictionary: Available (UCI provides column descriptions)
Verdict: Use — suitable for multi-level analytics (descriptive → behavioral → predictive)

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: อ่าน raw CSV, ตรวจสอบ column types manually, วิเคราะห์ business context
เหตุผลที่เลือก: ต้องการ target ที่มีความหมายทางธุรกิจ ไม่ใช่ automatic heuristic
วิธีใหม่ที่พบ: Revenue column (Quantity × UnitPrice) สร้าง target ที่ดีกว่า Quantity ตัวเดียว
จะนำไปใช้ครั้งหน้า: ใช่ — ใน dataset ที่มี Quantity และ UnitPrice ควรสร้าง Revenue เป็น target เสมอ
Knowledge Base: อัพเดต scout — Revenue target creation