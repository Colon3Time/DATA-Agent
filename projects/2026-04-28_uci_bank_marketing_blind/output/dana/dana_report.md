Dana Cleaning Report
====================
Before: 41188 rows, 21 columns
After:  41188 rows, 22 columns

Missing Values:
- No missing values detected (confirmed Scout report)

Data Types Conversion:
- Categorical columns converted: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, y
- Numeric columns verified: age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

Categorical Column Cardinality:
- job: 12 unique, most common="admin.", least common="unknown" (330 rows)
- marital: 4 unique, most common="married", least common="unknown" (80 rows)
- education: 8 unique, most common="university.degree", least common="illiterate" (18 rows)
- default: 3 unique, most common="no", least common="yes" (3 rows)
- housing: 3 unique, most common="yes", least common="unknown" (990 rows)
- loan: 3 unique, most common="no", least common="unknown" (990 rows)
- contact: 2 unique, most common="cellular", least common="telephone" (15044 rows)
- month: 10 unique, most common="may", least common="dec" (182 rows)
- day_of_week: 5 unique, most common="thu", least common="fri" (7827 rows)
- poutcome: 3 unique, most common="nonexistent", least common="success" (1373 rows)
- y: 2 unique, most common="no", least common="yes" (4640 rows)

Outlier Detection:
- Method: Isolation Forest (contamination=0.05) + IQR (1.5x)
- Likely Error (แก้ไขแล้ว):
  - row 1689: duration=2462.00 -> capped
  - row 2313: duration=3366.00 -> capped
  - row 3774: duration=2680.00 -> capped
  - row 4213: duration=3078.00 -> capped
  - row 6280: duration=3094.00 -> capped
  - ... and 143 more
- Likely Real (เก็บไว้ + flagged):
  - row 27713: age=70.00
  - row 27757: age=76.00
  - row 27780: age=73.00
  - row 27800: age=88.00
  - row 27802: age=88.00
  - ... and 13272 more
- Uncertain (Isolation Forest):
  - 80 rows flagged by Isolation Forest
- outlier_flags.csv: 13505 rows total

Data Quality Score:
- Completeness: Before 100.0% -> After 100.0%
- Validity: Before 100.0% -> After 99.6%
- Overall: Before 100.0% -> After 99.8%

Column Stats (Before -> After):
- age: mean 40.02 -> 40.02, std 10.42 -> 10.42
- duration: mean 258.29 -> 256.08, std 259.28 -> 247.05
- campaign: mean 2.57 -> 2.49, std 2.77 -> 2.39
- pdays: mean 962.48 -> 962.48, std 186.91 -> 186.91
- previous: mean 0.17 -> 0.17, std 0.49 -> 0.49
- emp.var.rate: mean 0.08 -> 0.08, std 1.57 -> 1.57
- cons.price.idx: mean 93.58 -> 93.58, std 0.58 -> 0.58
- cons.conf.idx: mean -40.50 -> -40.50, std 4.63 -> 4.63
- euribor3m: mean 3.62 -> 3.62, std 1.73 -> 1.73
- nr.employed: mean 5167.04 -> 5167.04, std 72.25 -> 72.25
- age: mean 40.02 -> 40.02, std 10.42 -> 10.42

New Method Found: None