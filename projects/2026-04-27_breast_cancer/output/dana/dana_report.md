Dana Cleaning Report
====================
Before: 569 rows, 31 cols
After:  569 rows, 32 cols

Missing Values:
- No missing values detected

Outlier Detection:
- Method: Isolation Forest (contamination=0.05) + IQR (1.5x)

Likely Error (fixed):
  None

Likely Real / Uncertain (kept + flagged):
  - row 82, mean radius: Likely Real (mean radius=25.2200 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 108, mean radius: Likely Real (mean radius=22.2700 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 122, mean radius: Likely Real (mean radius=24.2500 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 164, mean radius: Likely Real (mean radius=23.2700 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 180, mean radius: Likely Real (mean radius=27.2200 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 202, mean radius: Likely Real (mean radius=23.2900 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 212, mean radius: Likely Real (mean radius=28.1100 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 236, mean radius: Likely Real (mean radius=23.2100 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 339, mean radius: Likely Real (mean radius=23.5100 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 352, mean radius: Likely Real (mean radius=25.7300 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 369, mean radius: Likely Real (mean radius=22.0100 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 461, mean radius: Likely Real (mean radius=27.4200 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 503, mean radius: Likely Real (mean radius=23.0900 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 521, mean radius: Likely Real (mean radius=24.6300 IQR outlier (bounds: [5.5800, 21.9000]))
  - row 219, mean texture: Likely Real (mean texture=32.4700 IQR outlier (bounds: [7.7250, 30.2450]))
  - row 232, mean texture: Likely Real (mean texture=33.8100 IQR outlier (bounds: [7.7250, 30.2450]))
  - row 239, mean texture: Likely Real (mean texture=39.2800 IQR outlier (bounds: [7.7250, 30.2450]))
  - row 259, mean texture: Likely Real (mean texture=33.5600 IQR outlier (bounds: [7.7250, 30.2450]))
  - row 265, mean texture: Likely Real (mean texture=31.1200 IQR outlier (bounds: [7.7250, 30.2450]))
  - row 455, mean texture: Likely Real (mean texture=30.7200 IQR outlier (bounds: [7.7250, 30.2450]))
  - ... and 588 more rows
- outlier_flags.csv: 608 rows total

Data Quality Score:
- Completeness: 100.00% -> 100.00%
- Validity: 100.00% -> 100.00%
- Overall: 100.00% -> 100.00%

Column Stats (Before -> After):
- mean radius: mean 14.1273->14.1273, std 3.5240->3.5240
- mean texture: mean 19.2896->19.2896, std 4.3010->4.3010
- mean perimeter: mean 91.9690->91.9690, std 24.2990->24.2990
- mean area: mean 654.8891->654.8891, std 351.9141->351.9141
- mean smoothness: mean 0.0964->0.0964, std 0.0141->0.0141

New Method Found: None