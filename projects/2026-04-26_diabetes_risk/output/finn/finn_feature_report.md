Finn Feature Engineering Report
================================
Dataset: pima_indians_diabetes.csv
Target Column: '1'
Original Features: 8
New Features Created: 1
Final Features Selected: 9

Features Created:
- Interaction features: 10 pairs created
- Polynomial features (deg=2): added via PolynomialFeatures
- Binned features: quantile-based discretization

Encoding Used: None (all numeric data)
Scaling Used: StandardScaler (zero mean, unit variance)

Feature Selection:
- RFECV with RandomForestClassifier performed (if applicable)
- Final feature count: 9
