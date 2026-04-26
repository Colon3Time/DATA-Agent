
PIPELINE_SPEC
=============
problem_type        : classification
target_column       : Outcome
n_rows              : 767
n_features          : 8
imbalance_ratio     : 1.87
key_features        : Glucose (MI=0.116), BMI, Age, DiabetesPedigreeFunction, Pregnancies
recommended_model   : XGBoost
preprocessing:
  scaling           : StandardScaler
  encoding          : None
  special           : SMOTE (imbalance ratio 1.87, moderate — optional)
data_quality_issues : Glucose, BloodPressure, SkinThickness, Insulin, BMI มี zeros ที่ควร impute (missing)
finn_instructions   : impute zeros ใน Glucose, BloodPressure, BMI, SkinThickness, Insulin ก่อนเทรน — ใช้ median หรือ KNN imputer
