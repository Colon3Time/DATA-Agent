MO_REPORT
=========

PRODUCTION_READINESS
====================
problem_type: regression
target_column: Value
winner model: numpy_ridge_alpha_1
rmse_log_scale: 0.175815
mae_log_scale: 0.138608
r2_log_scale: 0.992766
target_trained: log1p(Value)
target_transform: log1p(Value)
inverse_transform: np.expm1
target_transform_note: clipped 7 negative Value rows to 0 before log1p
target_transform_note: hard override for target Value
rmse_original_scale: 24.874606
mae_original_scale: 11.457821
r2_original_scale: 0.973347
train_rows: 84
test_rows: 21
validation: deterministic holdout split 80/20
missing_dependencies: sklearn, lightgbm, xgboost
business impact: use predictions as an analytical signal only until domain validation is complete
risk: builtin model is a minimum viable benchmark, not production approval
confidence: Medium