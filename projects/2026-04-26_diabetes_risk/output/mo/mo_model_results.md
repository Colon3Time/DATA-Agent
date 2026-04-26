# Mo Model Report
**Date:** 2026-04-26 12:28

## Data Summary
- Input file: finn_output.csv
- Rows: 767, Features: 9
- Target: 1
- Target distribution: {0: 500, 1: 267}

## Model Comparison

             Algorithm  CV_Accuracy_Mean  CV_Accuracy_Std  Test_Accuracy  Test_F1  Test_Precision  Test_Recall  Test_AUC   Time
0        Random Forest            0.7748           0.0303         0.7403   0.7364          0.7349       0.7403    0.8017  1.79s
1                  SVM            0.7699           0.0330         0.7338   0.7237          0.7251       0.7338    0.7696  0.14s
2                  KNN            0.7521           0.0329         0.7273   0.7232          0.7215       0.7273    0.7502  0.07s
3        Decision Tree            0.7227           0.0489         0.7143   0.7154          0.7168       0.7143    0.6906  0.03s
4          Naive Bayes            0.7748           0.0382         0.7078   0.7043          0.7023       0.7078    0.7854  0.02s
5  Logistic Regression            0.7878           0.0511         0.6883   0.6798          0.6774       0.6883    0.8028  1.93s

## Best Model
- **Random Forest**
- Test Accuracy: 0.7403
- Test F1: 0.7364
- Test AUC: 0.8017
- CV Score: 0.7748 ± 0.0303
