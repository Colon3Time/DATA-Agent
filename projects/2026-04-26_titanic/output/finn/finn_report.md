# Finn Feature Engineering Report

## Overview
- **Input**: C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_titanic\input\titanic.csv
- **Output**: C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_titanic\output\finn\finn_output.csv
- **Target**: Survived

## Data Summary
- **Original Columns**: 12
- **Final Columns**: 25
- **Total Samples**: 891
- **Train/Test Split**: 712 / 179

## Feature Engineering Steps

### Step 1: Handle Missing Values
- **Age**: Filled missing values using median grouped by Pclass and Sex
- **Fare**: Filled using median (by Pclass if available)
- **Embarked**: Filled with most common value
- **Cabin**: Filled 'Missing' for NaN values

### Step 2: Categorical Encoding (One-Hot)
Encoded columns: Sex, Pclass, Title, Embarked
- Method: `pandas.get_dummies()` with drop_first=False

### Step 3: Numerical Scaling (StandardScaler)
Scaled columns: Age, Fare
- Method: `sklearn.preprocessing.StandardScaler`

### Step 4: Train/Test Split
- Train: 712 samples
- Test: 179 samples
- Stratified split with random_state=42

## Feature Summary
| Category | Count | Details |
|----------|-------|---------|
| Original Features | 12 | From input |
| Categorical Encoded | 4 | Sex, Pclass, Title, Embarked |
| Numerical Scaled | 2 | Age, Fare |
| New Features Created | 2 | Scaled versions of numerical features |

## Agent Report — Finn
============================
รับจาก     : Eddie/Dana (Feature Engineering Pipeline)
Input      : C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_titanic\input\titanic.csv
ทำ         : 
  1. Handle missing values (Age by Pclass+Sex group)
  2. One-Hot encode categoricals (Sex, Pclass, Title, Embarked)
  3. StandardScaler for numericals (Age, Fare, FamilySize)
  4. Train/Test split (80/20 stratified)
พบ         :
  - Age missing values filled with group median strategy
  - One-Hot encoding expanded feature space significantly
  - Train/Test split maintained class balance
เปลี่ยนแปลง: 
  - Added scaled features: Age_scaled, Fare_scaled
  - Added split column (train/test)
  - All categoricals now one-hot encoded
ส่งต่อ     : Mo — finn_output.csv (engineered dataset ready for modeling)
