Vera Visualization Report
==========================

## Visuals Created

### 1. Feature Importance Bar Chart (Top 10)
- **Medium**: Survival prediction feature ranking
- **Audience**: Data scientists, ML engineers
- **Insight**: Shows which features most influence survival prediction
- **File**: charts/01_feature_importance.png

### 2. Survival by Sex (Grouped Bar)
- **Medium**: Gender-based survival comparison
- **Audience**: General audience, executives
- **Insight**: Clear visual comparison of survival rates between males and females
- **File**: charts/02_survival_by_sex.png

### 3. Survival by Pclass (Grouped Bar)
- **Medium**: Socioeconomic class impact on survival
- **Audience**: General audience, executives
- **Insight**: Shows how passenger class affected survival probability
- **File**: charts/03_survival_by_pclass.png

### 4. Age Distribution by Survival (Histogram overlay)
- **Medium**: Age patterns in survival
- **Audience**: Analysts, researchers
- **Insight**: Overlay histogram with KDE showing age distribution differences between survivors and non-survivors
- **File**: charts/04_age_distribution.png

### 5. Fare Distribution by Survival (Box Plot)
- **Medium**: Fare price relationship to survival
- **Audience**: Analysts, executives
- **Insight**: Box plot showing fare distribution differences, including outliers and median values
- **File**: charts/05_fare_distribution.png

### 6. Correlation Heatmap (significant only, p<0.05)
- **Medium**: Feature relationships matrix
- **Audience**: Data scientists
- **Insight**: Shows significant correlations between numeric features, masking non-significant ones
- **File**: charts/06_correlation_heatmap.png

## Key Visual: Survival by Sex + Pclass
These two grouped bar charts provide the most immediate and actionable insights for the Titanic survival analysis. They clearly show:
1. The "women and children first" protocol effect (Sex chart)
2. The socioeconomic disparity in survival (Pclass chart)

## Design Notes
- Used consistent color scheme: red for non-survivors, green for survivors
- Added value labels on all bar charts for precise reading
- Included statistical summaries on distribution charts
- Applied significance filtering on correlation heatmap to avoid misleading interpretations

---
Self-Improvement Report
=======================
**Method used**: Statistical visualization with matplotlib + seaborn
**Rationale**: These chart types are standard for binary classification analysis and provide clear, interpretable insights
**New techniques discovered**:
- Significance masking in correlation heatmaps (p-value thresholding)
- Jittered scatter overlay on box plots for raw data visibility
**Will apply next time**: Yes - significance masking is valuable for preventing misleading interpretations
**Knowledge Base**: Updated with significance masking technique
