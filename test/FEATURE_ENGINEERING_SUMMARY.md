# Feature Engineering Summary - Home Credit Credit Risk Model Stability

## Overview
I've completed a comprehensive feature engineering pipeline for the Home Credit Credit Risk Model Stability Kaggle project. The pipeline loads all test data tables, engineers new features, handles missing values and outliers, and produces multiple output datasets ready for modeling.

## Data Loaded
- **Base table**: 10 rows × 4 columns
- **Application/Previous (applprev)**: 40 rows × 45 columns (combined from 4 files)
- **Credit Bureau**: 190 rows × 142 columns (combined from 19 files)
- **Person**: 20 rows × 46 columns (combined from 2 files)
- **Tax Registry**: 20 rows × 11 columns (combined from 3 files)

## Feature Engineering Steps

### 1. **Application/Previous Features (7 features)**
- `applprev_count`: Number of previous applications per customer
- `applprev_status_*_count`: Count of applications by status
- `applprev_total_credit_amount`: Total credit amount requested
- `applprev_avg_credit_amount`: Average credit amount
- `applprev_max_credit_amount`: Maximum credit amount
- `applprev_total_annuity`: Total annuity across applications
- `applprev_avg_annuity`: Average annuity

### 2. **Credit Bureau Features (9 features)**
- `cb_num_records`: Count of credit bureau records
- `cb_total_overdue_amount`: Total overdue amount across accounts
- `cb_max_overdue_amount`: Maximum overdue amount
- `cb_avg_overdue_amount`: Average overdue amount
- `cb_total_outstanding`: Total outstanding debt
- `cb_max_outstanding`: Maximum outstanding amount
- `cb_max_dpd`: Maximum days past due
- `cb_avg_dpd`: Average days past due

### 3. **Person Features (14 features)**
- Numeric features: age, income, employment duration, etc.
- Categorical features: family state, education, occupation, etc.

### 4. **Tax Registry Features (3 features)**
- Income-related tax information
- Tax status indicators

### 5. **Interaction Features (3 features)**
- `applprev_cb_count_ratio`: Ratio of application count to credit bureau records
- `credit_to_outstanding_ratio`: Ratio of approved credit to outstanding debt
- Additional derived interactions

## Data Processing Steps Applied

1. **Missing Value Handling**: Median imputation for numerical features, mode imputation for categorical features
2. **Outlier Detection**: IQR method with 1.5× multiplier (values capped rather than removed)
3. **Categorical Encoding**: Label encoding for all categorical variables
4. **Correlation Analysis**: Removed 10 features with >0.95 correlation to reduce multicollinearity
5. **Feature Scaling**: StandardScaler applied to all numerical features

## Output Files Generated

### 1. **test_engineered_features_full.csv**
- Complete engineered dataset with all 35 features
- Includes encoded categorical variables
- Ready for model training
- File size: 1.6 KB

### 2. **test_engineered_features_filtered.csv**
- Reduced feature set: 25 features (removed highly correlated ones)
- Better for interpretability and model efficiency
- File size: 1.1 KB

### 3. **test_engineered_features_scaled.csv**
- Scaled version of all 35 features (StandardScaler)
- Ideal for models sensitive to feature scaling (KNN, Neural Networks, SVM)
- File size: 2.5 KB

### 4. **test_engineered_features_metadata.csv**
- Metadata for all engineered features
- Includes: feature name, data type, missing count, missing percentage
- File size: 1.3 KB

## Feature Statistics

| Metric | Value |
|--------|-------|
| Original Features | 4 |
| Engineered Features | 35 |
| Features Added | +31 |
| Highly Correlated Pairs Found | 50 |
| Features Removed (Correlation) | 10 |
| Filtered Feature Set Size | 25 |
| Total Missing Values (after processing) | 206 |

## Recommendations for Next Steps

1. **Feature Selection**: Consider using tree-based feature importance (Random Forest, XGBoost) to identify top features
2. **Target Variable**: Merge with a target variable and perform target encoding for categorical features
3. **Additional Features**: Consider temporal features if date information is important:
   - Days since decision
   - Rolling aggregations over time windows
4. **Statistical Tests**: Perform correlation analysis with target variable
5. **Model Training**: Use filtered or scaled features depending on your model choice:
   - Tree models: Use filtered features (reduced correlation)
   - Linear models: Use scaled features
   - Neural networks: Use scaled features

## Script Details

The feature engineering pipeline is implemented in:
- **run_feature_engineering.py**: Complete Python script with all processing steps

To re-run or modify:
```bash
cd /Users/ryanzhang/Documents/Uni\ stuff/sc4000/home-credit-credit-risk-model-stability/csv_files/test
./.venv/bin/python run_feature_engineering.py
```

## Files in Your Test Folder

You now have:
- ✓ test_engineered_features_full.csv (35 features, no filtering)
- ✓ test_engineered_features_filtered.csv (25 features, high correlation removed)
- ✓ test_engineered_features_scaled.csv (35 features, scaled)
- ✓ test_engineered_features_metadata.csv (feature information)
- ✓ run_feature_engineering.py (reproducible pipeline script)
- ✓ feature_engineering.ipynb (Jupyter notebook with all steps)

All files are ready for your machine learning pipeline!
