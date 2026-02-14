# Feature Engineering Tracker

## Ryan's Features (Feature_Engineering.ipynb)

### Approach
Brute-force aggregation across all 31 non-base tables:
- Numeric columns: groupby `case_id` -> `mean`, `std`, `min`, `max`, `count` (first 15 numeric cols per table)
- Categorical columns: groupby `case_id` -> `nunique`, `count` (first 8 cat cols per table)
- Stability filtering: Coefficient of Variation (CV) < 2.0 across WEEK_NUM groups
- Feature importance: RandomForest (100 trees, max_depth=10)

### Top 15 Features by RF Importance
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | train_static_0_0_avgdpdtolclosure24_3658938P_min | 0.022037 |
| 2 | train_static_0_0_avgdpdtolclosure24_3658938P_mean | 0.017131 |
| 3 | train_static_0_1_avgdpdtolclosure24_3658938P_mean | 0.015769 |
| 4 | train_static_0_0_avgdbddpdlast24m_3658932P_min | 0.015744 |
| 5 | train_applprev_1_0_maxdpdtolerance_577P_mean | 0.015086 |
| 6 | train_credit_bureau_a_1_1_dpdmax_139P_mean | 0.014760 |
| 7 | train_credit_bureau_a_1_2_dpdmax_757P_mean | 0.013470 |
| 8 | train_static_0_0_avgdpdtolclosure24_3658938P_max | 0.013417 |
| 9 | train_credit_bureau_a_1_1_dpdmax_139P_max | 0.012764 |
| 10 | train_credit_bureau_a_1_2_dpdmax_139P_mean | 0.012125 |
| 11 | train_applprev_1_0_maxdpdtolerance_577P_min | 0.011965 |
| 12 | train_credit_bureau_a_1_1_dpdmax_757P_mean | 0.011173 |
| 13 | train_applprev_1_0_maxdpdtolerance_577P_max | 0.010656 |
| 14 | train_static_cb_0_days120_123L_min | 0.010211 |
| 15 | train_credit_bureau_a_1_2_dpdmax_139P_max | 0.009074 |

### Stats
- Total after aggregation: 1820 features
- After stability filter: 1484 features
- Unstable removed: 332 features

### Issues Found
1. **Pandas on huge data** - loads 33M+ row tables into pandas (should use Polars lazy)
2. **No domain awareness** - blindly takes first N columns, no reasoning about which features matter
3. **No date handling** - D-suffix columns not converted to relative time
4. **No A/P string-to-float** - columns ending in A/P that loaded as strings are ignored
5. **Median fill on 2.2B nulls** - destructive; fills everything including legitimately missing data
6. **KFold is broken** - only folds 3 and 4 have data (fold assignment logic overwrites)
7. **RF accuracy misleading** - 96.86% accuracy on 96.9% majority class means nothing
8. **StandardScaler before RF** - unnecessary for tree-based models
9. **No proper CV evaluation** - trained on all data, no AUC/Gini reported on validation
10. **Test data feature names leaked** - Strategy section references `test_*` prefixed features
11. **CV threshold too generous** - 2.0 keeps nearly everything, doesn't really filter
12. **count aggregation redundant** - every numeric col gets a `_count` that's identical per case_id

### Verdict
The top features identified (DPD-related: days past due, tolerance, closure) are directionally correct - these are known strong predictors for credit risk. But the pipeline itself needs to be rebuilt properly.
