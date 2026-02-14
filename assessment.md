# Feature Engineering Assessment

---

## Task Checklist

1. **Aggregate depth=1 and depth=2 tables (max, min)**
   - Ryan: Partial - brute-force, no depth distinction, blind first-N cols
   - Darren: Done - MUST_AGG tables with Polars, incremental eval

2. **Split data into Applicant vs others (num_group1)**
   - Ryan: Not done
   - Darren: Not done - person tables tested but dropped

3. **Active and closed contracts**
   - Ryan: Not done
   - Darren: Done - status-based + date-based logic

4. **Time-windowed aggregations**
   - Ryan: Not done - no date parsing at all
   - Darren: Done - <=1y, 1-3y, >3y + alternative schemes tested

5. **DPD-conditional aggregations**
   - Ryan: Not done
   - Darren: Done - DPD>=30, DPD>=90 on active/closed

6. **Aggregate redundancy from multiple tables**
   - Ryan: Not done
   - Darren: Not done - no shard dedup

7. **StratifiedGroupKFold with WEEK_NUM**
   - Ryan: Broken - only 2 of 5 folds populated
   - Darren: Alternative - rolling time splits (50,60,70), arguably better

8. **Remove fluctuating features, rank importance**
   - Ryan: Partial - CV filter too loose, RF on all data
   - Darren: Done - incremental block eval + rolling stability

---

## Ryan - Feature_Engineering.ipynb

### What he did
- Loaded all 32 tables with pandas
- Brute-force aggregation: first 15 numeric cols (mean/std/min/max/count), first 8 cat cols (nunique/count)
- Stability filter: CV < 2.0 across WEEK_NUM
- RF importance ranking (100 trees, max_depth=10)
- Produced 1820 features, filtered to 1484

### Issues
- Pandas on 33M+ row tables, should be Polars lazy
- No domain awareness, blindly takes first N columns
- No date handling, D-suffix columns never parsed
- No A/P string-to-float conversion
- Median fill on 2.2B nulls, destroys signal
- KFold broken, only folds 3 and 4 have data
- RF accuracy misleading, 96.86% on 96.9% majority class
- StandardScaler before RF, unnecessary for trees
- No proper CV evaluation, no AUC/Gini on validation
- Test data feature names leaked into strategy section
- CV threshold of 2.0 too generous
- count aggregation redundant across numeric cols

### Verdict
0 usable features. Pipeline fundamentally corrupted by missing type conversion and mass median fill. Top feature names (DPD-related) are directionally correct but values are unreliable.

---

## Darren - feature-engineering-darren.ipynb

### What he did
- Built `build_contract_features()`, a reusable leak-safe aggregation template
- Processes tables incrementally: add block, check AUC delta, keep or drop
- Leak-safe mask: `age_years >= 0` ensures only records known at decision time
- Active/closed contract flags from status columns and date logic
- DPD conditional features (>=30, >=90) crossed with active/closed
- Time-windowed counts (<=1y, 1-3y, >3y) crossed with active/closed
- Rolling stability validation across multiple time cuts (50, 60, 70)
- AP1 window scheme comparison, tested 3 alternatives, selected best

### Tables processed
- credit_bureau_b_1 (cb1) - improved AUC, retained
- applprev_1_0 (ap0) - marginal gain, dropped for stability
- applprev_1_1 (ap1w) - strong gain, retained with custom windows
- deposit_1 (dep) - improved AUC, retained
- debitcard_1 (db) - reduced AUC, dropped
- tax_registry_a_1 (taxa) - strong lift, retained
- applprev_2 (ap2) - strong gain, retained
- person_1 (p1) - reduced AUC, dropped
- person_2 (p2) - reduced AUC, dropped

### Final output
- 74 features across 5 retained blocks: cb1, ap1w, dep, taxa, ap2
- Leak-safe, time-validated, incrementally evaluated

### Issues
- Some structural placeholder features (DPD cols for tables without DPD data, always zero)
- Didn't cover DANGEROUS tables (credit_bureau_a shards, the biggest data source)
- No applicant vs others split
- Hardcoded local paths
- Still reads full tables into memory, not lazy

### Verdict
74 usable features. Solid methodology, properly validated. Main gap is missing the large credit_bureau_a tables and static tables entirely.

---

## Overall Gap Analysis

### Done (from Darren)
- Active/closed contract features
- Time-windowed aggregations
- DPD-conditional aggregations
- Incremental block evaluation with AUC
- Rolling stability validation

### Still missing
- DANGEROUS tables (credit_bureau_a_1_*, credit_bureau_a_2_*), the largest data source, untouched
- Static tables (static_0_0, static_0_1, static_cb_0), 1:1 joins, easiest features, not used
- Applicant vs others split from person tables
- Aggregate redundancy and shard deduplication across table shards
- StratifiedGroupKFold (Darren used rolling splits instead)
- Proper feature importance ranking across all features
- No features from: other_1, tax_registry_b_1, tax_registry_c_1, credit_bureau_b_2
