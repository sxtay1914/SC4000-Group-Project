#!/usr/bin/env python3
"""
Home Credit - Credit Risk Model Stability
Feature Engineering Script
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import glob

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Define the test folder path
test_folder = '/Users/ryanzhang/Documents/Uni stuff/sc4000/home-credit-credit-risk-model-stability/csv_files/test'
print("="*80)
print("HOME CREDIT FEATURE ENGINEERING PIPELINE")
print("="*80)
print(f"\nWorking directory: {test_folder}")
print(f"Files in directory: {len(os.listdir(test_folder))}")

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: LOADING DATA")
print("="*80)

# Load base table
base = pd.read_csv(f'{test_folder}/test_base.csv')
print(f"\n✓ Base table: shape {base.shape}")

# Load application/previous credit tables
applprev_tables = []
applprev_files = sorted([f for f in os.listdir(test_folder) if f.startswith('test_applprev')])
print(f"✓ Found {len(applprev_files)} applprev files")
for file in applprev_files:
    df = pd.read_csv(f'{test_folder}/{file}')
    applprev_tables.append(df)
applprev = pd.concat(applprev_tables, ignore_index=False)
print(f"  Combined applprev: shape {applprev.shape}")

# Load credit bureau tables
creditbureau_tables = []
creditbureau_files = sorted([f for f in os.listdir(test_folder) if f.startswith('test_credit_bureau')])
print(f"✓ Found {len(creditbureau_files)} credit bureau files")
for file in creditbureau_files:
    df = pd.read_csv(f'{test_folder}/{file}')
    creditbureau_tables.append(df)
creditbureau = pd.concat(creditbureau_tables, ignore_index=False)
print(f"  Combined credit bureau: shape {creditbureau.shape}")

# Load person tables
person_tables = []
for file in sorted([f for f in os.listdir(test_folder) if f.startswith('test_person')]):
    df = pd.read_csv(f'{test_folder}/{file}')
    person_tables.append(df)
person = pd.concat(person_tables, ignore_index=False) if person_tables else None
print(f"✓ Person table: shape {person.shape if person is not None else 'None'}")

# Load tax registry tables
tax_registry_tables = []
for file in sorted([f for f in os.listdir(test_folder) if f.startswith('test_tax_registry')]):
    df = pd.read_csv(f'{test_folder}/{file}')
    tax_registry_tables.append(df)
tax_registry = pd.concat(tax_registry_tables, ignore_index=False) if tax_registry_tables else None
print(f"✓ Tax registry: shape {tax_registry.shape if tax_registry is not None else 'None'}")

# ============================================================================
# SECTION 2: FEATURE ENGINEERING FROM APPLICATION/PREVIOUS TABLE
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: FEATURE ENGINEERING - APPLICATION/PREVIOUS")
print("="*80)

numeric_cols_applprev = applprev.select_dtypes(include=[np.number]).columns.tolist()
print(f"✓ Found {len(numeric_cols_applprev)} numeric columns in applprev")

# Aggregate by case_id
applprev_agg = applprev.groupby('case_id')[numeric_cols_applprev].agg([
    'sum', 'mean', 'max', 'min', 'std', 'count'
]).fillna(0)
applprev_agg.columns = ['_'.join(col).strip() for col in applprev_agg.columns]
print(f"✓ Aggregated applprev features: shape {applprev_agg.shape}")

# Additional application features
applprev_features = applprev.groupby('case_id').agg({
    'status_219L': 'count',
}).rename(columns={'status_219L': 'applprev_count'}).reset_index()

if 'credamount_590A' in applprev.columns:
    amount_agg = applprev.groupby('case_id')['credamount_590A'].agg(['sum', 'mean', 'max']).reset_index()
    amount_agg.columns = ['case_id', 'applprev_total_credit_amount', 'applprev_avg_credit_amount', 'applprev_max_credit_amount']
    applprev_features = applprev_features.merge(amount_agg, on='case_id', how='left')

if 'annuity_853A' in applprev.columns:
    annuity_agg = applprev.groupby('case_id')['annuity_853A'].agg(['sum', 'mean']).reset_index()
    annuity_agg.columns = ['case_id', 'applprev_total_annuity', 'applprev_avg_annuity']
    applprev_features = applprev_features.merge(annuity_agg, on='case_id', how='left')

print(f"✓ Engineered applprev features: {applprev_features.shape[1] - 1} features")

# ============================================================================
# SECTION 3: FEATURE ENGINEERING FROM CREDIT BUREAU
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: FEATURE ENGINEERING - CREDIT BUREAU")
print("="*80)

numeric_cols_cb = creditbureau.select_dtypes(include=[np.number]).columns.tolist()
print(f"✓ Found {len(numeric_cols_cb)} numeric columns in credit bureau")

# Aggregate by case_id
creditbureau_agg = creditbureau.groupby('case_id')[numeric_cols_cb].agg([
    'sum', 'mean', 'max', 'min', 'count'
]).fillna(0)
creditbureau_agg.columns = ['cb_' + '_'.join(col).strip() for col in creditbureau_agg.columns]
print(f"✓ Aggregated credit bureau features: shape {creditbureau_agg.shape}")

# Additional credit bureau features
creditbureau_features = creditbureau.groupby('case_id')['overdueamount_31A'].count().reset_index()
creditbureau_features.rename(columns={'overdueamount_31A': 'cb_num_records'}, inplace=True)

if 'overdueamount_31A' in creditbureau.columns:
    cb_overdue = creditbureau.groupby('case_id')['overdueamount_31A'].agg(['sum', 'max', 'mean']).reset_index()
    cb_overdue.columns = ['case_id', 'cb_total_overdue_amount', 'cb_max_overdue_amount', 'cb_avg_overdue_amount']
    creditbureau_features = creditbureau_features.merge(cb_overdue, on='case_id')

if 'outstandingamount_354A' in creditbureau.columns:
    cb_outstanding = creditbureau.groupby('case_id')['outstandingamount_354A'].agg(['sum', 'max']).reset_index()
    cb_outstanding.columns = ['case_id', 'cb_total_outstanding', 'cb_max_outstanding']
    creditbureau_features = creditbureau_features.merge(cb_outstanding, on='case_id')

if 'dpdmax_139P' in creditbureau.columns:
    cb_dpd = creditbureau.groupby('case_id')['dpdmax_139P'].agg(['max', 'mean']).reset_index()
    cb_dpd.columns = ['case_id', 'cb_max_dpd', 'cb_avg_dpd']
    creditbureau_features = creditbureau_features.merge(cb_dpd, on='case_id')

print(f"✓ Engineered credit bureau features: {creditbureau_features.shape[1] - 1} features")

# ============================================================================
# SECTION 4: FEATURE ENGINEERING FROM PERSON AND TAX REGISTRY
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: FEATURE ENGINEERING - PERSON & TAX REGISTRY")
print("="*80)

person_features = pd.DataFrame()
if person is not None:
    person_unique = person.drop_duplicates(subset=['case_id'], keep='first')
    person_features = person_unique[['case_id']].copy()
    
    numeric_person = person_unique.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_person:
        if col != 'case_id' and col in person_unique.columns:
            person_features[f'person_{col}'] = person_unique[col].values
    
    categorical_person = person_unique.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_person[:3]:
        if col != 'case_id':
            person_features[f'person_{col}'] = person_unique[col].values

print(f"✓ Person features: {person_features.shape[1] - 1 if len(person_features) > 0 else 0} features")

tax_features = pd.DataFrame()
if tax_registry is not None:
    tax_unique = tax_registry.drop_duplicates(subset=['case_id'], keep='first')
    tax_features = tax_unique[['case_id']].copy()
    
    numeric_tax = tax_unique.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_tax:
        if col != 'case_id':
            tax_features[f'tax_{col}'] = tax_unique[col].values

print(f"✓ Tax features: {tax_features.shape[1] - 1 if len(tax_features) > 0 else 0} features")

# ============================================================================
# SECTION 5: CREATE INTERACTION FEATURES
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: CREATING INTERACTION FEATURES")
print("="*80)

interaction_features = pd.DataFrame()
if len(applprev_features) > 0 and len(creditbureau_features) > 0:
    # Merge to align for interactions
    temp_merged = applprev_features.merge(creditbureau_features, on='case_id', how='left')
    
    interaction_features = temp_merged[['case_id']].copy()
    
    if 'applprev_count' in temp_merged.columns and 'cb_num_records' in temp_merged.columns:
        interaction_features['applprev_cb_count_ratio'] = (
            temp_merged['applprev_count'] / (temp_merged['cb_num_records'] + 1)
        )
    
    if 'applprev_total_credit_amount' in temp_merged.columns and 'cb_total_outstanding' in temp_merged.columns:
        interaction_features['credit_to_outstanding_ratio'] = (
            temp_merged['applprev_total_credit_amount'] / (temp_merged['cb_total_outstanding'] + 1)
        )

print(f"✓ Interaction features: {interaction_features.shape[1] - 1 if len(interaction_features) > 0 else 0} features")

# ============================================================================
# SECTION 6: MERGE ALL FEATURES
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: MERGING ALL FEATURES")
print("="*80)

engineered_df = base.copy()
print(f"Base table: {engineered_df.shape}")

# Merge applprev features
if len(applprev_features) > 0:
    engineered_df = engineered_df.merge(applprev_features, on='case_id', how='left')
    print(f"After applprev merge: {engineered_df.shape}")

# Merge credit bureau features
if len(creditbureau_features) > 0:
    engineered_df = engineered_df.merge(creditbureau_features, on='case_id', how='left')
    print(f"After creditbureau merge: {engineered_df.shape}")

# Merge person features
if len(person_features) > 0:
    engineered_df = engineered_df.merge(person_features, on='case_id', how='left')
    print(f"After person merge: {engineered_df.shape}")

# Merge tax features
if len(tax_features) > 0:
    engineered_df = engineered_df.merge(tax_features, on='case_id', how='left')
    print(f"After tax merge: {engineered_df.shape}")

# Merge interaction features
if len(interaction_features) > 0:
    engineered_df = engineered_df.merge(interaction_features, on='case_id', how='left')
    print(f"After interaction merge: {engineered_df.shape}")

# ============================================================================
# SECTION 7: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: HANDLING MISSING VALUES")
print("="*80)

numeric_features = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = engineered_df.select_dtypes(include=['object']).columns.tolist()

# Fill missing values directly
for col in numeric_features:
    engineered_df[col].fillna(engineered_df[col].median(), inplace=True)

for col in categorical_features:
    engineered_df[col].fillna(engineered_df[col].mode()[0] if not engineered_df[col].mode().empty else 'Unknown', inplace=True)

print(f"✓ Missing values after imputation: {engineered_df.isnull().sum().sum()}")

# ============================================================================
# SECTION 8: HANDLE OUTLIERS
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: HANDLING OUTLIERS")
print("="*80)

def handle_outliers_iqr(df, columns=None, multiplier=1.5):
    df_clean = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
engineered_df_handled = handle_outliers_iqr(engineered_df, numeric_cols, multiplier=1.5)
engineered_df = engineered_df_handled
print("✓ Outliers handled using IQR method (capped)")

# ============================================================================
# SECTION 9: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "="*80)
print("SECTION 9: ENCODING CATEGORICAL VARIABLES")
print("="*80)

categorical_cols = engineered_df.select_dtypes(include=['object']).columns.tolist()
engineered_df_encoded = engineered_df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    engineered_df_encoded[col] = engineered_df_encoded[col].fillna('Missing')
    engineered_df_encoded[col] = le.fit_transform(engineered_df_encoded[col].astype(str))

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# ============================================================================
# SECTION 10: FEATURE CORRELATION AND SELECTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 10: FEATURE SELECTION - CORRELATION ANALYSIS")
print("="*80)

numeric_features_df = engineered_df_encoded.select_dtypes(include=[np.number])
correlation_matrix = numeric_features_df.corr()

def find_highly_correlated_pairs(corr_matrix, threshold=0.95):
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    return pairs

highly_correlated = find_highly_correlated_pairs(correlation_matrix, threshold=0.95)
print(f"✓ Found {len(highly_correlated)} highly correlated pairs (>0.95)")

def remove_highly_correlated_features(df, corr_matrix, threshold=0.95):
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                to_drop.add(corr_matrix.columns[j])
    
    df_filtered = df.drop(columns=list(to_drop), errors='ignore')
    return df_filtered

engineered_df_filtered = remove_highly_correlated_features(
    engineered_df_encoded, correlation_matrix, threshold=0.95
)
print(f"✓ Removed {engineered_df_encoded.shape[1] - engineered_df_filtered.shape[1]} highly correlated features")

# ============================================================================
# SECTION 11: FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("SECTION 11: FEATURE SCALING")
print("="*80)

numeric_cols_all = engineered_df_encoded.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
engineered_df_scaled = engineered_df_encoded.copy()
engineered_df_scaled[numeric_cols_all] = scaler.fit_transform(engineered_df_encoded[numeric_cols_all])
print("✓ Features scaled using StandardScaler")

# ============================================================================
# SECTION 12: EXPORT FEATURES
# ============================================================================
print("\n" + "="*80)
print("SECTION 12: EXPORTING FEATURES")
print("="*80)

# Export versions
output_file_1 = f'{test_folder}/test_engineered_features_full.csv'
engineered_df_encoded.to_csv(output_file_1, index=False)
print(f"✓ Exported: test_engineered_features_full.csv ({engineered_df_encoded.shape[1]} features)")

output_file_2 = f'{test_folder}/test_engineered_features_filtered.csv'
engineered_df_filtered.to_csv(output_file_2, index=False)
print(f"✓ Exported: test_engineered_features_filtered.csv ({engineered_df_filtered.shape[1]} features)")

output_file_3 = f'{test_folder}/test_engineered_features_scaled.csv'
engineered_df_scaled.to_csv(output_file_3, index=False)
print(f"✓ Exported: test_engineered_features_scaled.csv ({engineered_df_scaled.shape[1]} features)")

# Feature metadata
feature_metadata = pd.DataFrame({
    'Feature': engineered_df_encoded.columns,
    'Data_Type': engineered_df_encoded.dtypes.values,
    'Missing_Count': engineered_df_encoded.isnull().sum().values,
    'Missing_Percentage': (engineered_df_encoded.isnull().sum() / len(engineered_df_encoded) * 100).values
})

output_file_4 = f'{test_folder}/test_engineered_features_metadata.csv'
feature_metadata.to_csv(output_file_4, index=False)
print(f"✓ Exported: test_engineered_features_metadata.csv (Feature metadata)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)

print(f"\nOriginal dataset: {base.shape[0]} rows × {base.shape[1]} columns")
print(f"Engineered dataset: {engineered_df_encoded.shape[0]} rows × {engineered_df_encoded.shape[1]} columns")
print(f"  → Features added: {engineered_df_encoded.shape[1] - base.shape[1]}")
print(f"\nFiltered dataset: {engineered_df_encoded.shape[0]} rows × {engineered_df_filtered.shape[1]} columns")
print(f"  → Features removed (high correlation): {engineered_df_encoded.shape[1] - engineered_df_filtered.shape[1]}")

print(f"\nFeature sources:")
print(f"  - Base table: {base.shape[1]} features")
print(f"  - Application history: {len(applprev_features.columns)} engineered features")
print(f"  - Credit bureau: {len(creditbureau_features.columns)} engineered features")
print(f"  - Person data: {len(person_features.columns)} features")
print(f"  - Tax registry: {len(tax_features.columns)} features")
print(f"  - Interaction features: {len(interaction_features.columns)} features")

print(f"\n✓ Feature engineering completed successfully!")
print("="*80)
