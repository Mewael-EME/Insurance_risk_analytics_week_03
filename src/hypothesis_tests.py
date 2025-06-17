# File: hypothesis_tests.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

def compute_kpis(df):
    """
    Add KPI columns to DataFrame:
    - HasClaim (bool): True if TotalClaims > 0
    - ClaimSeverity: Average claim amount given a claim occurred
    - Margin: TotalPremium - TotalClaims
    """
    df = df.copy()
    df['HasClaim'] = df['TotalClaims'] > 0
    # Avoid division by zero - only compute severity for claims > 0
    df['ClaimSeverity'] = np.where(df['HasClaim'], df['TotalClaims'], np.nan)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def segment_data(df, feature_col, group_a_val, group_b_val):
    """
    Create control (Group A) and test (Group B) groups based on feature_col values.
    Returns two DataFrames.
    """
    group_a = df[df[feature_col] == group_a_val]
    group_b = df[df[feature_col] == group_b_val]
    return group_a, group_b

def t_test(group_a, group_b, kpi_col):
    """
    Perform two-sample t-test on KPI numerical columns.
    Returns t-statistic and p-value.
    """
    data_a = group_a[kpi_col].dropna()
    data_b = group_b[kpi_col].dropna()
    t_stat, p_val = ttest_ind(data_a, data_b, equal_var=False)  # Welch’s t-test
    return t_stat, p_val

def chi_squared_test(group_a, group_b, kpi_col):
    """
    Perform chi-square test on binary/categorical KPI.
    Expects kpi_col to be binary indicator (e.g., HasClaim).
    Returns chi2 statistic and p-value.
    """
    contingency_table = pd.DataFrame({
        'GroupA': [group_a[kpi_col].sum(), (~group_a[kpi_col]).sum()],
        'GroupB': [group_b[kpi_col].sum(), (~group_b[kpi_col]).sum()]
    }, index=[f'{kpi_col}=True', f'{kpi_col}=False'])
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p

def interpret_result(p_val, alpha=0.05):
    """
    Returns interpretation string based on p-value.
    """
    if p_val < alpha:
        return f"Reject Null Hypothesis (p = {p_val:.4f}) — statistically significant difference found."
    else:
        return f"Fail to Reject Null Hypothesis (p = {p_val:.4f}) — no statistically significant difference."

