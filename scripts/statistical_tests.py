import numpy as np
from scipy.stats import ttest_ind, levene
from statsmodels.stats.proportion import proportions_ztest

def run_ttest(group1: np.ndarray, group2: np.ndarray):
    """Returns (t_stat, p_value) using Welch or Student's t-test."""
    stat_levene, p_levene = levene(group1, group2)
    equal_var = p_levene > 0.05
    t_stat, p_val = ttest_ind(group1, group2, equal_var=equal_var)
    return t_stat, p_val, equal_var

def run_ztest(success_a: int, nobs_a: int, success_b: int, nobs_b: int):
    """Returns (z_stat, p_value) for two-proportion test."""
    stat, p_val = proportions_ztest([success_a, success_b], [nobs_a, nobs_b])
    return stat, p_val
