import numpy as np
from scipy.stats import kstest
from core.statistics_core.utils import evaluate_trajectories

def compute_ks_test(x_obs, x_exp, sigma_x, cdf="norm"):
    """
    Calculates Kolmogorov-Smirnov (KS) statistic (supremum distance) and p-value
    comparing normalized residuals against a target reference cumulative distribution (default: standard normal).
    """
    x_obs = np.array(x_obs, dtype=float)
    x_exp = np.array(x_exp, dtype=float)
    sigma_x = np.array(sigma_x, dtype=float)

    if len(x_obs) == 0:
        return {
            "ks_stat": 0.0,
            "ks_p_value": 1.0,
            "null_hypothesis_accepted": False
        }

    # Calculate normalized residuals z = (x_obs - x_exp) / sigma_x
    normalized_residuals = (x_obs - x_exp) / sigma_x
    
    # Perform unbinned KS test against normal CDF N(0, 1)
    res = kstest(normalized_residuals, cdf)
    ks_stat = float(res.statistic)
    p_val_ks = float(res.pvalue)

    return {
        "ks_stat": round(ks_stat, 4),
        "ks_p_value": p_val_ks,
        "null_hypothesis_accepted": bool(p_val_ks >= 0.05)
    }


def run_ks_test(experiments_file_path, script_or_func, sigma_x_func,
                cdf='norm', eval_data=None):
    """
    Standalone function for agents or scripts to run Kolmogorov-Smirnov test.
    Accepts either a script file path or direct callable function.
    """
    if eval_data is None:
        eval_data = evaluate_trajectories(experiments_file_path, script_or_func, sigma_x_func)  # ← CHANGED

    return compute_ks_test(
        x_obs=eval_data["x_obs"],
        x_exp=eval_data["x_exp"],
        sigma_x=eval_data["sigma_x"],
        cdf=cdf
    )
