import numpy as np
from scipy.stats import chi2
from core.statistics_core.utils import evaluate_trajectories

def compute_chi_squared(x_obs, x_exp, sigma_x, num_params_fitted):
    """
    Calculates Chi-Squared statistic, reduced Chi-Squared, and p-value from arrays.
    (Unchanged — operates on pre-computed arrays.)
    """
    residuals = (x_obs - x_exp) / sigma_x
    total_chi2 = float(np.sum(residuals ** 2))

    total_points = len(x_obs)
    dof = max(1, total_points - int(num_params_fitted))
    reduced_chi2 = total_chi2 / dof if dof > 0 else float('inf')
    p_val_chi2 = float(chi2.sf(total_chi2, dof))

    return {
        "chi2_statistic": round(total_chi2, 4),
        "degrees_of_freedom": int(dof),
        "reduced_chi2": round(reduced_chi2, 4),
        "chi2_p_value": p_val_chi2,
        "null_hypothesis_accepted": bool(0.5 <= reduced_chi2 <= 2.0)
    }


def run_chi_squared(experiments_file_path, script_or_func, sigma_x_func, num_params_fitted, eval_data=None, bias_x_func=None):  # ← CHANGED: removed calc_const_noise, calc_lin_noise; added sigma_x_func
    """
    Standalone function for agents or scripts to run Chi-Squared validation.
    Accepts either a script file path or direct callable function.
    sigma_x_func: callable with signature sigma_x(x, v, t) -> float or ndarray.
    """
    if eval_data is None:
            eval_data = evaluate_trajectories(experiments_file_path, script_or_func, sigma_x_func, bias_x_func=bias_x_func)  # ← CHANGED: Pass bias_x_func to evaluate_trajectories
    return compute_chi_squared(
        x_obs=eval_data["x_obs"],
        x_exp=eval_data["x_exp"],
        sigma_x=eval_data["sigma_x"],
        num_params_fitted=num_params_fitted
    )
