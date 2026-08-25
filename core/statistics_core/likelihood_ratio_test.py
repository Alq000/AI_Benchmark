import numpy as np
from scipy.stats import chi2
from core.statistics_core.utils import evaluate_trajectories, compute_gaussian_log_likelihood

def compute_likelihood_ratio_test(log_likelihood_full, log_likelihood_reduced, num_params_full, num_params_reduced=1):
    """
    Evaluates Likelihood Ratio Test statistic LR = 2 * (ln(L_full) - ln(L_reduced)) and chi2 p-value.
    """
    df = int(num_params_full) - int(num_params_reduced)
    
    if df > 0:
        lr_stat = 2.0 * (float(log_likelihood_full) - float(log_likelihood_reduced))
        lr_stat = max(0.0, lr_stat)
        p_val_lr = float(chi2.sf(lr_stat, df))
    else:
        lr_stat = 0.0
        p_val_lr = 1.0

    return {
        "lr_stat": round(lr_stat, 4),
        "lr_p_value": p_val_lr,
        "lr_df": int(max(0, df))
    }


def run_likelihood_ratio_test(experiments_file_path, script_or_func, sigma_x_func,
                               num_params_fitted, num_params_reduced=1,
                               log_likelihood_reduced=None, eval_data=None):
    """
    Standalone function for agents or scripts to run Likelihood Ratio Test model comparison.
    Accepts either a script file path or direct callable function.
    """
    if eval_data is None:
        eval_data = evaluate_trajectories(experiments_file_path, script_or_func, sigma_x_func)  # ← CHANGED

    # Compute full model Gaussian log-likelihood
    ll_full = compute_gaussian_log_likelihood(
        x_obs=eval_data["x_obs"],
        x_exp=eval_data["x_exp"],
        sigma_x=eval_data["sigma_x"]
    )

    if log_likelihood_reduced is not None:
        ll_red = float(log_likelihood_reduced)
    else:
        # Default baseline log-likelihood assuming mean baseline prediction
        x_mean = np.mean(eval_data["x_obs"]) if len(eval_data["x_obs"]) > 0 else 0.0
        x_baseline = np.full_like(eval_data["x_obs"], fill_value=x_mean)
        ll_red = compute_gaussian_log_likelihood(
            x_obs=eval_data["x_obs"],
            x_exp=x_baseline,
            sigma_x=eval_data["sigma_x"]
        )

    return compute_likelihood_ratio_test(
        log_likelihood_full=ll_full,
        log_likelihood_reduced=ll_red,
        num_params_full=num_params_fitted,
        num_params_reduced=num_params_reduced
    )
