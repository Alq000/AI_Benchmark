from scipy.stats import f
from core.statistics_core.utils import evaluate_trajectories

def compute_f_test(rss_full, rss_reduced, total_points, num_params_full, num_params_reduced=1):
    """
    Calculates F-statistic, degrees of freedom, and p-value from RSS inputs.
    """
    p_full = int(num_params_full)
    p_red = int(num_params_reduced)
    
    df_num = p_full - p_red
    df_denom = total_points - p_full
    
    if df_num > 0 and df_denom > 0 and rss_full > 0:
        f_stat = float(((rss_reduced - rss_full) / df_num) / (rss_full / df_denom))
        f_stat = max(0.0, f_stat)
        p_val_f_test = float(f.sf(f_stat, df_num, df_denom))
    else:
        f_stat = 0.0
        p_val_f_test = 1.0

    return {
        "f_stat": round(f_stat, 4),
        "f_test_p_value": p_val_f_test,
        "f_test_df_numerator": int(df_num),
        "f_test_df_denominator": int(df_denom)
    }


def run_f_test(experiments_file_path, script_or_func, sigma_x_func,
               num_params_fitted, num_params_reduced=1, rss_reduced=None, eval_data=None):
    """
    Standalone function for agents or scripts to run F-test model comparison.
    Accepts either a script file path or direct callable function.
    """
    if eval_data is None:
        eval_data = evaluate_trajectories(experiments_file_path, script_or_func, sigma_x_func)  # ← CHANGED

    rss_red = float(rss_reduced) if rss_reduced is not None else float(eval_data["total_rss_mean_baseline"])

    return compute_f_test(
        rss_full=eval_data["total_rss_full"],
        rss_reduced=rss_red,
        total_points=eval_data["total_points"],
        num_params_full=num_params_fitted,
        num_params_reduced=num_params_reduced
    )
