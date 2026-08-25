import numpy as np
from scipy.stats import f
from core.statistics_core.utils import evaluate_trajectories

def compute_chow_test(rss_pooled, rss_1, rss_2, total_points, num_params_fitted):
    """
    Calculates Chow Test F-statistic and p-value for structural stability/subgroup consistency.
    F = [ (RSS_pooled - (RSS_1 + RSS_2)) / k ] / [ (RSS_1 + RSS_2) / (N - 2k) ]
    """
    k = int(num_params_fitted)
    n = int(total_points)
    
    df_num = k
    df_denom = n - (2 * k)

    rss_sum = rss_1 + rss_2

    if df_num > 0 and df_denom > 0 and rss_sum > 0:
        chow_stat = float(((rss_pooled - rss_sum) / df_num) / (rss_sum / df_denom))
        chow_stat = max(0.0, chow_stat)
        p_val_chow = float(f.sf(chow_stat, df_num, df_denom))
    else:
        chow_stat = 0.0
        p_val_chow = 1.0

    return {
        "chow_stat": round(chow_stat, 4),
        "chow_p_value": p_val_chow,
        "chow_df_numerator": int(df_num),
        "chow_df_denominator": int(max(0, df_denom))
    }


def run_chow_test(experiments_file_path, script_or_func, sigma_x_func,
                  num_params_fitted, split_ratio=0.5, eval_data=None):

    """
    Standalone function for agents or scripts to evaluate structural consistency via Chow Test.
    Splits evaluated residuals into two temporal/structural subsets (default 50/50 split).
    """
    if eval_data is None:
        eval_data = evaluate_trajectories(experiments_file_path, script_or_func, sigma_x_func)  # ← CHANGED

    x_obs = eval_data["x_obs"]
    x_exp = eval_data["x_exp"]
    n_points = len(x_obs)

    if n_points == 0:
        return {
            "chow_stat": 0.0,
            "chow_p_value": 1.0,
            "chow_df_numerator": int(num_params_fitted),
            "chow_df_denominator": 0
        }

    split_idx = int(n_points * float(split_ratio))
    split_idx = max(1, min(n_points - 1, split_idx))

    residuals = x_obs - x_exp
    rss_pooled = float(np.sum(residuals ** 2))

    res_1 = residuals[:split_idx]
    res_2 = residuals[split_idx:]

    rss_1 = float(np.sum(res_1 ** 2))
    rss_2 = float(np.sum(res_2 ** 2))

    return compute_chow_test(
        rss_pooled=rss_pooled,
        rss_1=rss_1,
        rss_2=rss_2,
        total_points=n_points,
        num_params_fitted=num_params_fitted
    )
