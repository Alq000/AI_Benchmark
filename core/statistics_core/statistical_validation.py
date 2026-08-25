import json
import os
from core.statistics_core.utils import evaluate_trajectories
from core.statistics_core.chi_squared import run_chi_squared
from core.statistics_core.f_test import run_f_test
from core.statistics_core.ks_test import run_ks_test
from core.statistics_core.likelihood_ratio_test import run_likelihood_ratio_test
from core.statistics_core.chow_test import run_chow_test
from core.statistics_core.information_criteria import run_ic_evaluation

def compute_ensemble_chi_squared(
    experiments_file_path,
    script_or_func,
    sigma_x_func,                  # ← CHANGED: was calc_const_noise, calc_lin_noise (two floats)
    num_params_fitted,
    bias_x_func=None,
    num_params_reduced=1,
    rss_reduced=None,
    log_likelihood_reduced=None,
    split_ratio=0.5
):
    """
    Orchestrates full ensemble statistical validation by invoking Chi-Squared, F-Test,
    Kolmogorov-Smirnov Test, Likelihood Ratio Test, Chow Test, and Information Criteria tools.
    sigma_x_func: callable with signature sigma_x(x, v, t) -> float or ndarray,
                  i.e. the same sigma_x function from your submission.
    """
    # 1. Evaluate trajectory data once via shared utility
    eval_data = evaluate_trajectories(
        experiments_file_path,
        script_or_func,
        sigma_x_func,                            
        bias_x_func=bias_x_func
    )

    # 2. Delegate calculations to individual standalone tools
    chi2_results = run_chi_squared(
        experiments_file_path=experiments_file_path,
        script_or_func=script_or_func,
        sigma_x_func=sigma_x_func,                          # ← CHANGED
        num_params_fitted=num_params_fitted,
        eval_data=eval_data
    )

    f_test_results = run_f_test(
        experiments_file_path=experiments_file_path,
        script_or_func=script_or_func,
        sigma_x_func=sigma_x_func,                          # ← CHANGED
        num_params_fitted=num_params_fitted,
        num_params_reduced=num_params_reduced,
        rss_reduced=rss_reduced,
        eval_data=eval_data
    )

    ks_results = run_ks_test(
        experiments_file_path=experiments_file_path,
        script_or_func=script_or_func,
        sigma_x_func=sigma_x_func,                          # ← CHANGED
        eval_data=eval_data
    )

    lr_results = run_likelihood_ratio_test(
        experiments_file_path=experiments_file_path,
        script_or_func=script_or_func,
        sigma_x_func=sigma_x_func,                          # ← CHANGED
        num_params_fitted=num_params_fitted,
        num_params_reduced=num_params_reduced,
        log_likelihood_reduced=log_likelihood_reduced,
        eval_data=eval_data
    )

    chow_results = run_chow_test(
        experiments_file_path=experiments_file_path,
        script_or_func=script_or_func,
        sigma_x_func=sigma_x_func,                          # ← CHANGED
        num_params_fitted=num_params_fitted,
        split_ratio=split_ratio,
        eval_data=eval_data
    )

    ic_results = run_ic_evaluation(
        eval_data=eval_data,
        num_params_fitted=num_params_fitted
    )

    # 3. Consolidate results
    stat_results = {
        **chi2_results,
        **f_test_results,
        **ks_results,
        **lr_results,
        **chow_results,
        "aic": ic_results["aic"],
        "bic": ic_results["bic"]
    }

    # 4. Save results to trial directory root
    trial_dir = os.path.dirname(os.path.abspath(experiments_file_path))
    if os.path.basename(trial_dir) == "measurements":
        trial_dir = os.path.dirname(trial_dir)

    stat_file_path = os.path.join(trial_dir, "latest_stat_validation.json")
    try:
        with open(stat_file_path, "w") as sf:
            json.dump(stat_results, sf, indent=4)
    except Exception as e:
        print(f"[Warning] Failed to write latest_stat_validation.json: {e}")

    return stat_results
