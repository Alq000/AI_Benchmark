import numpy as np

def compute_information_criteria(rss, n_samples, num_params):
    """
    Applies objective penalties for additional parameter complexity against residual errors.
    """
    if n_samples <= 0 or rss <= 0:
        return {"aic": np.nan, "bic": np.nan}

    # Log-likelihood estimation from RSS assuming Gaussian errors
    log_L = -0.5 * n_samples * np.log(rss / n_samples)
    
    aic = 2 * num_params - 2 * log_L
    bic = num_params * np.log(n_samples) - 2 * log_L

    return {
        "aic": float(aic),
        "bic": float(bic),
        "rss": float(rss),
        "n_samples": int(n_samples),
        "num_params": int(num_params)
    }

def run_ic_evaluation(eval_data, num_params_fitted):
    """
    Wrapper to extract data from evaluate_trajectories dictionary.
    """
    rss = eval_data.get("total_rss_full", np.nan)
    n_samples = eval_data.get("total_points", 0)
    
    return compute_information_criteria(rss, n_samples, num_params_fitted)
