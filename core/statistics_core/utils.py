import json
import numpy as np
import os
import sys
import importlib.util
from core.statistics_core.trajectory_loader import load_trajectories

def load_predict_trajectory(script_or_func):
    """
    Retrieves predict_trajectory, accepting either a callable function or a file path string[cite: 18].
    """
    if callable(script_or_func):
        return script_or_func

    if isinstance(script_or_func, str):
        if not os.path.exists(script_or_func):
            raise FileNotFoundError(f"Agent script not found at '{script_or_func}'.")

        spec = importlib.util.spec_from_file_location("agent_discovered_model", script_or_func)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_discovered_model"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "predict_trajectory") or not callable(module.predict_trajectory):
            raise AttributeError(f"Script '{script_or_func}' must define a callable function named 'predict_trajectory(t_eval, x0, v0)'.")

        return module.predict_trajectory

    raise ValueError("`script_or_func` must be either a file path string or a callable function.")

def evaluate_trajectories(experiments_file_path, script_or_func, sigma_x_func, bias_x_func=None):
    """
    Loads experiment measurements and evaluates model predictions across all trajectories.
    Accepts either a script file path or a direct callable predict_trajectory function.
    sigma_x_func must be a callable with signature sigma_x(x, v, t) -> float or ndarray.
    """
    if not callable(sigma_x_func):                                               # ← NEW
        raise ValueError("`sigma_x_func` must be a callable, e.g. your submitted sigma_x(x, v, t).")  # ← NEW

    if bias_x_func is not None and not callable(bias_x_func):  # ← CHANGED: Validate bias_x_func if provided
            raise ValueError("`bias_x_func` must be a callable if provided, e.g. bias_x(x, v, t).")  # ← CHANGED

    predict_trajectory_func = load_predict_trajectory(script_or_func)

    # Use the new trajectory loader utility
    trajectories = load_trajectories(experiments_file_path)

    x_obs_list, x_exp_list, sigma_x_list = [], [], []
    total_rss_full = 0.0
    total_rss_mean_baseline = 0.0

    for traj in trajectories:
        t_arr = traj["t"]
        x_obs = traj["x"]
        v_obs = traj["v"]
        x0, v0 = traj["x0"], traj["v0"]

        try:
            x_exp, v_exp = predict_trajectory_func(t_arr, x0, v0)
        except Exception as e:
            print(f"[Warning] Trajectory prediction failed for IC (x0={x0}, v0={v0}): {e}")
            continue

        if x_exp is None or len(x_exp) != len(t_arr):
            continue

        x_exp = np.array(x_exp)
        # Subtract systematic bias if provided by the agent
        if bias_x_func is not None:  # ← CHANGED: Compute systematic bias prediction
            try:  # ← CHANGED
                raw_bias = bias_x_func(x_exp, v_exp, t_arr)  # ← CHANGED: Evaluate bias function
                bias_x = np.asarray(raw_bias, dtype=float)  # ← CHANGED
                if bias_x.ndim == 0:  # ← CHANGED
                    bias_x = np.full(len(x_obs), float(bias_x))  # ← CHANGED
            except Exception as e:  # ← CHANGED
                print(f"[Warning] bias_x_func evaluation failed: {e}")  # ← CHANGED
                bias_x = 0.0  # ← CHANGED: Fallback to zero bias if evaluation fails
        else:  # ← CHANGED
            bias_x = 0.0  # ← CHANGED: Default zero bias if no function supplied
        try:
            raw_sigma = sigma_x_func(x_obs, v_obs, t_arr)                       # ← CHANGED
            sigma_x = np.asarray(raw_sigma, dtype=float)                         # ← CHANGED
            if sigma_x.ndim == 0:                                                # ← CHANGED
                sigma_x = np.full(len(x_obs), float(sigma_x))                   # ← CHANGED
            if len(sigma_x) != len(x_obs):                                       # ← CHANGED
                raise ValueError(                                                 # ← CHANGED
                    f"sigma_x_func returned array of length {len(sigma_x)},"    # ← CHANGED
                    f" expected {len(x_obs)}."                                   # ← CHANGED
                )                                                                 # ← CHANGED
        except Exception as e:                                                    # ← CHANGED
            print(f"[Warning] sigma_x_func evaluation failed: {e}")             # ← CHANGED
            continue                                                              # ← CHANGED
        sigma_x = np.maximum(sigma_x, 1e-12)      

        x_obs_debiased = x_obs - bias_x  # ← CHANGED: Remove systematic bias from x_obs

        x_obs_list.extend(x_obs_debiased)
        x_exp_list.extend(x_exp)
        sigma_x_list.extend(sigma_x)

        raw_residuals = x_obs_debiased - x_exp
        total_rss_full += np.sum(raw_residuals ** 2)
        total_rss_mean_baseline += np.sum((x_obs_debiased - np.mean(x_obs_debiased)) ** 2)  # ← CHANGED: Baseline RSS using debiased observations
    return {
        "x_obs": np.array(x_obs_list),
        "x_exp": np.array(x_exp_list),
        "sigma_x": np.array(sigma_x_list),
        "total_rss_full": float(total_rss_full),
        "total_rss_mean_baseline": float(total_rss_mean_baseline),
        "total_points": len(x_obs_list)
    }

def compute_gaussian_log_likelihood(x_obs, x_exp, sigma_x):
    """
    Computes Gaussian log-likelihood given observations, predictions, and uncertainties[cite: 18].
    """
    x_obs = np.array(x_obs, dtype=float)
    x_exp = np.array(x_exp, dtype=float)
    sigma_x = np.array(sigma_x, dtype=float)

    n = len(x_obs)
    if n == 0:
        return 0.0

    residuals_sq = ((x_obs - x_exp) / sigma_x) ** 2
    log_likelihood = -0.5 * np.sum(residuals_sq + np.log(2 * np.pi * (sigma_x ** 2)))
    return float(log_likelihood)
