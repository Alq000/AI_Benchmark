import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import chi2

def compute_ensemble_chi_squared(experiments_file_path, discovered_eq_func, calc_const_noise, calc_lin_noise, num_params_fitted):
    """
    Computes pooled Chi-Squared goodness-of-fit across a batch of trajectories.
    Saves the inputs and statistical metrics to 'latest_stat_validation.json'.

    :param experiments_file_path: Path to the all_compiled_experiments.json file
    :param discovered_eq_func: Python function f(t, y) representing the ODE system
    :param calc_const_noise: Float value of constant noise calculated by the agent
    :param calc_lin_noise: Float value of linear noise calculated by the agent
    :param num_params_fitted: Number of parameters estimated in the discovered equation
    :return: dict with chi2 statistic, degrees of freedom, reduced chi2, p-value, and status
    """
    with open(experiments_file_path, "r") as f:
        data = json.load(f)

    total_chi2 = 0.0
    total_points = 0

    c_noise = float(calc_const_noise)
    l_noise = float(calc_lin_noise)

    for run_key, run_data in data.items():
        for traj in run_data.get("trajectories", []):
            ic = traj["initial_conditions"]
            x0 = ic["x0"]
            v0 = ic["v0"]

            measurements = traj["measurements"]
            t_arr = np.array([m["t"] for m in measurements])
            x_obs = np.array([m["x_measured"] for m in measurements])
            v_obs = np.array([m["v_measured"] for m in measurements])

            if len(t_arr) == 0:
                continue

            sol = solve_ivp(
                fun=discovered_eq_func,
                t_span=(t_arr[0], t_arr[-1]),
                y0=[x0, v0],
                t_eval=t_arr,
                method='RK45'
            )

            if not sol.success or len(sol.y[0]) != len(t_arr):
                continue

            x_exp = sol.y[0]
            sigma_x = c_noise + l_noise * np.abs(v_obs)

            residuals_x = (x_obs - x_exp) / sigma_x
            total_chi2 += np.sum(residuals_x ** 2)
            total_points += len(t_arr)

    dof = max(1, total_points - num_params_fitted)
    reduced_chi2 = total_chi2 / dof if dof > 0 else float('inf')
    p_val = float(chi2.sf(total_chi2, dof))

    stat_results = {
        "calc_const_noise": c_noise,
        "calc_lin_noise": l_noise,
        "chi2_statistic": round(float(total_chi2), 4),
        "degrees_of_freedom": int(dof),
        "reduced_chi2": round(float(reduced_chi2), 4),
        "p_value": p_val,
        "null_hypothesis_accepted": bool(0.5 <= reduced_chi2 <= 2.0)
    }

    # Save results to trial directory root
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
