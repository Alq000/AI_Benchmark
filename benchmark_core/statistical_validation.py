import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import chi2

def compute_ensemble_chi_squared(experiments_file_path, discovered_eq_func, noise_config, num_params_fitted):
    """
    Computes pooled Chi-Squared goodness-of-fit across a batch of trajectories.
    
    :param experiments_file_path: Path to the all_compiled_experiments.json file
    :param discovered_eq_func: Python function f(t, y) representing the ODE system
    :param noise_config: Dict containing 'meas_const_noise' and 'meas_lin_noise'
    :param num_params_fitted: Number of parameters estimated in the discovered equation
    :return: dict with chi2 statistic, degrees of freedom, reduced chi2, and p-value
    """
    with open(experiments_file_path, "r") as f:
        data = json.load(f)

    total_chi2 = 0.0
    total_points = 0
    
    c_noise = noise_config.get("meas_const_noise", 0.01)
    l_noise = noise_config.get("meas_lin_noise", 0.005)

    for run_key, run_data in data.items():
        for traj in run_data.get("trajectories", []):
            # Extract specific initial conditions
            ic = traj["initial_conditions"]
            x0 = ic["x0"]
            v0 = ic["v0"]
            
            # Extract measurements
            measurements = traj["measurements"]
            t_arr = np.array([m["t"] for m in measurements])
            x_obs = np.array([m["x_measured"] for m in measurements])
            v_obs = np.array([m["v_measured"] for m in measurements])
            
            if len(t_arr) == 0:
                continue

            # Simulate expected trajectory
            sol = solve_ivp(
                fun=discovered_eq_func,
                t_span=(t_arr[0], t_arr[-1]),
                y0=[x0, v0],
                t_eval=t_arr,
                method='RK45'
            )
            
            if not sol.success or len(sol.y[0]) != len(t_arr):
                continue  # Skip failed integrations
                
            x_exp = sol.y[0]
            
            # Compute point-by-point sigma
            sigma_x = c_noise + l_noise * np.abs(v_obs)
            
            # Accumulate weighted squared residuals
            residuals_x = (x_obs - x_exp) / sigma_x
            total_chi2 += np.sum(residuals_x ** 2)
            total_points += len(t_arr)

    # Calculate statistics
    dof = max(1, total_points - num_params_fitted)
    reduced_chi2 = total_chi2 / dof if dof > 0 else float('inf')
    p_val = float(chi2.sf(total_chi2, dof))

    return {
        "chi2_statistic": round(float(total_chi2), 4),
        "degrees_of_freedom": int(dof),
        "reduced_chi2": round(float(reduced_chi2), 4),
        "p_value": p_val,
        "null_hypothesis_accepted": bool(0.5 <= reduced_chi2 <= 2.0)
    }
