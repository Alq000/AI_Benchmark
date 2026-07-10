import sys
import json
import numpy as np
from pathlib import Path

def calculate_derivatives(t, v):
    """Calculates numerical acceleration from velocity using central differences."""
    a = np.zeros_like(v)
    dt = np.diff(t)
    dv = np.diff(v)
    
    # Forward difference for the first point
    a[0] = dv[0] / dt[0]
    # Backward difference for the last point
    a[-1] = dv[-1] / dt[-1]
    # Central difference for the interior
    a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
    
    return a

def build_feature_library(x, v):
    """
    Builds the Theta matrix up to order 3.
    Returns the matrix and the column index corresponding to the linear 'x' term (k_0).
    """
    # Columns: [1, x, v, x^2, x*v, v^2, x^3, x^2*v, x*v^2, v^3]
    Theta = np.column_stack([
        np.ones_like(x),     # 0: Constant
        x,                   # 1: Linear x (This is k_0)
        v,                   # 2: Linear v
        x**2,                # 3: x^2
        x * v,               # 4: xv
        v**2,                # 5: v^2
        x**3,                # 6: x^3
        (x**2) * v,          # 7: x^2 v
        x * (v**2),          # 8: x v^2
        v**3                 # 9: v^3
    ])
    
    k0_index = 1 
    return Theta, k0_index

def calculate_best_error(trial_dir):
    """
    Reads the agent's sampled trajectories, computes the CRLB for k_0, 
    and updates the summary.json file.
    """
    trial_path = Path(trial_dir)
    measurements_file = trial_path / "measurements" / "all_compiled_experiments.json"
    summary_file = trial_path / "summary.json"

    if not measurements_file.exists() or not summary_file.exists():
        print(f"Error: Missing required files in {trial_dir}")
        sys.exit(1)

    # 1. Load the agent's experimental data
    with open(measurements_file, 'r') as f:
        data = json.load(f)

    all_x = []
    all_v = []
    all_a = []

    # Parse all nested trajectories to build a massive dataset
    for run_key, run_data in data.items():
        for trajectory in run_data.get("trajectories", []):
            measurements = trajectory.get("measurements", [])
            
            # Need at least 3 points to calculate central difference derivatives
            if len(measurements) < 3:
                continue
                
            # Extract lists from the list of measurement dictionaries
            t = np.array([m['t'] for m in measurements])
            x = np.array([m['x_measured'] for m in measurements])
            v = np.array([m['v_measured'] for m in measurements])
            
            a = calculate_derivatives(t, v)
            
            all_x.extend(x)
            all_v.extend(v)
            all_a.extend(a)

    all_x = np.array(all_x)
    all_v = np.array(all_v)
    all_a = np.array(all_a)
    
    # [Continue to Step 2. Build the Feature Library (Theta)...]
    # 2. Build the Feature Library (Theta)
    Theta, k0_index = build_feature_library(all_x, all_v)
    
    # 3. Calculate Information Matrix and Optimal Fit
    # Using pseudoinverse (pinv) to handle poorly conditioned matrices 
    # if the agent's sampling was physically highly restricted.
    Theta_T_Theta = Theta.T @ Theta
    Information_Matrix_Inv = np.linalg.pinv(Theta_T_Theta)
    
    c_optimal = Information_Matrix_Inv @ Theta.T @ all_a

    # 4. Residual Analysis (Noise Floor)
    a_predicted = Theta @ c_optimal
    residuals = all_a - a_predicted
    
    N = len(all_a)
    p = Theta.shape[1]
    
    # Variance of the residuals (sigma^2)
    sigma_epsilon_sq = np.sum(residuals**2) / (N - p)
    sigma_epsilon = np.sqrt(sigma_epsilon_sq)

    # 5. Calculate Best Reasonable Error (Standard Error of k_0)
    variance_k0 = sigma_epsilon_sq * Information_Matrix_Inv[k0_index, k0_index]
    
    # Absolute baseline error
    E_best_absolute = np.sqrt(max(variance_k0, 0)) 

    # 6. Update the summary.json
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)

    summary_data['empirical_uncertainty'] = {
        "best_case_k0_error_absolute": E_best_absolute,
        "residual_noise_sigma": sigma_epsilon,
        "fisher_information_k0_weight": Information_Matrix_Inv[k0_index, k0_index]
    }

    # If the true k0 is in the summary, we can calculate the relative best error bound
    if 'true_k0' in summary_data and summary_data['true_k0'] != 0:
        true_k0 = summary_data['true_k0']
        E_best_relative = E_best_absolute / abs(true_k0)
        summary_data['empirical_uncertainty']['best_case_k0_error_relative'] = E_best_relative

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"Successfully calculated empirical bounds for {trial_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_empirical_error.py <path_to_trial_directory>")
        sys.exit(1)
    
    calculate_best_error(sys.argv[1])
