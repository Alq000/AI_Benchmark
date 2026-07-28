import sys
import json
import numpy as np
from pathlib import Path

def calculate_derivatives(t, v):
    """Calculates numerical acceleration from velocity using central differences safely."""
    a = np.zeros_like(v, dtype=float)
    if len(t) < 3:
        return a

    dt = np.diff(t)
    dv = np.diff(v)

    # Avoid zero or micro-dt steps
    dt_safe = np.where(np.abs(dt) < 1e-8, np.nan, dt)

    a[0] = dv[0] / dt_safe[0] if not np.isnan(dt_safe[0]) else 0.0
    a[-1] = dv[-1] / dt_safe[-1] if not np.isnan(dt_safe[-1]) else 0.0

    dt_central = t[2:] - t[:-2]
    dt_central_safe = np.where(np.abs(dt_central) < 1e-8, np.nan, dt_central)
    a[1:-1] = (v[2:] - v[:-2]) / dt_central_safe

    return a

def build_feature_library(x, v):
    """
    Builds the Theta matrix up to order 3.
    Returns the matrix and column index corresponding to linear 'x' term (k_0).
    """
    Theta = np.column_stack([
        np.ones_like(x),     # 0: Constant
        x,                   # 1: Linear x (k_0)
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
    trial_path = Path(trial_dir)
    measurements_file = trial_path / "measurements" / "all_compiled_experiments.json"
    summary_file = trial_path / "summary.json"

    if not measurements_file.exists() or not summary_file.exists():
        print(f"Error: Missing required files in {trial_dir}")
        sys.exit(1)

    with open(measurements_file, 'r') as f:
        data = json.load(f)

    all_x, all_v, all_a = [], [], []

    for run_key, run_data in data.items():
        for trajectory in run_data.get("trajectories", []):
            measurements = trajectory.get("measurements", [])
            if len(measurements) < 3:
                continue

            t = np.array([m['t'] for m in measurements], dtype=float)
            x = np.array([m['x_measured'] for m in measurements], dtype=float)
            v = np.array([m['v_measured'] for m in measurements], dtype=float)

            a = calculate_derivatives(t, v)

            all_x.extend(x)
            all_v.extend(v)
            all_a.extend(a)

    all_x = np.array(all_x, dtype=float)
    all_v = np.array(all_v, dtype=float)
    all_a = np.array(all_a, dtype=float)

    # Filter out invalid / numerical explosion values
    valid_mask = np.isfinite(all_x) & np.isfinite(all_v) & np.isfinite(all_a)
    all_x = all_x[valid_mask]
    all_v = all_v[valid_mask]
    all_a = all_a[valid_mask]

    if len(all_a) < 10:
        print(f"[Warning] Insufficient valid points ({len(all_a)}) in {trial_dir}")
        return

    Theta, k0_index = build_feature_library(all_x, all_v)

    # Feature scaling for matrix stability
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Theta_scaled = Theta / col_norms

    Theta_T_Theta_scaled = Theta_scaled.T @ Theta_scaled
    Info_Matrix_Inv_scaled = np.linalg.pinv(Theta_T_Theta_scaled, rcond=1e-10)

    c_optimal = (Info_Matrix_Inv_scaled @ (Theta_scaled.T @ all_a)) / col_norms

    a_predicted = Theta @ c_optimal
    residuals = all_a - a_predicted

    N = len(all_a)
    p = Theta.shape[1]

    sigma_epsilon_sq = np.sum(residuals**2) / max(N - p, 1)
    sigma_epsilon = np.sqrt(max(sigma_epsilon_sq, 0))

    # Scale variance weight back to original feature scale
    fisher_weight_k0 = Info_Matrix_Inv_scaled[k0_index, k0_index] / (col_norms[k0_index]**2)
    variance_k0 = sigma_epsilon_sq * fisher_weight_k0
    E_best_absolute = np.sqrt(max(variance_k0, 0))

    with open(summary_file, 'r') as f:
        summary_data = json.load(f)

    summary_data['empirical_uncertainty'] = {
        "best_case_k0_error_absolute": float(E_best_absolute),
        "residual_noise_sigma": float(sigma_epsilon),
        "fisher_information_k0_weight": float(fisher_weight_k0)
    }

    if 'true_k0' in summary_data and summary_data['true_k0'] != 0:
        true_k0 = summary_data['true_k0']
        E_best_relative = E_best_absolute / abs(true_k0)
        summary_data['empirical_uncertainty']['best_case_k0_error_relative'] = float(E_best_relative)

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"Successfully calculated empirical bounds for {trial_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_empirical_error.py <path_to_trial_directory>")
        sys.exit(1)

    calculate_best_error(sys.argv[1])
