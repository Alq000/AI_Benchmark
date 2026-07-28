import argparse
import json
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def fit_sindy_coefficient(master_file_path):
    if not os.path.exists(master_file_path):
        return None
    with open(master_file_path, "r") as f:
        database = json.load(f)
    X_data, X_dot_data = [], []
    for _, run_data in database.items():
        dt = run_data.get("delta_t", 0.1)
        for traj in run_data.get("trajectories", []):
            measurements = traj.get("measurements", [])
            if len(measurements) < 3: continue
            x = np.array([m["x_measured"] for m in measurements])
            v = np.array([m["v_measured"] for m in measurements])
            v_dot = np.gradient(v, dt)
            for i in range(len(x)):
                X_data.append([x[i], v[i]])
                X_dot_data.append(v_dot[i])
    if len(X_data) < 10: return None
    X_global, Y_global = np.array(X_data), np.array(X_dot_data)
    x_col, v_col = X_global[:, 0], X_global[:, 1]
    Theta = np.column_stack([
        np.ones_like(x_col), x_col, v_col, x_col**2, x_col * v_col,
        v_col**2, x_col**3, (x_col**2) * v_col, x_col * (v_col**2), v_col**3
    ])
    try:
        alpha = 0.01
        coeffs = np.linalg.inv(Theta.T @ Theta + alpha * np.eye(Theta.shape[1])) @ Theta.T @ Y_global
        for _ in range(4):
            threshold = max(0.08 * np.max(np.abs(coeffs)), 0.05)
            small_indices = np.abs(coeffs) < threshold
            coeffs[small_indices] = 0.0
            remaining = ~small_indices
            if not np.any(remaining): break
            Theta_sparse = Theta[:, remaining]
            coeffs[remaining] = np.linalg.inv(Theta_sparse.T @ Theta_sparse + alpha * np.eye(Theta_sparse.shape[1])) @ Theta_sparse.T @ Y_global
        return float(coeffs[1])
    except Exception:
        return None

def calculate_kde_line(data, x_span):
    if len(data) < 2 or np.all(data == data[0]):
        return np.zeros_like(x_span)
    kde = gaussian_kde(data, bw_method='scott')
    return kde(x_span)

def get_plot_bounds(errors, zoom_percentile):
    if zoom_percentile >= 100 or len(errors) < 2:
        return min(errors.min(), -1.0) * 1.2, max(errors.max(), 1.0) * 1.2
    
    tail = (100 - zoom_percentile) / 2.0
    min_val = np.percentile(errors, tail)
    max_val = np.percentile(errors, 100 - tail)
    
    pad = (max_val - min_val) * 0.15
    if pad == 0: pad = 0.5
    
    return min_val - pad, max_val + pad

def main():
    ZOOM_PERCENTILE = 95

    parser = argparse.ArgumentParser(description="Graph Relative Difference Between SINDy Error and Agent Error")
    parser.add_argument("--errors_path", type=str, required=True, help="Path to errors_log.json")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated plot")
    parser.add_argument("--vary_params", action="store_true", help="Varying parameter flag")
    args = parser.parse_args()

    config = load_config(args.config)
    true_k0_default = config.TRUE_COEFFS.get('k_0', -4.761)

    base_dir = os.path.dirname(args.errors_path)
    trials_path = os.path.join(base_dir, "trials")

    trials_data = []
    if os.path.exists(args.errors_path):
        with open(args.errors_path, "r") as f:
            log_data = json.load(f)
        for entry in log_data:
            t_num = entry.get("trial")
            agent_err = entry.get("error")
            if t_num is None or agent_err is None: continue
            
            trial_dir = os.path.join(trials_path, f"trial_{t_num}")
            history_file = os.path.join(trial_dir, "measurements", "all_compiled_experiments.json")
            
            t_k0 = true_k0_default
            coeffs_path = os.path.join(trial_dir, "true_coeffs.json")
            if os.path.exists(coeffs_path):
                with open(coeffs_path, "r") as cf:
                    t_k0 = json.load(cf).get("k_0", true_k0_default)

            sindy_k0 = fit_sindy_coefficient(history_file)
            if sindy_k0 is not None:
                sindy_err = sindy_k0 - t_k0
                eps = 1e-9 if sindy_err == 0 else 0
                rel_diff = (sindy_err - agent_err) / (sindy_err + eps)
                trials_data.append(rel_diff)

    if not trials_data:
        print("[Plotting Error] No valid matched SINDy vs Agent trial data found.")
        return

    rel_diffs = np.array(trials_data)
    N = len(rel_diffs)
    mean_diff, var_diff = np.mean(rel_diffs), np.var(rel_diffs)
    std_diff, median_diff = np.std(rel_diffs), np.median(rel_diffs)

    min_val, max_val = get_plot_bounds(rel_diffs, ZOOM_PERCENTILE)
    zoomed_diffs = rel_diffs[(rel_diffs >= min_val) & (rel_diffs <= max_val)]
    x_span = np.linspace(min_val, max_val, 1000)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(zoomed_diffs, bins='auto', density=True, color="#93C5FD", edgecolor="black", alpha=0.6, label="Distribution Bins")
    ax.plot(x_span, calculate_kde_line(rel_diffs, x_span), color="#1E40AF", linewidth=2.5, label="KDE Density Profile")
    
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Diff (Equal Performance)")
    ax.axvline(x=mean_diff, color="#D97706", linestyle="-", linewidth=2, label=rf"Mean Diff ($\mu={mean_diff:.4f}$)")

    context_text = (f"N = {N} trials\n"
                    rf"Mean ($\mu$): {mean_diff:.4f}" + "\n"
                    rf"Variance ($\sigma^2$): {var_diff:.4f}" + "\n"
                    rf"Std Dev ($\sigma$): {std_diff:.4f}" + "\n"
                    f"Median: {median_diff:.4f}")

    ax.text(0.02, 0.95, context_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    title_suffix = f" (Middle {ZOOM_PERCENTILE}%)" if ZOOM_PERCENTILE < 100 else ""
    ax.set_title(rf"SINDy vs Agent Error Relative Difference: $\frac{{e_{{\mathrm{{sindy}}}} - e_{{\mathrm{{agent}}}}}}{{e_{{\mathrm{{sindy}}}}}}${title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Difference Magnitude", fontsize=11)
    ax.set_ylabel("Density Scaling Metrics", fontsize=11)
    
    ax.set_xlim(min_val, max_val)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()
    print(f"[Plotting Engine] SINDy vs Agent relative diff plot generated at: {args.output_path}")

if __name__ == "__main__":
    main()
