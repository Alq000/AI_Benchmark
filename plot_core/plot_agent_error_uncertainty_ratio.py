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

def calculate_kde_line(data, x_span):
    if len(data) < 2 or np.all(data == data[0]):
        return np.zeros_like(x_span)
    kde = gaussian_kde(data, bw_method='scott')
    return kde(x_span)

def get_plot_bounds(ratios, zoom_percentile):
    if zoom_percentile >= 100 or len(ratios) < 2:
        return 0, max(ratios.max(), 1.5) * 1.2
    
    max_val = np.percentile(ratios, zoom_percentile)
    pad = max_val * 0.15
    if pad == 0: pad = 0.5
    
    return 0, max_val + pad

def extract_numeric_uncertainty(data):
    if data is None:
        return None
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, dict):
        # 1. Target nested empirical_uncertainty dictionary first
        if "empirical_uncertainty" in data and isinstance(data["empirical_uncertainty"], dict):
            emp = data["empirical_uncertainty"]
            for k in ["best_case_k0_error_absolute", "best_case_k0_error_relative", "uncertainty"]:
                if k in emp and isinstance(emp[k], (int, float)):
                    return float(emp[k])
        
        # 2. Check direct keys while avoiding agent error ("error")
        for k in ["best_case_k0_error_absolute", "best_case_k0_error_relative", "uncertainty", "std_dev"]:
            if k in data and isinstance(data[k], (int, float)):
                return float(data[k])
    return None

def main():
    ZOOM_PERCENTILE = 95

    parser = argparse.ArgumentParser(description="Graph Ratio of Agent Error and Empirical Uncertainty")
    parser.add_argument("--errors_path", type=str, required=True, help="Path to errors_log.json")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated plot")
    parser.add_argument("--vary_params", action="store_true", help="Varying parameter flag")
    args = parser.parse_args()

    base_dir = os.path.dirname(args.errors_path)
    trials_path = os.path.join(base_dir, "trials")

    ratios = []

    if os.path.exists(args.errors_path):
        with open(args.errors_path, "r") as f:
            log_data = json.load(f)
        for entry in log_data:
            t_num = entry.get("trial")
            agent_err = entry.get("error")
            if t_num is None or agent_err is None: continue
            
            trial_dir = os.path.join(trials_path, f"trial_{t_num}")
            summary_path = os.path.join(trial_dir, "summary.json")
            empirical_path = os.path.join(trial_dir, "empirical_error.json")
            
            uncertainty = None
            for p in [summary_path, empirical_path]:
                if os.path.exists(p):
                    with open(p, "r") as sf:
                        s_data = json.load(sf)
                        uncertainty = extract_numeric_uncertainty(s_data)
                        if uncertainty is not None: break

            if uncertainty is not None and uncertainty > 0:
                ratio = abs(agent_err) / uncertainty
                ratios.append(ratio)

    if not ratios:
        print("[Plotting Error] No valid agent error to empirical uncertainty ratios computed.")
        return

    ratios = np.array(ratios)
    N = len(ratios)
    mean_ratio, var_ratio = np.mean(ratios), np.var(ratios)
    std_ratio, median_ratio = np.std(ratios), np.median(ratios)
    within_noise_pct = np.mean(ratios <= 1.0) * 100.0

    min_val, max_val = get_plot_bounds(ratios, ZOOM_PERCENTILE)
    zoomed_ratios = ratios[(ratios >= min_val) & (ratios <= max_val)]
    x_span = np.linspace(min_val, max_val, 1000)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(zoomed_ratios, bins='auto', density=True, color="#A7F3D0", edgecolor="black", alpha=0.6, label="Ratio Bins")
    ax.plot(x_span, calculate_kde_line(ratios, x_span), color="#047857", linewidth=2.5, label="Ratio KDE")
    
    ax.axvline(x=1.0, color="crimson", linestyle="--", linewidth=2, label="Threshold Line (Ratio = 1.0)")
    ax.axvline(x=mean_ratio, color="#D97706", linestyle="-", linewidth=2, label=rf"Mean Ratio ($\mu={mean_ratio:.4f}$)")

    context_text = (f"N = {N} trials\n"
                    rf"Mean ($\mu$): {mean_ratio:.4f}" + "\n"
                    rf"Variance ($\sigma^2$): {var_ratio:.4f}" + "\n"
                    rf"Std Dev ($\sigma$): {std_ratio:.4f}" + "\n"
                    f"Median: {median_ratio:.4f}\n"
                    f"Within Noise Floor: {within_noise_pct:.1f}%")

    ax.text(0.02, 0.95, context_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    title_suffix = f" (Middle {ZOOM_PERCENTILE}%)" if ZOOM_PERCENTILE < 100 else ""
    ax.set_title(rf"Agent Error to Empirical Uncertainty Ratio Distribution Profile: $\frac{{|e_{{\mathrm{{agent}}}}|}}{{\sigma_{{\mathrm{{empirical}}}}}}${title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Error / Uncertainty Ratio Magnitude", fontsize=11)
    ax.set_ylabel("Density Scaling Metrics", fontsize=11)
    
    ax.set_xlim(min_val, max_val)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()
    print(f"[Plotting Engine] Agent error to empirical uncertainty ratio plot generated at: {args.output_path}")

if __name__ == "__main__":
    main()
