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

def build_real_equation_string(config):
    terms = []
    # Map internal dictionary keys to their human-readable math equivalents
    var_map = {"k_0": "x", "k_1": "v", "k_2": "v|v|"}
    for k, v in config.TRUE_COEFFS.items():
        if v != 0:
            var_name = var_map.get(k, k)
            terms.append(f"{v}*{var_name}")
    return "x_ddot = " + " + ".join(terms) if terms else "x_ddot = 0"

def fit_sindy_coefficient(master_file_path, config):
    if not os.path.exists(master_file_path):
        return None, "No historical tracking file found."
    
    with open(master_file_path, "r") as f:
        database = json.load(f)
        
    X_data = []
    X_dot_data = []
    
    for run_key, run_data in database.items():
        dt = run_data.get("delta_t", 0.1)
        
        for traj in run_data.get("trajectories", []):
            measurements = traj.get("measurements", [])
            if len(measurements) < 3:
                continue
                
            x = np.array([m["x_measured"] for m in measurements])
            v = np.array([m["v_measured"] for m in measurements])
            
            v_dot = np.gradient(v, dt)
            
            for i in range(len(x)):
                X_data.append([x[i], v[i]])
                X_dot_data.append(v_dot[i])
                
    if len(X_data) < 10:
        return None, "Insufficient data matrix constraints."
        
    X_global = np.array(X_data)
    Y_global = np.array(X_dot_data)
    
    x_col = X_global[:, 0]
    v_col = X_global[:, 1]
    
    Theta = np.column_stack([
        np.ones_like(x_col),  
        x_col,                
        v_col,                
        x_col**2,             
        x_col * v_col,        
        v_col**2,             
        x_col**3,             
        (x_col**2) * v_col,   
        x_col * (v_col**2),   
        v_col**3              
    ])
    
    feature_labels = ["1", "x", "v", "x^2", "x*v", "v^2", "x^3", "x^2*v", "x*v^2", "v^3"]

    try:
        alpha = 0.01
        coeffs = np.linalg.inv(Theta.T @ Theta + alpha * np.eye(Theta.shape[1])) @ Theta.T @ Y_global
        
        for _ in range(4):
            max_coeff = np.max(np.abs(coeffs))
            threshold = max(0.08 * max_coeff, 0.05) 
            
            small_indices = np.abs(coeffs) < threshold
            coeffs[small_indices] = 0.0
            remaining_indices = ~small_indices
            
            if not np.any(remaining_indices):
                break
                
            Theta_sparse = Theta[:, remaining_indices]
            sparse_coeffs = np.linalg.inv(Theta_sparse.T @ Theta_sparse + alpha * np.eye(Theta_sparse.shape[1])) @ Theta_sparse.T @ Y_global
            coeffs[remaining_indices] = sparse_coeffs

        equation_terms = []
        for c, lbl in zip(coeffs, feature_labels):
            if abs(c) > 1e-3:
                equation_terms.append(f"{c:+.4f}*{lbl}")
        
        formula_str = "x_ddot = " + " ".join(equation_terms) if equation_terms else "x_ddot = 0"
        predicted_k0 = float(coeffs[1])
        return predicted_k0, formula_str
    except Exception:
        return None, "Regression Error"

def calculate_kde_line(errors, x_span):
    if len(errors) < 2 or np.all(errors == errors[0]):
        return np.zeros_like(x_span)
    kde = gaussian_kde(errors, bw_method='scott')
    return kde(x_span)

def get_plot_bounds(errors, k_1, zoom_percentile):
    if zoom_percentile >= 100 or len(errors) < 2:
        return min(errors.min(), -k_1) * 1.4, max(errors.max(), k_1) * 1.4
    
    tail = (100 - zoom_percentile) / 2.0
    min_val = np.percentile(errors, tail)
    max_val = np.percentile(errors, 100 - tail)
    
    pad = (max_val - min_val) * 0.15
    if pad == 0: pad = 0.5
    
    return min_val - pad, max_val + pad

def main():
    ZOOM_PERCENTILE = 80

    parser = argparse.ArgumentParser(description="Generate Separated Agent and Blind SINDy Evaluation Output")
    parser.add_argument("--errors_path", type=str, required=True, help="Path to errors_log.json")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_path", type=str, required=True, help="Base path for Agent plot image")
    args = parser.parse_args()

    config = load_config(args.config)
    true_k0 = config.TRUE_COEFFS.get('k_0', -4.761)
    k_1 = abs(config.TRUE_COEFFS.get('k_1', -1.234))
    real_equation = build_real_equation_string(config)

    agent_errors = []
    if os.path.exists(args.errors_path):
        with open(args.errors_path, "r") as f:
            log_data = json.load(f)
        agent_errors = [entry["error"] for entry in log_data if "error" in entry and entry["error"] is not None]

    base_dir = os.path.dirname(args.errors_path)
    sindy_errors = []
    trials_path = os.path.join(base_dir, "trials")
    
    if os.path.exists(trials_path):
        for t_folder in sorted(os.listdir(trials_path)):
            trial_dir = os.path.join(trials_path, t_folder)
            history_file = os.path.join(trial_dir, "measurements", "all_compiled_experiments.json")
            if os.path.exists(history_file):
                k0_est, formula = fit_sindy_coefficient(history_file, config)
                if k0_est is not None:
                    sindy_errors.append(k0_est - true_k0)
                    with open(os.path.join(trial_dir, "sindy_report.json"), "w") as rf:
                        json.dump({"estimated_k0": k0_est, "signed_error": k0_est - true_k0, "discovered_governing_formula": formula}, rf, indent=4)
                        
    if not agent_errors: agent_errors = [0.0]
    if not sindy_errors: sindy_errors = [0.0]

    agent_errors = np.array(agent_errors)
    sindy_errors = np.array(sindy_errors)
    n_agent = len(agent_errors)
    n_sindy = len(sindy_errors)

    mean_agent, var_agent = np.mean(agent_errors), np.var(agent_errors)
    mean_sindy, var_sindy = np.mean(sindy_errors), np.var(sindy_errors)

    min_agent, max_agent = get_plot_bounds(agent_errors, k_1, ZOOM_PERCENTILE)
    zoomed_agent_errors = agent_errors[(agent_errors >= min_agent) & (agent_errors <= max_agent)]
    x_span_agent = np.linspace(min_agent - abs(min_agent)*0.5, max_agent + abs(max_agent)*0.5, 1000)

    # ---------------------------------------------------------
    # Plot 1: Agent Submissions
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(zoomed_agent_errors, bins='auto', density=True, color="#4682B4", edgecolor="black", alpha=0.55, label="Agent Submissions")
    ax.plot(x_span_agent, calculate_kde_line(agent_errors, x_span_agent), color="#1D4ED8", linewidth=2.5, 
            label=rf"Agent (KDE)\nMean ($\mu$): {mean_agent:.4f}\nVar ($\sigma^2$): {var_agent:.4f}")
    
    # Core target and threshold lines
    ax.axvline(x=0, color="green", linestyle=":", linewidth=2, label="True Baseline Target")
    
    # NEW: Dedicated line for the Agent's Mean
    ax.axvline(x=mean_agent, color="#D97706", linestyle="-", linewidth=2.5, label=f"Agent Mean Error ({mean_agent:.4f})")
    
    ax.axvline(x=k_1, color="purple", linestyle="-.", linewidth=1.5, label=f"+k_1 Threshold ({k_1:.3f})")
    ax.axvline(x=-k_1, color="purple", linestyle="-.", linewidth=1.5, label=f"-k_1 Threshold (-{k_1:.3f})")
    
    # NEW: Add textual context box summarizing analytical ground truth
    context_text = f"N = {n_agent} trials\nTrue Target k_0: {true_k0}\nReal Eq: {real_equation}"
    ax.text(0.02, 0.95, context_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    title_suffix = f" (Middle {ZOOM_PERCENTILE}%)" if ZOOM_PERCENTILE < 100 else ""
    ax.set_title(f"Agent Search Error Distribution Profile{title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Estimation Error Space", fontsize=11)
    ax.set_ylabel("Density Scaling Metrics", fontsize=11)
    
    ax.set_xlim(min_agent, max_agent)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=120)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: SINDy Benchmark
    # ---------------------------------------------------------
    sindy_only_path = args.output_path.replace(".png", "_sindy_pure.png")
    min_sindy, max_sindy = get_plot_bounds(sindy_errors, k_1, ZOOM_PERCENTILE)
    zoomed_sindy_errors = sindy_errors[(sindy_errors >= min_sindy) & (sindy_errors <= max_sindy)]
    x_span_sindy = np.linspace(min_sindy - abs(min_sindy)*0.5, max_sindy + abs(max_sindy)*0.5, 1000)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    if np.max(zoomed_sindy_errors) == np.min(zoomed_sindy_errors):
        ax2.hist(zoomed_sindy_errors, bins=[-0.5, 0.5], density=True, color="#10B981", edgecolor="black", alpha=0.6, label="SINDy Bins")
    else:
        ax2.hist(zoomed_sindy_errors, bins='auto', density=True, color="#10B981", edgecolor="black", alpha=0.6, label="SINDy Bins")
        
    ax2.plot(x_span_sindy, calculate_kde_line(sindy_errors, x_span_sindy), color="#047857", linewidth=2.5, 
            label=rf"SINDy Model Discovery (KDE)\nMean ($\mu$): {mean_sindy:.4f}\nVar ($\sigma^2$): {var_sindy:.4f}")
            
    ax2.axvline(x=0, color="green", linestyle=":", linewidth=2, label="True Baseline Target")
    ax2.axvline(x=mean_sindy, color="#D97706", linestyle="-", linewidth=2.5, label=f"SINDy Mean Error ({mean_sindy:.4f})")
    ax2.axvline(x=k_1, color="purple", linestyle="-.", linewidth=1.5, label=f"+k_1 Threshold ({k_1:.3f})")
    ax2.axvline(x=-k_1, color="purple", linestyle="-.", linewidth=1.5, label=f"-k_1 Threshold (-{k_1:.3f})")
    
    context_text_sindy = f"N = {n_sindy} trajectories parsed\nTrue Target k_0: {true_k0}\nReal Eq: {real_equation}"
    ax2.text(0.02, 0.95, context_text_sindy, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    ax2.set_title(f"Blind SINDy Equation Derivation Error Space{title_suffix}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Relative Parameter Error Matrix", fontsize=11)
    ax2.set_ylabel("Density Scaling Metrics", fontsize=11)
    
    ax2.set_xlim(min_sindy, max_sindy)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(sindy_only_path, dpi=120)
    plt.close()

if __name__ == "__main__":
    main()
