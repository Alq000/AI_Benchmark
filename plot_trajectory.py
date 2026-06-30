import argparse
import json
import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    parser = argparse.ArgumentParser(description="Standalone Fire-and-Forget Trajectory Plotter")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--x_init", type=float, required=True, help="Initial displacement x(0)")
    parser.add_argument("--v_init", type=float, required=True, help="Initial velocity v(0)")
    parser.add_argument("--t", type=float, required=True, help="Target measurement timestamp")
    parser.add_argument("--extra_params", type=str, default="{}", help="JSON string of extra environment args")
    parser.add_argument("--output_path", type=str, required=True, help="File path to save the generated figure")
    args = parser.parse_args()

    # Load configuration module dynamically
    config = load_config(args.config)
    extra_params = json.loads(args.extra_params)
    
    # Enforce horizontal window extending to 130% of target time t
    t_plot_max = max(args.t * 1.3, 1e-5)
    
    extra_keys = [k for k in config.ENV_SCHEMA.keys() if k not in ['x', 'v', 't']]
    args_tuple = tuple(extra_params.get(k, 0.0) for k in extra_keys)
    
    # Establish fine-grained timestamps for a high-fidelity trajectory curve
    t_eval = np.linspace(0, t_plot_max, 350)
    if args.t not in t_eval and args.t <= t_plot_max:
        t_eval = np.sort(np.append(t_eval, args.t))
        
    # Solve trajectory up to full 1.3x window
    sol = solve_ivp(config.hidden_diffeq, [0, t_plot_max], [args.x_init, args.v_init], args=args_tuple, t_eval=t_eval)
    
    # Locate exact target values at measurement duration t
    sol_at_t = solve_ivp(config.hidden_diffeq, [0, max(args.t, 1e-5)], [args.x_init, args.v_init], args=args_tuple)
    x_at_t = sol_at_t.y[0, -1]
    v_at_t = sol_at_t.y[1, -1]

    # Generate layout
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], label="Displacement x(t)", color="blue", linewidth=2)
    plt.plot(sol.t, sol.y[1], label="Velocity v(t)", color="orange", linewidth=1.5, linestyle="--")
    
    # Highlight the target measurement coordinate with visible markers
    plt.scatter([args.t], [x_at_t], color="red", s=100, zorder=5, label=f"Measured x({args.t:.2f}) = {x_at_t:.4f}")
    plt.scatter([args.t], [v_at_t], color="darkred", s=100, marker="X", zorder=5, label=f"Measured v({args.t:.2f}) = {v_at_t:.4f}")
    
    plt.title("Oscillator Phase Space Trajectory Tracking", fontsize=13, fontweight="bold")
    plt.xlabel("Time Duration (seconds)", fontsize=11)
    plt.ylabel("State Vector Metric Magnitudes", fontsize=11)
    
    # Construct conditions info text box overlay
    cond_text = f"Parameters & Scope:\n• target t = {args.t}s\n• x(0) = {args.x_init}\n• v(0) = {args.v_init}"
    for param_key, param_val in extra_params.items():
        cond_text += f"\n• {param_key} = {param_val}"
        
    plt.text(0.02, 0.05, cond_text, transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9, edgecolor="silver"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    
    # Ensure nested subdirectories are initialized safely
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
