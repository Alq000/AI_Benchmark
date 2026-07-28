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
    parser = argparse.ArgumentParser(description="Standalone Trajectory Plotter for Sequential Sampling")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--x_init", type=float, required=True, help="Initial displacement x(0)")
    parser.add_argument("--v_init", type=float, required=True, help="Initial velocity v(0)")
    parser.add_argument("--t_start", type=float, required=True, help="Target measurement start timestamp")
    parser.add_argument("--t_end", type=float, required=True, help="Target measurement end timestamp")
    parser.add_argument("--measurements_json", type=str, default="[]", help="JSON string of sequential measurement dictionary")
    parser.add_argument("--output_path", type=str, required=True, help="File path to save the generated figure")
    args = parser.parse_args()

    config = load_config(args.config)
    measurements = json.loads(args.measurements_json)
    
    t_plot_max = max(args.t_end * 1.3, 1e-5)
    t_eval = np.linspace(0, t_plot_max, 350)
    
    sol = solve_ivp(config.hidden_diffeq, [0, t_plot_max], [args.x_init, args.v_init], t_eval=t_eval)

    plt.figure(figsize=(10, 5))
    
    # Pristine analytical line
    plt.plot(sol.t, sol.y[0], label="True Displacement x(t)", color="blue", linewidth=2)
    plt.plot(sol.t, sol.y[1], label="True Velocity v(t)", color="orange", linewidth=1.5, linestyle="--")
    
    # Overlay the corrupted tracking points
    if measurements:
        t_vals = [m["t"] for m in measurements]
        x_vals = [m["x_measured"] for m in measurements]
        v_vals = [m["v_measured"] for m in measurements]
        
        plt.scatter(t_vals, x_vals, color="red", s=40, edgecolors="black", zorder=6, label="Noisy Measured x(t)")
        plt.scatter(t_vals, v_vals, color="darkred", s=40, marker="X", edgecolors="black", zorder=6, label="Noisy Measured v(t)")

    plt.title("Laboratory Tracking: Clean Physics Line vs Noisy Batch Observations", fontsize=13, fontweight="bold")
    plt.xlabel("Time Duration (seconds)", fontsize=11)
    plt.ylabel("State Vector Metric Magnitudes", fontsize=11)
    
    cond_text = f"Agent Phase-Space Setup:\n• Intended x(0) = {args.x_init:.4f}\n• Intended v(0) = {args.v_init:.4f}"
    
    plt.text(0.02, 0.05, cond_text, transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9, edgecolor="silver"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    ax = plt.gca()
    data_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    data_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_aspect((data_width / data_height) / 2.0)
    plt.savefig(args.output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
