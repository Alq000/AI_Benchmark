import argparse
import json
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    parser = argparse.ArgumentParser(description="Plot Error Distribution Histogram")
    parser.add_argument("--errors_path", type=str, required=True, help="Path to errors_log.json")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--output_path", type=str, required=True, help="File path to save the histogram image")
    args = parser.parse_args()

    # Load config to extract k_1
    config = load_config(args.config)
    k_1 = config.TRUE_COEFFS.get('k_1', 0.0)
    
    # Read the error log data
    if not os.path.exists(args.errors_path):
        print(f"Error: Log file not found at {args.errors_path}")
        return
        
    with open(args.errors_path, "r") as f:
        log_data = json.load(f)
        
    errors = [entry["error"] for entry in log_data if "error" in entry and entry["error"] is not None]
    
    plt.figure(figsize=(10, 6))
    
    if errors:
        # Generate the histogram distribution
        plt.hist(errors, bins=15, color="skyblue", edgecolor="black", alpha=0.7, label="Trial Errors")
        # Compute the value that encompasses 90% of the data points
        percentile_90 = np.percentile(errors, 90)
    else:
        plt.text(0.5, 0.5, "No successful trial data available to plot.", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        percentile_90 = 0.0

    # Draw explicit baseline validation lines at k_1 and -k_1
    plt.axvline(x=k_1, color="red", linestyle="--", linewidth=2, label=f"k_1 ({k_1:.4f})")
    plt.axvline(x=-k_1, color="darkred", linestyle="--", linewidth=2, label=f"-k_1 ({-k_1:.4f})")

    # Set boundaries containing both threshold markers AND at least 90% of data points
    x_min = min(-abs(k_1), 0.0)
    x_max = max(abs(k_1), percentile_90)
    padding = max((x_max - x_min) * 0.1, 0.1) # Add a 10% margin padding
    
    plt.xlim(x_min - padding, x_max + padding)
    
    plt.title("Distribution of Absolute Estimation Errors: abs(k_pred - k_0)", fontsize=12, fontweight="bold")
    plt.xlabel("Absolute Error Value Magnitude", fontsize=11)
    plt.ylabel("Frequency Count (Trials)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    
    # Save the plot securely
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()
    print(f"Histogram successfully updated and saved to: {args.output_path}")

if __name__ == "__main__":
    main()
