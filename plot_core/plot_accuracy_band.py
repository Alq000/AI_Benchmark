import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_k0_prediction(submission_data):
    """Safely extracts the k_0 ('x') coefficient, handling both dicts, lists, and JSON strings."""
    if not submission_data:
        return None
        
    # If it was saved as a raw string, parse it.
    if isinstance(submission_data, str):
        try:
            subs = json.loads(submission_data)
        except Exception:
            return None
    else:
        subs = submission_data

    # Unwrap dictionary if top-level keys exist
    if isinstance(subs, dict):
        subs = subs.get("discovered_terms", [])

    try:
        for term_dict in subs:
            if isinstance(term_dict, dict) and term_dict.get("term") == "x":
                return float(term_dict.get("coeff"))
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot True vs Predicted Accuracy with SEM Band")
    parser.add_argument("--submissions_path", type=str, required=True, help="Path to submissions_log.json")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the plot")
    args = parser.parse_args()

    if not os.path.exists(args.submissions_path):
        print(f"[Plotting Error] Submissions file missing: {args.submissions_path}")
        return

    with open(args.submissions_path, "r") as f:
        try:
            submissions_log = json.load(f)
        except json.JSONDecodeError:
            print(f"[Plotting Error] Submissions file is corrupted or empty: {args.submissions_path}")
            return

    true_k0s = []
    predicted_k0s = []
    
    for entry in submissions_log:
        pred = extract_k0_prediction(entry.get("submission"))
        
        # Access the dynamic dictionary saved into the tracking logs
        true_coeffs = entry.get("true_coeffs", {})
        t_k0 = true_coeffs.get("k_0")
        
        if pred is not None and t_k0 is not None:
            predicted_k0s.append(pred)
            true_k0s.append(t_k0)

    N = len(predicted_k0s)
    if N == 0:
        print("[Plotting Error] Zero valid predicted vs true parameter pairs found. Skipping plot.")
        return

    true_k0s = np.array(true_k0s)
    predicted_k0s = np.array(predicted_k0s)
    
    residuals = predicted_k0s - true_k0s
    mean_residual = np.mean(residuals)
    std_residuals = np.std(residuals, ddof=1) if N > 1 else 0.0
    sem = std_residuals / np.sqrt(N) if N > 0 else 0.0
    ci_margin = 1.96 * sem

    fig, ax = plt.subplots(figsize=(8, 8))
    
    min_val = min(np.min(true_k0s), np.min(predicted_k0s))
    max_val = max(np.max(true_k0s), np.max(predicted_k0s))
    padding = abs(max_val - min_val) * 0.1 if max_val != min_val else 1.0
    span = np.linspace(min_val - padding, max_val + padding, 100)

    # Plot baseline reference and confidence bands along the full diagonal span
    ax.plot(span, span, 'k--', linewidth=2, label="Perfect Accuracy ($y=x$)")
    ax.fill_between(span, span + mean_residual - ci_margin, span + mean_residual + ci_margin, 
                    color='blue', alpha=0.15, label=rf"95% CI Band ($\pm {ci_margin:.4f}$)")
    
    ax.plot(span, span + mean_residual, 'b-', linewidth=1, alpha=0.5, label="Agent Bias Trend")
    ax.scatter(true_k0s, predicted_k0s, color='darkorange', edgecolor='black', s=50, alpha=0.7, label=f"Trials (N={N})")

    ax.set_title(r"Dynamic Accuracy Map: Predicted vs True $k_0$", fontsize=14, fontweight='bold')
    ax.set_xlabel("True Coefficient ($k_{true}$)", fontsize=12)
    ax.set_ylabel("Predicted Coefficient ($k_{pred}$)", fontsize=12)
    
    stats_text = (f"Residual $\\mu$: {mean_residual:.4f}\n"
                  f"Residual $\\sigma$: {std_residuals:.4f}\n"
                  f"SEM: {sem:.4f}\n"
                  f"Valid Trials: {N}")
                  
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)

    plt.tight_layout()
    
    # Ensure directory exists before saving
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(args.output_path, dpi=300)
    print(f"[Plotting Engine] Accuracy band plot generated with {N} points at: {args.output_path}")

if __name__ == "__main__":
    main()
