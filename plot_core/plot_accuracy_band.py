import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_k0_prediction(submission_data):
    """Safely extracts the k_0 ('x') coefficient, handling dicts, lists, and JSON strings."""
    if not submission_data:
        return None
        
    if isinstance(submission_data, str):
        try:
            subs = json.loads(submission_data)
        except Exception:
            return None
    else:
        subs = submission_data

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
    parser = argparse.ArgumentParser(description="Plot True vs Predicted Accuracy with Error Band")
    parser.add_argument("--submissions_path", type=str, required=True, help="Path to submissions_log.json")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the plot")
    parser.add_argument("--band_style", type=str, choices=["straight", "tip-to-tip"], default="straight", 
                        help="Style of error band boundary lines")
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

    # Global Residual Statistics
    residuals = predicted_k0s - true_k0s
    mean_residual = np.mean(residuals)
    std_residuals = np.std(residuals, ddof=1) if N > 1 else 0.0
    sem = std_residuals / np.sqrt(N) if N > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 8))
    
    min_val = min(np.min(true_k0s), np.min(predicted_k0s))
    max_val = max(np.max(true_k0s), np.max(predicted_k0s))
    padding = abs(max_val - min_val) * 0.1 if max_val != min_val else 1.0
    span = np.linspace(min_val - padding, max_val + padding, 100)

    # Reference Diagonal Line (y = x)
    ax.plot(span, span, 'k--', linewidth=2, label="Perfect Accuracy ($y=x$)")
    
    # Scatter Individual Trials
    ax.scatter(true_k0s, predicted_k0s, color='darkorange', edgecolor='black', s=50, alpha=0.7, label=f"Trials (N={N})")

    # Grouped Error Band & Trend Lines
    df = pd.DataFrame({'true_k0': true_k0s, 'predicted_k0': predicted_k0s})
    df['group_k0'] = df['true_k0'].round(5)
    stats = df.groupby('group_k0')['predicted_k0'].agg(['mean', 'std', 'count']).reset_index()
    stats['std'] = stats['std'].fillna(0.0)
    stats['upper'] = stats['mean'] + stats['std']
    stats['lower'] = stats['mean'] - stats['std']
    
    if len(stats) > 1:
        x_vals = stats['group_k0'].values
        
        # Error Bars (always point at raw std values)
        ax.errorbar(x_vals, stats['mean'], yerr=stats['std'], fmt='none', ecolor='red', 
                    elinewidth=1.5, capsize=4, label=r"Group Error Bars ($\pm 1 \sigma$)")

        if args.band_style == "straight":
            # Linear Fit for Boundaries (Straight lines through graph)
            m_mean, b_mean = np.polyfit(x_vals, stats['mean'], 1)
            m_upper, b_upper = np.polyfit(x_vals, stats['upper'], 1)
            m_lower, b_lower = np.polyfit(x_vals, stats['lower'], 1)
            
            fit_mean = m_mean * x_vals + b_mean
            fit_upper = m_upper * x_vals + b_upper
            fit_lower = m_lower * x_vals + b_lower
            
            ax.plot(x_vals, fit_mean, 'b--', linewidth=2, label="Linear Mean Trend")
            ax.fill_between(x_vals, fit_lower, fit_upper, color='red', alpha=0.15, label="Linear Error Band")
            ax.plot(x_vals, fit_upper, 'r--', linewidth=1, alpha=0.8)
            ax.plot(x_vals, fit_lower, 'r--', linewidth=1, alpha=0.8)
        else:
            # Tip-to-Tip direct connection
            ax.plot(x_vals, stats['mean'], 'b-o', linewidth=2, label="Group Mean Trend")
            ax.fill_between(x_vals, stats['lower'], stats['upper'], color='red', alpha=0.15, label="Tip-to-Tip Error Band")
            ax.plot(x_vals, stats['upper'], 'r--', linewidth=1, alpha=0.8)
            ax.plot(x_vals, stats['lower'], 'r--', linewidth=1, alpha=0.8)

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
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(args.output_path, dpi=300)
    print(f"[Plotting Engine] Accuracy band plot generated with {N} points at: {args.output_path}")

if __name__ == "__main__":
    main()
