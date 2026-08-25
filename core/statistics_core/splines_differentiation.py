import numpy as np
import os
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def compute_spline_derivatives(t, x, smoothing_factor=None, num_dense_points=500, save_dir=None, plot_title="Spline Derivatives"):
    """
    Fits a smoothing spline to x(t), analytically computes derivatives, 
    and returns both discrete and dense continuous location data.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    
    # Ensure chronological sorting
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    x = x[sort_idx]

    # Fit UnivariateSpline and its derivatives
    spline = UnivariateSpline(t, x, s=smoothing_factor)
    spline_v = spline.derivative(n=1)
    spline_a = spline.derivative(n=2)

    # 1. Discrete evaluations at exact input timestamps
    x_smooth = spline(t)
    v_smooth = spline_v(t)
    a_smooth = spline_a(t)

    # 2. High-resolution dense evaluations for smooth continuous plotting
    t_dense = np.linspace(t.min(), t.max(), num_dense_points)
    x_dense = spline(t_dense)
    v_dense = spline_v(t_dense)
    a_dense = spline_a(t_dense)

    result = {
        # Discrete outputs (aligned with input 't')
        "x_smooth": x_smooth,
        "v_smooth": v_smooth,
        "a_smooth": a_smooth,
        
        # Dense continuous location data for high-res plotting
        "t_dense": t_dense,
        "x_dense": x_dense,
        "v_dense": v_dense,
        "a_dense": a_dense,
        
        # Spline metadata & callable object
        "knots": spline.get_knots(),
        "spline_obj": spline,
        "image_path": None
    }

    # Plot generation
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"{plot_title.replace(' ', '_').lower()}.png")
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        # Position plot (dense spline curve over discrete scatter points)
        axes[0].scatter(t, x, s=15, label="Observed Data", color="black", alpha=0.6)
        axes[0].plot(t_dense, x_dense, label="Spline x(t)", color="blue", linewidth=2)
        axes[0].set_ylabel("Position (x)")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Velocity plot
        axes[1].plot(t_dense, v_dense, label="Spline v(t) = x'(t)", color="orange", linewidth=2)
        axes[1].set_ylabel("Velocity (v)")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        # Acceleration plot
        axes[2].plot(t_dense, a_dense, label="Spline a(t) = x''(t)", color="green", linewidth=2)
        axes[2].set_ylabel("Acceleration (a)")
        axes[2].set_xlabel("Time (t)")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend()

        plt.suptitle(plot_title)
        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        plt.close(fig)
        
        result["image_path"] = image_path

    return result
