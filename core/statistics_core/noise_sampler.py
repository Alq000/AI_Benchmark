import numpy as np
from scipy.stats import gaussian_kde

def build_non_parametric_sampler(residuals):
    """
    Constructs a non-parametric empirical KDE noise sampler from calibration residuals.
    """
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    if len(residuals) < 2:
        raise ValueError("Not enough residuals to build KDE.")

    kde = gaussian_kde(residuals)

    def sample_noise(n_samples=1):
        return kde.resample(n_samples)[0]

    return sample_noise

def run_conditional_noise_estimation(x_obs, x_exp):
    """
    Extracts raw residuals and builds a callable non-parametric stochastic sampler.
    """
    x_obs = np.asarray(x_obs)
    x_exp = np.asarray(x_exp)
    residuals = x_obs - x_exp
    
    sampler = build_non_parametric_sampler(residuals)
    
    return {
        "residuals_mean": float(np.mean(residuals)),
        "residuals_std": float(np.std(residuals)),
        "sampler_func": sampler
    }
