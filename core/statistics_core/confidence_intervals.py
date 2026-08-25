import numpy as np
from scipy.stats import norm, t

def compute_confidence_intervals(params, jacobian, residuals, noise_std=None, jacobian_pred=None):
    """
    Computes parameter standard errors, 1-sigma and 2-sigma confidence bounds,
    optional trajectory prediction bands, and residual significance metrics.

    Parameters
    ----------
    params : array-like, shape (P,)
        Fitted model parameter vector \hat{\theta}.
    jacobian : array-like, shape (N, P)
        Jacobian matrix of model evaluations with respect to parameters: J_ij = d(y_i)/d(\theta_j).
    residuals : array-like, shape (N,)
        Residual vector r = y_obs - y_pred.
    noise_std : float or array-like, shape (N,), optional
        Known measurement uncertainties \sigma. If None, estimated from residual variance s^2.
    jacobian_pred : array-like, shape (M, P), optional
        Jacobian evaluated on a dense time/evaluation grid for computing curve confidence bands.

    Returns
    -------
    dict
        Contains covariance matrix, parameter standard errors, 1-sigma and 2-sigma intervals,
        residual significance (z-scores), and optional spatial prediction bands.
    """
    params = np.asarray(params, dtype=float)
    J = np.asarray(jacobian, dtype=float)
    r = np.asarray(residuals, dtype=float)
    
    N, P = J.shape
    dof = N - P

    if dof <= 0:
        raise ValueError(f"Insufficient degrees of freedom (N={N}, P={P}). Need N > P.")

    # 1. Compute Asymptotic Covariance Matrix
    if noise_std is not None:
        sigma = np.asarray(noise_std, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(N, sigma)
        W = np.diag(1.0 / (sigma ** 2))
        # Cov = (J^T * W * J)^(-1)
        JT_W = J.T @ W
        JT_W_J = JT_W @ J
        cov_matrix = np.linalg.pinv(JT_W_J)
    else:
        # Unscaled residuals: variance estimate s^2 = RSS / (N - P)
        rss = np.sum(r ** 2)
        s_sq = rss / dof
        JT_J = J.T @ J
        cov_matrix = s_sq * np.linalg.pinv(JT_J)

    # 2. Parameter Standard Errors
    param_se = np.sqrt(np.maximum(0.0, np.diag(cov_matrix)))

    # 3. Z-scores / Multipliers for 1-sigma (68.27%) and 2-sigma (95.45%)
    # Use Student-t distribution for small samples (dof < 30), standard normal otherwise
    if dof < 30:
        z_1sigma = t.ppf((1 + 0.682689492137086) / 2, dof)
        z_2sigma = t.ppf((1 + 0.954499736103642) / 2, dof)
    else:
        z_1sigma = 1.0  # 68.27%
        z_2sigma = 2.0  # 95.45%

    param_bounds_1sigma = np.column_stack((params - z_1sigma * param_se, params + z_1sigma * param_se))
    param_bounds_2sigma = np.column_stack((params - z_2sigma * param_se, params + z_2sigma * param_se))

    # 4. Residual Significance (Normalized residual z-scores)
    if noise_std is not None:
        effective_sigma = sigma
    else:
        effective_sigma = np.sqrt(np.sum(r ** 2) / dof)

    residual_z_scores = r / effective_sigma
    within_1sigma_ratio = float(np.mean(np.abs(residual_z_scores) <= 1.0))
    within_2sigma_ratio = float(np.mean(np.abs(residual_z_scores) <= 2.0))
    within_3sigma_ratio = float(np.mean(np.abs(residual_z_scores) <= 3.0))

    # 5. Continuous Prediction Confidence Bands (Linearized Error Propagation)
    pred_band_1sigma = None
    pred_band_2sigma = None
    if jacobian_pred is not None:
        J_pred = np.asarray(jacobian_pred, dtype=float)
        # \sigma_pred^2 = diag(J_pred * Cov * J_pred^T)
        pred_var = np.sum((J_pred @ cov_matrix) * J_pred, axis=1)
        pred_se = np.sqrt(np.maximum(0.0, pred_var))
        
        pred_band_1sigma = z_1sigma * pred_se
        pred_band_2sigma = z_2sigma * pred_se

    return {
        "covariance_matrix": cov_matrix,
        "param_se": param_se,
        "param_bounds_1sigma": param_bounds_1sigma,
        "param_bounds_2sigma": param_bounds_2sigma,
        "residual_z_scores": residual_z_scores,
        "within_1sigma_ratio": within_1sigma_ratio,
        "within_2sigma_ratio": within_2sigma_ratio,
        "within_3sigma_ratio": within_3sigma_ratio,
        "pred_band_1sigma": pred_band_1sigma,
        "pred_band_2sigma": pred_band_2sigma
    }
