import numpy as np

def run_kinematic_rts_smoother(t, x_obs, v_obs, sigma_x, sigma_v):
    """
    Rauch-Tung-Striebel (RTS) Kalman smoother using the fixed kinematic relationship \dot{x} = v.
    Cleanly filters noise from both x and v simultaneously.
    """
    n = len(t)
    if n == 0:
        return {"x_smooth": np.array([]), "v_smooth": np.array([])}

    # State vector: [x, v]^T
    X_f = np.zeros((n, 2))  # Forward filtered states
    P_f = np.zeros((n, 2, 2)) # Forward covariances
    
    X_s = np.zeros((n, 2))  # Smoothed states
    P_s = np.zeros((n, 2, 2)) # Smoothed covariances

    # Measurement matrix (observing both x and v)
    H = np.eye(2)
    R = np.diag([sigma_x**2, sigma_v**2])

    # Initial state
    X_f[0] = [x_obs[0], v_obs[0]]
    P_f[0] = R.copy()

    # Forward Pass (Kalman Filter)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        F = np.array([[1, dt], [0, 1]]) # State transition
        Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * (sigma_v * 0.1)**2 # Process noise heuristic

        # Predict
        x_pred = F @ X_f[i-1]
        P_pred = F @ P_f[i-1] @ F.T + Q

        # Update
        z = np.array([x_obs[i], v_obs[i]])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        X_f[i] = x_pred + K @ y
        P_f[i] = (np.eye(2) - K @ H) @ P_pred

    # Backward Pass (RTS Smoother)
    X_s[-1] = X_f[-1]
    P_s[-1] = P_f[-1]

    for i in range(n-2, -1, -1):
        dt = t[i+1] - t[i]
        F = np.array([[1, dt], [0, 1]])

        P_pred = F @ P_f[i] @ F.T + Q
        C = P_f[i] @ F.T @ np.linalg.inv(P_pred)

        X_s[i] = X_f[i] + C @ (X_s[i+1] - (F @ X_f[i]))
        P_s[i] = P_f[i] + C @ (P_s[i+1] - P_pred) @ C.T

    return {
        "x_smooth": X_s[:, 0],
        "v_smooth": X_s[:, 1]
    }
