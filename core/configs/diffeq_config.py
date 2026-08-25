import numpy as np
from scipy.stats.qmc import LatinHypercube

# =========================================================================
# 1. TOP-LEVEL CONFIGURATION FLAGS & TRAJECTORY SETTINGS
# =========================================================================
ALLOW_CUSTOM_INITIAL_CONDITIONS = False
VARY_PARAMS = False
NUM_TRAJECTORIES = 100
MAX_TURNS = 40

# =========================================================================
# 2. ENVIRONMENT SCHEMA & PARAMETER SPACE
# =========================================================================
ENV_SCHEMA = {
    "t": {"description": "Independent variable representing time.", "range": (0.0, 50.0)},
    "x": {"description": "Dependent state variable: displacement.", "range": (-10.0, 10.0)},
    "v": {"description": "Dependent state variable: velocity (dx/dt).", "range": (-20.0, 20.0)},
}

COEFF_RANGES = {
    "k_0": (-10.0, -1.0),
    "k_1": (-3.0, -0.0),
}

# =========================================================================
# 3. TERM LIBRARY
# =========================================================================
TERM_LIBRARY = {
    "k_0": lambda **kwargs: kwargs.get('x'),
    "k_1": lambda **kwargs: kwargs.get('v'),
}

def generate_trial_coefficients(num_trials, seed=42, grouping=None):
    param_names = list(COEFF_RANGES.keys())

    if grouping is None or grouping <= 0:
        dimensions = len(param_names)
        sampler = LatinHypercube(d=dimensions, seed=seed)
        samples = sampler.random(n=num_trials)
        trial_list = []
        for i in range(num_trials):
            low_k0, high_k0 = COEFF_RANGES["k_0"]
            k0 = float(low_k0 + samples[i, 0] * (high_k0 - low_k0))
            k1_crit = -2.0 * np.sqrt(-k0)
            low_k1 = max(COEFF_RANGES["k_1"][0], k1_crit)
            high_k1 = COEFF_RANGES["k_1"][1]
            k1 = float(low_k1 + samples[i, 1] * (high_k1 - low_k1))
            trial_list.append({"k_0": k0, "k_1": k1})
        return trial_list

    num_groups = max(1, int(round(num_trials / grouping)))
    low_k0, high_k0 = COEFF_RANGES["k_0"]

    if num_groups == 1:
        k0_values = np.array([(low_k0 + high_k0) / 2.0])
    else:
        k0_values = np.linspace(low_k0, high_k0, num_groups)

    base_pts = num_trials // num_groups
    rem = num_trials % num_groups
    group_counts = [base_pts + 1 if i < rem else base_pts for i in range(num_groups)]

    k0_all = []
    for k0_val, count in zip(k0_values, group_counts):
        k0_all.extend([float(k0_val)] * count)

    rng = np.random.default_rng(seed)
    high_k1 = COEFF_RANGES["k_1"][1]
    k1_all = []
    for k0 in k0_all:
        k1_crit = -2.0 * np.sqrt(-k0)
        low_k1 = max(COEFF_RANGES["k_1"][0], k1_crit)
        k1_val = rng.uniform(low_k1, high_k1)
        k1_all.append(float(k1_val))

    trial_list = []
    for i in range(num_trials):
        trial_list.append({"k_0": float(k0_all[i]), "k_1": float(k1_all[i])})

    return trial_list

# =========================================================================
# SYSTEMATIC BIASES & UNCERTAINTY PARAMETERS
# =========================================================================
SYSTEMATIC_BIAS = {
    "x_c_0": 0.4,
    "x_c_1": 0.1,
    "x_c_2": 0.05,
    "v_c_0": 0.2,
    "v_c_1": 0.02,
    "v_c_2": 0.08,
    "v_c_3": 0.05
}


def true_bias_x(x, v, t):
    """Returns the true systematic measurement bias for x at state (x, v, t)."""
    return SYSTEMATIC_BIAS["x_c_0"] + SYSTEMATIC_BIAS["x_c_1"] * t + SYSTEMATIC_BIAS["x_c_2"] * t * t


def true_bias_v(x, v, t):
    """Returns the true systematic measurement bias for v at state (x, v, t)."""
    return SYSTEMATIC_BIAS["v_c_0"] + SYSTEMATIC_BIAS["v_c_1"] * t + SYSTEMATIC_BIAS["v_c_2"] * t * t +  + SYSTEMATIC_BIAS["v_c_3"] * x 

DEFAULT_NOISE_CONFIG = {
    "sigma_0_x": 0.10,
    "sigma_1_x": 0.05,
    "sigma_0_v": 0.10,
    "sigma_1_v": 0.05,
    "alpha": 1.0,
    "beta": 1.0,
    "use_alpha_beta_grid": False
}

ALPHA_VALUES = [0.5, 2.0, 10.0]
BETA_VALUES = [0.5, 2.0, 10.0]

# =========================================================================
# SYSTEMATIC SIGMA (NOISE) FUNCTIONS
# =========================================================================
def true_sigma_x(x, v, t, noise_config=None):
    """Returns the true measurement noise standard deviation (sigma_x) at state (x, v, t)."""
    if noise_config is None:
        cfg = DEFAULT_NOISE_CONFIG
    else:
        cfg = {**DEFAULT_NOISE_CONFIG, **noise_config}

    x_arr = np.asarray(x, dtype=float)

    if cfg.get("use_alpha_beta_grid", False):
        alpha   = cfg["alpha"]
        beta    = cfg["beta"]
        sigma_0 = cfg["sigma_0_x"]
        sigma_1 = cfg["sigma_1_x"]
        variance = alpha * (sigma_0**2) + beta * (sigma_1**2) * (x_arr**2)
        return np.sqrt(np.maximum(variance, 1e-12))
    else:
        return cfg.get("meas_const_noise", 0.2) + cfg.get("meas_lin_noise", 0.2) * np.abs(x_arr)


def true_sigma_v(x, v, t, noise_config=None):
    """Returns the true measurement noise standard deviation (sigma_v) at state (x, v, t)."""
    if noise_config is None:
        cfg = DEFAULT_NOISE_CONFIG
    else:
        cfg = {**DEFAULT_NOISE_CONFIG, **noise_config}

    v_arr = np.asarray(v, dtype=float)

    if cfg.get("use_alpha_beta_grid", False):
        alpha   = cfg["alpha"]
        beta    = cfg["beta"]
        sigma_0 = cfg["sigma_0_v"]
        sigma_1 = cfg["sigma_1_v"]
        variance = alpha * (sigma_0**2) + beta * (sigma_1**2) * (v_arr**2)
        return np.sqrt(np.maximum(variance, 1e-12))
    else:
        return cfg.get("meas_const_noise", 0.2) + cfg.get("meas_lin_noise", 0.2) * np.abs(v_arr)


# =========================================================================
# UNIFIED UNCERTAINTY MODELS (epsilon_x and epsilon_v)
# =========================================================================
def true_epsilon_x(x, v, t, noise_config=None):
    x_arr = np.asarray(x, dtype=float)
    sigma = true_sigma_x(x_arr, v, t, noise_config=noise_config)
    stochastic = np.random.normal(0.0, sigma, size=x_arr.shape) if x_arr.ndim > 0 else np.random.normal(0.0, sigma)

    return true_bias_x(x_arr, v, t) + stochastic


def true_epsilon_v(x, v, t, noise_config=None):
    v_arr = np.asarray(v, dtype=float)
    sigma = true_sigma_v(x, v_arr, t, noise_config=noise_config)
    stochastic = np.random.normal(0.0, sigma, size=v_arr.shape) if v_arr.ndim > 0 else np.random.normal(0.0, sigma)

    return true_bias_v(x, v_arr, t) + stochastic

def apply_measurement_noise(result, noise_config, t=0.0):
    x_true, v_true = result[0], result[1]
    eps_x = true_epsilon_x(x_true, v_true, t, noise_config)
    eps_v = true_epsilon_v(x_true, v_true, t, noise_config)
    return [x_true + eps_x, v_true + eps_v]

# =========================================================================
# 6. DYNAMIC ENGINE
# =========================================================================
def hidden_diffeq(t, y, *args, coeffs=None):
    if coeffs is None:
        coeffs = TRUE_COEFFS

    x = y[0]
    v = y[1]

    extra_param_keys = [key for key in ENV_SCHEMA.keys() if key not in ['x', 'v', 't']]
    context = {'t': t, 'x': x, 'v': v}
    for i, key in enumerate(extra_param_keys):
        if i < len(args):
            context[key] = args[i]

    forcing = 0.0
    for term_name, formula in TERM_LIBRARY.items():
        if term_name in coeffs:
            forcing += coeffs[term_name] * formula(**context)

    return [v, forcing]

def calibration_diffeq(t, y, *args, coeffs=None):
    if coeffs is None:
        coeffs = getattr(globals(), 'TRUE_COEFFS', {})
    x  = y[0]
    v  = y[1]
    k0 = coeffs.get("k_0", -4.0)
    return [v, k0 * x]


DIFFEQ_TYPES = {
    "calibration": calibration_diffeq,
    "black_box":   hidden_diffeq
}
