import numpy as np
from scipy.stats.qmc import LatinHypercube

# =========================================================================
# 1. TOP-LEVEL CONFIGURATION FLAGS & TRAJECTORY SETTINGS
# =========================================================================
ALLOW_CUSTOM_INITIAL_CONDITIONS = False  # Toggle whether agent can specify x0, v0
VARY_PARAMS = False                      # Default setting for varying trial coefficients
NUM_TRAJECTORIES = 100                  # Number of trajectories sampled per experiment run
MAX_TURNS = 40
# =========================================================================
# 2. ENVIRONMENT SCHEMA & PARAMETER SPACE
# =========================================================================
ENV_SCHEMA = {
    "t": {"description": "Independent variable representing time.", "range": (0.0, 50.0)},
    "x": {"description": "Dependent state variable: displacement.", "range": (-10.0, 10.0)},
    "v": {"description": "Dependent state variable: velocity (dx/dt).", "range": (-20.0, 20.0)},
}

# Fallback for single-run evaluation
TRUE_COEFFS = {"k_0": -4.761, "k_1": -1.234}

COEFF_RANGES = {
    "k_0": (-10.0, -1.0),   # Linear displacement range
    "k_1": (-5.0, 5.0),     # Velocity damping range
}

# =========================================================================
# 3. TERM LIBRARY
# =========================================================================
TERM_LIBRARY = {
    "k_0": lambda **kwargs: kwargs.get('x'),
    "k_1": lambda **kwargs: kwargs.get('v'),
}

def generate_trial_coefficients(num_trials, seed=42):
    """
    Generates a space-filling lattice of coefficients using Latin Hypercube Sampling.
    """
    param_names = list(COEFF_RANGES.keys())
    dimensions = len(param_names)
    
    sampler = LatinHypercube(d=dimensions, seed=seed)
    samples = sampler.random(n=num_trials)
    
    trial_list = []
    for i in range(num_trials):
        trial_coeffs = {}
        for d, name in enumerate(param_names):
            low, high = COEFF_RANGES[name]
            trial_coeffs[name] = float(low + samples[i, d] * (high - low))
        trial_list.append(trial_coeffs)
        
    return trial_list

# =========================================================================
# 4. NOISE ENGINE CONFIGURATIONS
# =========================================================================
DEFAULT_NOISE_CONFIG = {
    "input_const_noise": 0.002,     
    "input_lin_noise": 0.001,       
    "meas_const_noise": 0.01,       
    "meas_lin_noise": 0.005,        
}

# =========================================================================
# 5. SYSTEMATIC BIASES & NOISE FUNCTIONS
# =========================================================================
SYSTEMATIC_BIAS = {
    "measurement_x_offset": 0.045,   
    "measurement_v_offset": -0.022,  
    "actuator_scale_x": 0.982,       
    "actuator_scale_v": 1.015        
}

def add_noise_distribution(value, const_noise, lin_noise):
    if isinstance(value, (list, np.ndarray)):
        value = np.array(value)
        sigma = const_noise + lin_noise * np.abs(value)
        return value + np.random.normal(0, sigma, size=value.shape)
    else:
        sigma = const_noise + lin_noise * np.abs(value)
        return value + np.random.normal(0, sigma)

def apply_actuator_bias(requested_pt, noise_override):
    noisy_pt = requested_pt.copy()
    noisy_pt['x'] *= SYSTEMATIC_BIAS["actuator_scale_x"]
    noisy_pt['v'] *= SYSTEMATIC_BIAS["actuator_scale_v"]
    for key in noisy_pt:
        noisy_pt[key] = add_noise_distribution(
            noisy_pt[key], 
            noise_override["input_const_noise"], 
            noise_override["input_lin_noise"]
        )
    return noisy_pt

def apply_measurement_noise(result, noise_override):
    biased_x = result[0] + SYSTEMATIC_BIAS["measurement_x_offset"]
    biased_v = result[1] + SYSTEMATIC_BIAS["measurement_v_offset"]
    noisy_x = add_noise_distribution(biased_x, noise_override["meas_const_noise"], noise_override["meas_lin_noise"])
    noisy_v = add_noise_distribution(biased_v, noise_override["meas_const_noise"], noise_override["meas_lin_noise"])
    return [noisy_x, noisy_v]

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
