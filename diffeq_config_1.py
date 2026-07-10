import numpy as np

# 1. THE ENVIRONMENT SCHEMA
ENV_SCHEMA = {
    "t": {"description": "Independent variable representing time.", "range": (0.0, 50.0)},
    "x": {"description": "Dependent state variable: displacement.", "range": (-10.0, 10.0)},
    "v": {"description": "Dependent state variable: velocity (dx/dt).", "range": (-20.0, 20.0)},
}

# 2. EXPERIMENT SETTINGS
NUM_TRAJECTORIES = 100  # Number of independent phase-space paths generated per experiment call

# 3. THE TERM LIBRARY
TERM_LIBRARY = {
    "k_0": lambda **kwargs: kwargs.get('x'),
    #"k_1": lambda **kwargs: kwargs.get('v'),
    #"k_2": lambda **kwargs: kwargs.get('v') * abs(kwargs.get('v')),
}

TRUE_COEFFS = {"k_0": -4.761, "k_1": 0}

# 4. NOISE ENGINE CONFIGURATIONS (Defaults)
DEFAULT_NOISE_CONFIG = {
    "input_const_noise": 0.002,     
    "input_lin_noise": 0.001,       
    "meas_const_noise": 0.01,       
    "meas_lin_noise": 0.005,        
}

# 5. SYSTEMATIC BIASES (Predictable physical instrument errors)
# Defaults for a perfectly calibrated neutral machine: Offsets = 0.0, Scales = 1.0
SYSTEMATIC_BIAS = {
    "measurement_x_offset": 0.05,  # e.g., +0.05 for a misaligned position sensor
    "measurement_v_offset": 0.05,  # e.g., -0.02 for a misaligned velocity tracker
    "actuator_scale_x": 1.0,      # e.g., 1.02 for a 2% physical overshoot on launch
    "actuator_scale_v": 1.0       # e.g., 0.98 for a 2% undershoot on launch
}

def add_noise_distribution(value, const_std, lin_std):
    """Applies a realistic normal distribution combining constant floor & linear scaling error."""
    std = const_std + (lin_std * abs(value))
    if std <= 0:
        return value
    return float(np.random.normal(value, std))

def apply_input_noise(pt, noise_override):
    """Transforms input parameter targets with systematic actuator bias and noise."""
    noisy_pt = pt.copy()
    
    # 1. Apply physical actuator scaling biases first
    if 'x' in noisy_pt: noisy_pt['x'] *= SYSTEMATIC_BIAS["actuator_scale_x"]
    if 'v' in noisy_pt: noisy_pt['v'] *= SYSTEMATIC_BIAS["actuator_scale_v"]
    
    # 2. Add statistical launch noise
    for key in noisy_pt:
        noisy_pt[key] = add_noise_distribution(
            noisy_pt[key], 
            noise_override["input_const_noise"], 
            noise_override["input_lin_noise"]
        )
    return noisy_pt

def apply_measurement_noise(result, noise_override):
    """Transforms precise solver outputs with systematic sensor drift and tracking noise."""
    # 1. Apply physical systematic sensor offsets
    biased_x = result[0] + SYSTEMATIC_BIAS["measurement_x_offset"]
    biased_v = result[1] + SYSTEMATIC_BIAS["measurement_v_offset"]
    
    # 2. Add statistical measurement noise floor
    noisy_x = add_noise_distribution(biased_x, noise_override["meas_const_noise"], noise_override["meas_lin_noise"])
    noisy_v = add_noise_distribution(biased_v, noise_override["meas_const_noise"], noise_override["meas_lin_noise"])
    return [noisy_x, noisy_v]

# 6. THE DYNAMIC ENGINE
def hidden_diffeq(t, y, *args, coeffs=TRUE_COEFFS):
    x = y[0]
    v = y[1]
    
    extra_param_keys = [key for key in ENV_SCHEMA.keys() if key not in ['x', 'v', 't']]
    
    context = {'t': t, 'x': x, 'v': v}
    for key, val in zip(extra_param_keys, args):
        context[key] = val

    dxdt = v
    dvdt = 0.0
    
    for key, coefficient in coeffs.items():
        if key in TERM_LIBRARY:
            dvdt += coefficient * TERM_LIBRARY[key](**context)
            
    return [dxdt, dvdt]
