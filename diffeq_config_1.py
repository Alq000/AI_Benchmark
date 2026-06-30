import numpy as np

# 1. THE ENVIRONMENT SCHEMA
ENV_SCHEMA = {
    "t": {"description": "Independent variable representing time.", "range": (0.0, 50.0)},
    "x": {"description": "Dependent state variable: displacement.", "range": (-10.0, 10.0)},
    "v": {"description": "Dependent state variable: velocity (dx/dt).", "range": (-20.0, 20.0)},
    "m": {"description": "Mass of the oscillator.", "range": (0.5, 3.0)} # Changed from voltage_gain to m for your example
}

# 2. THE TERM LIBRARY
TERM_LIBRARY = {
    "k_0": lambda **kwargs: kwargs.get('x'),
    "k_1": lambda **kwargs: kwargs.get('v'),
    "k_2": lambda **kwargs: kwargs.get('v')**2,
    "k_3": lambda **kwargs: kwargs.get('v') * np.sin(kwargs.get('x')),
}

# The true value we want the agent to find for k_0 is 4.761
TRUE_COEFFS = {"k_0": 4.761, "k_1": 0.123, "k_2": 0.282, "k_3": 0.006}

# 3. THE DYNAMIC ENGINE
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
