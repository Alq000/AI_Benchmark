import json

def get_system_prompt(env_schema):
    # Dynamically extract parameters the agent is allowed to query
    allowed_params = [k for k in env_schema.keys() if k not in ['x', 'v']]
    
    return f"""You are an expert physicist and AI scientist tasked with identifying a hidden non-linear differential equation.
Your goal is to find the coefficient of the linear displacement term 'x' (often denoted as k_0).

You can run experiments by passing parameters to the system. 
The system accepts the following parameters for experiments: {allowed_params}

ENVIRONMENT SCHEMA:
{json.dumps(env_schema, indent=2)}

ACTIONS ALLOWED:
1. RUN EXPERIMENT: To get the displacement 'x' at specific points, output a JSON list inside XML tags.
<run_experiment>
[ {{"t": 2.5, "m": 1.2}}, {{"t": 5, "m": 1.2}} ]
</run_experiment>

2. RUN PYTHON: To do math, you can execute python code. Print your results.
<run_python>
import numpy as np
print(np.log(5))
</run_python>

3. SUBMIT: When you are confident in your terms, submit your final coefficients in SymPy format.
<submission>
[
  {{ "term": "x", "coeff": 4.761 }},
  {{ "term": "x_dot", "coeff": 0.123 }}
]
</submission>
"""

ERROR_PROMPT = "Your last output was formatted incorrectly or caused an error. Please ensure you strictly use the XML tags <run_experiment>, <run_python>, or <submission> with valid content."
