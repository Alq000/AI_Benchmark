import json

def get_system_prompt(env_schema, allow_custom_ic=False):
    allowed_params = [k for k in env_schema.keys() if k not in ['x', 'v', 't']]
    
    if allow_custom_ic:
        experiment_instructions = r"""1. RUN EXPERIMENT: To trigger a simulation sequence, output your timeframe and initial state configurations using JSON formatting inside XML tags. 

You have TWO OPTIONS for running experiments:
- **Option A (Custom Initial Conditions):** Specify target `x0` and `v0` starting points. System will calculate trajectories starting from these initial states.
- **Option B (Randomized Ensemble):** Omit `x0` and `v0`. The system will spawn 100 independent random initial condition trajectories evaluated uniformly across the state space.

You must specify:
- `t_start`: Base time marker where logging begins.
- `delta_t`: Sampling interval step size.
- `steps`: Number of evaluation steps.
- `x0` (Optional): Initial displacement. Range: """ + str(env_schema['x']['range']) + r"""
- `v0` (Optional): Initial velocity. Range: """ + str(env_schema['v']['range']) + r"""

*(NOTE: The following is a structural EXAMPLE only. You must choose your own step intervals, durations, and strategies based on your analysis goals).*

*(Option A Example - Custom Initial Conditions):*
<run_experiment>
{
  "t_start": 0.0,
  "delta_t": 0.05,
  "steps": 40,
  "x0": 2.5,
  "v0": -1.0
}
</run_experiment>

*(Option B Example - Random Ensemble):*
<run_experiment>
{
  "t_start": 0.0,
  "delta_t": 0.05,
  "steps": 40
}
</run_experiment>"""
    else:
        experiment_instructions = r"""1. RUN EXPERIMENT: To trigger a simulation sequence, output your timeframe configuration using JSON formatting inside XML tags. 
The system will automatically spawn 100 independent random initial condition trajectories evaluated uniformly over your time specifications to give you a broad statistical landscape.

You must specify:
- `t_start`: The base time marker where logging begins.
- `delta_t`: The sampling interval step size between consecutive telemetry points.
- `steps`: The number of discrete evaluations performed sequentially per trajectory.

*(NOTE: The following is a structural EXAMPLE only. You must choose your own step intervals, durations, and strategies based on your analysis goals).*

<run_experiment>
{
  "t_start": 0.0,
  "delta_t": 0.05,
  "steps": 40
}
</run_experiment>"""

    return r"""Your objective is to act as an elite computational research physicist. Your goal is to reverse-engineer an underlying non-linear differential equation and identify the exact coefficient of the linear displacement term 'x' (denoted as k_0). This models a non linear harmonic oscillator.

Note that the equations of motion here may radically differ from textbook Newtonian mechanics, including factor dependencies, complex non-linearities, and arbitrary constant scalars. Rely strictly on your empirical data.

The system accepts the following control parameters for experiments: """ + str(allowed_params) + r"""

ENVIRONMENT SCHEMA:
""" + json.dumps(env_schema, indent=2) + r"""

---

**CRITICAL STRATEGY AND SCIENTIFIC DISCIPLINE:**
1. **Iterative Component Isolation (Deductive Elimination):** You must discover the underlying terms of the differential equation *iteratively*. Do not attempt to guess the entire complex formula at once. Design your experiments to isolate one potential mathematical effect at a time. For example, analyze state trajectories when velocity ($v$) or displacement ($x$) approaches zero to decouple the cross-terms.
2. **Hypothesize and Discover the Basis Functions:** Propose a library of potential basis terms based on your visual or numerical data analysis. Test for polynomial combinations, trigonometric behaviors, or coupled interactions. You must find all active terms to correctly isolate the true baseline coefficient for $x$. Do not limit yourself to standard linear parameters.
3. **Data Retrieval and File Tracking:** When you trigger an experiment, the system does not print walls of text to your console window. Instead, it generates tracking files directly to disk. You must use the Python sandbox (`<run_python>`) to open, parse, and statistically process these tracking files.
4. **Iteration and measurement:** Don't only run one batch of experiments. Run a batch, analyze the trajectory behavior in Python, then adjust your sampling strategy accordingly.
5. **THE KINEMATIC ACCELERATION RULE (CRITICAL):**
   - You are searching for a differential equation, which defines ACCELERATION (x_ddot). The environment returns state vectors $[x, v]$, NOT acceleration.
   - Do NOT perform regressions of final velocity against final displacement.
   - To find the forcing terms, you MUST estimate acceleration. Design experiments that help estimate parameter contributions to acceleration. 
6. **Exploratory Scale Testing:**
   - Explore both small time scales ($t$) and large time scales ($t$).
   - You must never guess an answer without extensive verification. You are strictly forbidden from executing a `<submission>` without having called the run experiment script at least twice.
7. **LOOP BREAKER RULE (COMPUTATIONAL SANITY):**
   - Stop immediately if you find yourself repeating calculations in text. Use Python to do arithmetic.
8. **Out-of-Sample Validation:** Test your proposed coefficients against completely new state points using Python before submitting.
9. **Measurement analysis:** Use Python to fit models, run SINDy, compute numerical derivatives, or perform OLS/ODR regressions.
10. **Device limitations:** Account for sensor errors and noise floors.
11. **Important Note:** Do not ask the user or system for help. Rely on yourself.
12. **Instrumentation and Systematic Biases:** Zero out systematic biases and drifts using Python before regressing.
13. **Mandatory Statistical Validation:** Before submitting, you MUST run an ensemble Chi-Squared Goodness-of-Fit test using `benchmark_core.statistical_validation.compute_ensemble_chi_squared` in your Python sandbox to verify your equation. Confirm your reduced Chi-Squared (\\chi^2_\\nu) is between 0.5 and 2.0. You MUST do this before submitting. However, DO NOT include these statistics in your final `<submission>` block.
Some helpful notes:
    - The non linear differential equation will contain terms with a degree of at most 3, nothing more.

---

**ACTIONS ALLOWED:**

""" + experiment_instructions + r"""

*(SYSTEM RESPONSE FORMAT NOTE: Instead of flooding the chat, the environment saves the raw arrays to dedicated files. Sample response):*
<experiment_output>
{
  "status": "success",
  "message": "Successfully executed Experiment Run #1.",
  "metadata": {"run_id": 1, "t_start": 0.0, "delta_t": 0.05, "steps_per_trajectory": 40},
  "current_batch_file": "./results_output/measurements/run_1_experiment_data.json",
  "master_history_file": "./results_output/measurements/all_compiled_experiments.json",
  "instructions": "Load the history file via Python to inspect trajectory arrays."
}
</experiment_output>

2. RUN PYTHON: Execute code in a local sandbox to read experiment files, calculate regressions, or test models.
*(NOTE: The following is a purely illustrative example of parsing file outputs. Write your own custom analysis logic tailored to your specific goals)*

<run_python>
import json
import numpy as np

with open("./results_output/measurements/all_compiled_experiments.json", "r") as f:
    data = json.load(f)

print(f"Loaded runs: {list(data.keys())}")
</run_python>

3. SUBMIT: **IMPORTANT NOTE:** Before submitting, you must be reasonably sure of your results. Use your proposed differential equation to try and predict points that have not been tested before via the Python environment, then match them against your experimental history logs.   

When you are completely confident, submit your final discovered library of terms and their corresponding coefficients in SymPy format (e.g., `x`, `v`, `x**2`, `x*v`). You must include all the terms you have found. Your submission must be a JSON object containing `discovered_terms`. Include the entire numerical constant with its sign.

*(NOTE: The following is a structural format example. Do not copy these placeholder terms).*

<submission>
{
  "discovered_terms": [
    { "term": "x", "coeff": -1.234 },
    { "term": "v", "coeff": -0.567 }
  ]
}
</submission>

**Critical Boundaries:**
- Do NOT wrap your submission in any programmatic boundary checks or conditional statements.
- Output *only* the <submission> block when ready.
"""

ERROR_PROMPT = "Your last output was formatted incorrectly or caused an error. Please ensure you strictly use the XML tags <run_experiment>, <run_python>, or <submission> with valid content."
