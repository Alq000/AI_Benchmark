import os
import sys
import json
import time
import re
import requests
import inspect
import importlib.util
import numpy as np
import subprocess
from scipy.integrate import solve_ivp
from configs import prompts

def call_openrouter_stream(messages, model, verbosity, openrouter_api_key):
    if model == "no_agent":
        assistant_turns = len([m for m in messages if m["role"] == "assistant"])
        if assistant_turns == 0:
            mock_output = (
                "Testing sandbox setup.\n"
                "<run_python>\n"
                "print('Python works')\n"
                "</run_python>"
            )
        elif assistant_turns == 1:
            mock_output = (
                "Sandbox works. Executing experiment run.\n"
                "<run_experiment>\n"
                "{\n"
                "  \"type\": \"calibration\",\n"
                "  \"t_start\": 10.0,\n"
                "  \"delta_t\": 0.05,\n"
                "  \"steps\": 3,\n"
                "  \"k_0\": -9.5\n"
                "}\n"
                "</run_experiment>"
            )
        elif assistant_turns == 2:
            mock_output = (
                "<run_experiment>\n"
                "{\n"
                "  \"type\": \"black_box\",\n"
                "  \"t_start\": 0.0,\n"
                "  \"delta_t\": 0.1,\n"
                "  \"steps\": 3,\n"
                "  \"x0\": 2.5,\n"
                "  \"v0\": -1.0\n"
                "}\n"
                "</run_experiment>"
            )
        elif assistant_turns == 3:
            mock_output = """
Testing sandbox setup.
<run_python>
    
import os
import json
import numpy as np
import sys
if '/app' not in sys.path:
    sys.path.append('/app')
from scipy.integrate import solve_ivp
from core.statistics_core.statistical_validation import compute_ensemble_chi_squared

def predict_trajectory(t_eval, x0, v0):
    def ode(t, y):
        x, v = y
        return [v, -4.761 * x - 5.0 * v]
    sol = solve_ivp(ode, (t_eval[0], t_eval[-1]), [x0, v0], t_eval=t_eval)
    return sol.y[0], sol.y[1]

def sigma_x(x, v, t):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(1e-12, (0.15**2) + ((0.075)**2)*x**2))

compute_ensemble_chi_squared(
    './results_output/measurements/all_compiled_experiments.json',
    predict_trajectory,
    sigma_x,
    num_params_fitted=2
)
    
</run_python>
"""
        else:
            mock_output = """
<submission>
import numpy as np
from scipy.integrate import solve_ivp

_C = np.array([
    6.76015874,
    -7.99199384,
    -2.01235976,
    0.0381498258,
    0.0321132571,
    0.00576032137,
    -0.0142088860,
    -0.0165074496,
    -0.00614334724,
    -0.000198909057
], dtype=float)

def _acceleration(x, v):
    return (
        _C[0] + _C[1]*x + _C[2]*v
        + _C[3]*x**2 + _C[4]*x*v + _C[5]*v**2
        + _C[6]*x**3 + _C[7]*x**2*v
        + _C[8]*x*v**2 + _C[9]*v**3
    )

def predict_trajectory(t_eval, x0, v0):
    t = np.asarray(t_eval, dtype=float).reshape(-1)

    if t.size == 0:
        raise ValueError("t_eval must be nonempty")
    if t.size == 1:
        return np.array([float(x0)]), np.array([float(v0)])

    sol = solve_ivp(
        lambda time, y: [y[1], _acceleration(y[0], y[1])],
        (float(t[0]), float(t[-1])),
        [float(x0), float(v0)],
        t_eval=t,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y[0].reshape(-1), sol.y[1].reshape(-1)

def bias_x(x, v, t):
    x, v, t = np.broadcast_arrays(x, v, t)
    return (
        0.801008767
        + 0.110607946900*t
    )

def bias_v(x, v, t):
    x, v, t = np.broadcast_arrays(x, v, t)
    return (
        0.198322760
        + 0.03*t
    )

def sigma_x(x, v, t):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(
        1e-12,
        ((0.1005**2) + ((0.05005)**2)*x**2)*2
    ))

def sigma_v(x, v, t):
    x, v = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(v, dtype=float)
    )
    return np.sqrt(np.maximum(
        1e-12,
        (0.15**2) + ((0.075)**2)*v**2
    ))

def epsilon_x(x, v, t):
    shape = np.broadcast(x, v, t).shape
    return bias_x(x, v, t) + np.random.normal(
        0.0,
        sigma_x(x, v, t),
        size=shape
    )

def epsilon_v(x, v, t):
    shape = np.broadcast(x, v, t).shape
    return bias_v(x, v, t) + np.random.normal(
        0.0,
        sigma_v(x, v, t),
        size=shape
    )
</submission>
            """
        if verbosity >= 2:
            print(mock_output, flush=True)
        return mock_output

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    data = {"model": model, "messages": messages, "stream": True}
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, stream=True)
    if not response.ok:
        print(f"\n[OpenRouter API Error] Status Code: {response.status_code}")
        print(f"[OpenRouter API Error] Response Body: {response.text}\n")
        response.raise_for_status() 
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                json_str = line[6:]
                if json_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if verbosity >= 2:
                                print(content, end="", flush=True)
                            full_response += content
                except json.JSONDecodeError:
                    pass
    if verbosity >= 2:
        print()
    return full_response


def run_trial(trial_id, config, model_id, verbosity, plotting_mode, config_path, measurements_dir, noise_config, dynamic_coeffs, openrouter_api_key, allow_custom_ic=False, max_turns=25):
    system_prompt = prompts.get_system_prompt(config.ENV_SCHEMA, allow_custom_ic=allow_custom_ic, max_turns=max_turns)
    messages = [{"role": "system", "content": system_prompt}]
    
    if verbosity == 4:
        print(f"\n{'#'*60}\n[SYSTEM PROMPT LOADED FOR TRIAL {trial_id}]\n{'#'*60}")
        print(system_prompt)
        print(f"{'#'*60}\n")
    
    run_id = 1
    calibration_run_id = 1
    trial_root_dir = os.path.dirname(os.path.abspath(measurements_dir))
    
    for turn in range(max_turns):
        if model_id != "no_agent":
            time.sleep(3)
        if verbosity >= 1:
            print(f"\n--- [Trial {trial_id} - Turn {turn}] Agent is thinking... ---")
        
        response = call_openrouter_stream(messages, model_id, verbosity, openrouter_api_key)
        messages.append({"role": "assistant", "content": response})

        sub_matches = re.findall(r"<submission>(.*?)</submission>", response, re.DOTALL)
        exp_matches = re.findall(r"<run_experiment>(.*?)</run_experiment>", response, re.DOTALL)
        py_matches = re.findall(r"<run_python>(.*?)</run_python>", response, re.DOTALL)

        num_types = (len(sub_matches) > 0) + (len(exp_matches) > 0) + (len(py_matches) > 0)

        if num_types > 1 or len(py_matches) > 1 or len(sub_matches) > 1:
            error_msg = (
                "Tool call rule violation: You can only issue one type of call per message. "
                "You cannot mix different call types (<run_experiment>, <run_python>, <submission>). "
                "Multiple <run_experiment> calls in one message are allowed, but only a single "
                "<run_python> call or single <submission> call is permitted per message."
            )
            if verbosity >= 3:
                print(f"\n[System Error] {error_msg}")
            messages.append({"role": "user", "content": error_msg})
            continue

        # 1. Check for Submission
        if sub_matches:
            try:
                stat_file_path = os.path.join(trial_root_dir, "latest_stat_validation.json")
                if not os.path.exists(stat_file_path):
                    raise ValueError(
                        "You MUST run the ensemble Chi-Squared Goodness-of-Fit test using your Python sandbox "
                        "and evaluate the results before submitting to determine if they are satisfactory."
                    )

                if verbosity >= 1:
                    print(f"\n[System] Parsing agent submission Python script...")

                submitted_code = sub_matches[0].strip()
                if not submitted_code:
                    raise ValueError("Submission block is empty. Must contain valid Python script code.")

                submission_script_path = os.path.join(trial_root_dir, "submission_model.py")
                with open(submission_script_path, "w") as sf:
                    sf.write(submitted_code)

                spec = importlib.util.spec_from_file_location("agent_submission", submission_script_path)
                agent_mod = importlib.util.module_from_spec(spec)
                sys.modules["agent_submission"] = agent_mod
                spec.loader.exec_module(agent_mod)

                if not hasattr(agent_mod, "predict_trajectory") or not callable(agent_mod.predict_trajectory):
                    raise ValueError("Submitted Python script must define a callable function 'predict_trajectory(t_eval, x0, v0)'.")

                # STRICT CHECK FOR SIGMA FUNCTION SUBMISSION
                if not hasattr(agent_mod, "sigma_x") or not callable(agent_mod.sigma_x):
                    raise ValueError("Submitted Python script MUST define a callable noise function 'sigma_x(x, v, t)'.")
                
                if not hasattr(agent_mod, "sigma_v") or not callable(agent_mod.sigma_v):
                    raise ValueError("Submitted Python script MUST define a callable noise function 'sigma_v(x, v, t)'.")

                if not hasattr(agent_mod, "bias_x") or not callable(agent_mod.bias_x):    # ← NEW
                    raise ValueError("Submitted Python script MUST define a callable bias function 'bias_x(x, v, t)'.")  # ← NEW

                if not hasattr(agent_mod, "bias_v") or not callable(agent_mod.bias_v):    # ← NEW
                    raise ValueError("Submitted Python script MUST define a callable bias function 'bias_v(x, v, t)'.")  # ← NEW

                t_test = np.linspace(0, 10, 200)

                num_ic_samples = 20
                np.random.seed(42)
                x0_samples = np.random.uniform(-10.0, 10.0, size=num_ic_samples)
                v0_samples = np.random.uniform(-10.0, 10.0, size=num_ic_samples)

                fun = lambda t, y: config.hidden_diffeq(t, y, coeffs=dynamic_coeffs)

                mse_x_list = []
                mse_v_list = []

                for x0_val, v0_val in zip(x0_samples, v0_samples):
                    x_pred, v_pred = agent_mod.predict_trajectory(t_test, x0_val, v0_val)

                    if len(x_pred) != len(t_test) or len(v_pred) != len(t_test):
                        raise ValueError(f"predict_trajectory output array length mismatch: Expected {len(t_test)}, got {len(x_pred)}.")

                    sol_true = solve_ivp(fun, [t_test[0], t_test[-1]], [x0_val, v0_val], t_eval=t_test, method="DOP853")

                    if not sol_true.success or len(sol_true.y[0]) != len(t_test):
                        continue

                    mse_x_ic = np.mean((x_pred - sol_true.y[0]) ** 2)
                    mse_v_ic = np.mean((v_pred - sol_true.y[1]) ** 2)

                    mse_x_list.append(mse_x_ic)
                    mse_v_list.append(mse_v_ic)

                if len(mse_x_list) == 0:
                    raise RuntimeError("Failed to compute ground truth trajectory solutions during evaluation.")

                mse_x = float(np.mean(mse_x_list))
                mse_v = float(np.mean(mse_v_list))
                mse_total = float((mse_x + mse_v) / 2.0)
                error = mse_total

                # Clean submission payload containing only execution status and core metrics
                submission_payload = {
                    "code": submitted_code,
                    "script_path": submission_script_path,
                    "status": "accepted",
                    "num_eval_trajectories": len(mse_x_list),
                    "points_per_trajectory": len(t_test),
                }

                if verbosity >= 1:
                    print(f"\nSUCCESS! Agent submitted valid Python script.")
                    print(f" -> Saved to: {submission_script_path}")
                    print(f" -> Evaluated Trajectory MSE Total Error = {error:.6f}")

                return submission_payload, error, messages

            except Exception as e:
                error_msg = f"Submission failed: {e}. {prompts.ERROR_PROMPT}"
                if verbosity >= 3:
                    print(f"\n[System Error] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue

        # 2. Check for Experiment
        if exp_matches:
            try:
                exp_outputs = []
                for exp_raw in exp_matches:
                    params = json.loads(exp_raw)
                    exp_type = params.get('type', 'black_box').lower()
                    t_start = float(params.get('t_start', 0.0))
                    delta_t = float(params.get('delta_t', 0.1))
                    steps = int(params.get('steps', 50))
                    
                    has_custom_ic = ("x0" in params and "v0" in params)
                    if has_custom_ic and not allow_custom_ic:
                        raise ValueError("Custom initial conditions are disabled for this benchmark run.")
                    
                    if exp_type == "calibration":
                        if hasattr(config, "calibration_diffeq"):
                            target_diffeq = config.calibration_diffeq
                        elif hasattr(config, "DIFFEQ_TYPES") and "calibration" in config.DIFFEQ_TYPES:
                            target_diffeq = config.DIFFEQ_TYPES["calibration"]
                        else:
                            raise ValueError("Calibration differential equation function is not defined in config.")
                        current_run_id = calibration_run_id
                        exp_coeffs = dynamic_coeffs.copy() if dynamic_coeffs else {}
                        if "k_0" in params:
                            exp_coeffs["k_0"] = float(params["k_0"])
                        elif "k0" in params:
                            exp_coeffs["k_0"] = float(params["k0"])
                    else:
                        if hasattr(config, "DIFFEQ_TYPES") and "black_box" in config.DIFFEQ_TYPES:
                            target_diffeq = config.DIFFEQ_TYPES["black_box"]
                        else:
                            target_diffeq = config.hidden_diffeq
                        current_run_id = run_id
                        exp_coeffs = dynamic_coeffs

                    if verbosity >= 3:
                        print(f"\n[System Experiment Triggered] Type: {exp_type} | Run #{current_run_id}")
                        print(f" -> Agent Parameters: {json.dumps(params, indent=2)}")
                    
                    t_eval = np.linspace(t_start, t_start + delta_t * steps, steps + 1)
                    t_end = t_eval[-1]
                    
                    current_run_data = {
                        "run_id": current_run_id,
                        "type": exp_type,
                        "t_start": t_start,
                        "delta_t": delta_t,
                        "steps_per_trajectory": steps,
                        "trajectories": []
                    }
                    if exp_type == "calibration" and "k_0" in exp_coeffs:
                        current_run_data["k_0"] = exp_coeffs["k_0"]
                                    
                    num_trajectories = 1 if has_custom_ic else getattr(config, "NUM_TRAJECTORIES", 100)
                    
                    for traj_idx in range(num_trajectories):
                        if has_custom_ic:
                            x0_requested = float(params["x0"])
                            v0_requested = float(params["v0"])
                        else:
                            x_min, x_max = config.ENV_SCHEMA['x']['range']
                            v_min, v_max = config.ENV_SCHEMA['v']['range']
                            x0_requested = np.random.uniform(x_min, x_max)
                            v0_requested = np.random.uniform(v_min, v_max)
                        
                        noisy_initial = {"x": x0_requested, "v": v0_requested}

                        x0_noisy = noisy_initial["x"]
                        v0_noisy = noisy_initial["v"] 
                        
                        fun = lambda t, y: target_diffeq(t, y, coeffs=exp_coeffs)
                        sol = solve_ivp(fun, [t_start, max(t_end, 1e-5)], [x0_noisy, v0_noisy], t_eval=t_eval)

                        measurements = []
                        for i, t_val in enumerate(sol.t):
                            x_true = sol.y[0, i]
                            v_true = sol.y[1, i]
                            noisy_x, noisy_v = config.apply_measurement_noise([x_true, v_true], noise_config, t=t_val)  # ← FIXED
                            measurements.append({
                                "t": float(t_val),
                                "x_measured": float(noisy_x),
                                "v_measured": float(noisy_v)
                            })
                            
                        current_run_data["trajectories"].append({
                            "trajectory_id": traj_idx + 1,
                            "initial_conditions": {"x0": float(x0_requested), "v0": float(v0_requested)},
                            "measurements": measurements
                        })
                        
                        trigger_plot = (traj_idx == 0 and plotting_mode > 0)
                        if trigger_plot:
                            plot_filename = f"plot_turn_{turn}_{exp_type}_run_{current_run_id}.png"
                            output_plot_path = os.path.join(measurements_dir, plot_filename)
                            
                            plot_script = os.path.join("core", "plot_core", "plot_trajectory.py")
                            
                            plot_cmd = [
                                sys.executable, plot_script,
                                "--config", config_path,
                                "--x_init", str(x0_noisy),
                                "--v_init", str(v0_noisy),
                                "--t_start", str(t_start),
                                "--t_end", str(t_end),
                                "--output_path", output_plot_path,
                                "--measurements_json", json.dumps(measurements)
                            ]
                            subprocess.run(plot_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if exp_type == "calibration":
                        run_file_name = f"run_{current_run_id}_calibration_data.json"
                        master_file_name = "all_compiled_calibration_experiments.json"
                    else:
                        run_file_name = f"run_{current_run_id}_experiment_data.json"
                        master_file_name = "all_compiled_experiments.json"

                    run_file = os.path.join(measurements_dir, run_file_name)
                    with open(run_file, "w") as f:
                        json.dump(current_run_data, f, indent=4)
                        
                    master_file = os.path.join(measurements_dir, master_file_name)
                    master_data = {}
                    if os.path.exists(master_file):
                        with open(master_file, "r") as f:
                            master_data = json.load(f)
                    master_data[f"run_{current_run_id}"] = current_run_data
                    with open(master_file, "w") as f:
                        json.dump(master_data, f, indent=4)
                    
                    meta_dict = {
                        "run_id": current_run_id,
                        "type": exp_type,
                        "t_start": t_start,
                        "delta_t": delta_t,
                        "steps_per_trajectory": steps
                    }
                    if exp_type == "calibration":
                        try:
                            func_str = inspect.getsource(target_diffeq).strip()
                        except Exception:
                            func_str = str(target_diffeq)
                        meta_dict["function"] = func_str
                        meta_dict["k_0"] = exp_coeffs.get("k_0", -4.0)

                    exp_output_payload = {
                        "status": "success",
                        "message": f"Successfully executed {exp_type.capitalize()} Run #{current_run_id}. Generated {num_trajectories} trajectory/trajectories.",
                        "metadata": meta_dict,
                        "current_batch_file": f"./results_output/measurements/{run_file_name}",
                        "master_history_file": f"./results_output/measurements/{master_file_name}",
                        "instructions": "Load the history file via Python to inspect trajectory arrays."
                    }
                    exp_output_str = f"<experiment_output>\n{json.dumps(exp_output_payload, indent=2)}\n</experiment_output>"
                    exp_outputs.append(exp_output_str)

                    if exp_type == "calibration":
                        calibration_run_id += 1
                    else:
                        run_id += 1

                combined_exp_output = "\n\n".join(exp_outputs)
                if verbosity >= 3:
                    print(f"[Console Output - Experiment Results ({len(exp_outputs)} calls)]\n{combined_exp_output}")

                messages.append({"role": "user", "content": combined_exp_output})
                continue
            except Exception as e:
                error_msg = f"Experiment failed: {e}."
                if verbosity >= 3:
                    print(f"\n[Console Output - Experiment Error]\n{error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue
        
        # 3. Check for Python Sandbox Execution
        if py_matches:
            code = py_matches[0].strip()
            max_install_retries = 3
            py_output = ""
            
            for attempt in range(max_install_retries):
                try:
                    if verbosity >= 3:
                        print(f"\n[Console Output - Executing Python Sandbox (Attempt {attempt+1})]")
                        print(f"Code to execute:\n{code}")
                    
                    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        py_output = f"[Python Execution Results]\n{result.stdout}\n{result.stderr}"
                        break
                    
                    stderr_output = result.stderr
                    match = re.search(r"ModuleNotFoundError:\s+No\s+module\s+named\s+'([^']+)'", stderr_output)
                    
                    if match:
                        missing_module = match.group(1)
                        pip_package = "scikit-learn" if missing_module == "sklearn" else ("opencv-python" if missing_module == "cv2" else missing_module)
                            
                        if verbosity >= 3:
                            print(f"\n[Auto-Installer] Missing module '{missing_module}' caught. Running pip install {pip_package}...")
                            
                        install_result = subprocess.run([sys.executable, "-m", "pip", "install", pip_package], capture_output=True, text=True)
                        
                        if install_result.returncode == 0:
                            continue 
                        else:
                            py_output = f"[Python Execution Results]\n{result.stdout}\n{result.stderr}"
                            break
                    else:
                        py_output = f"[Python Execution Results]\n{result.stdout}\n{result.stderr}"
                        break
                        
                except subprocess.TimeoutExpired:
                    py_output = "Python execution timed out (>15s)."
                    break

            if verbosity >= 3:
                print(f"[Console Output - Python Sandbox Output]\n{py_output}")

            messages.append({"role": "user", "content": py_output})
            continue
    return None, None, messages
