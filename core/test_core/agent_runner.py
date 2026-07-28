import os
import sys
import json
import time
import re
import requests
import sympy as sp
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
                "  \"t_start\": 10.0,\n"
                "  \"delta_t\": 0.05,\n"
                "  \"steps\": 20\n"
                "}\n"
                "</run_experiment>"
            )
        elif assistant_turns == 2:
            mock_output = (
                "<run_experiment>\n"
                "{\n"
                "  \"t_start\": 0.0,\n"
                "  \"delta_t\": 0.1,\n"
                "  \"steps\": 20,\n"
                "  \"x0\": 2.5,\n"
                "  \"v0\": -1.0\n"
                "}\n"
                "</run_experiment>"
            )
        else:
            mock_output = (
                f"<submission>\n"
                f"{{\n"
                f"  \"discovered_terms\": [{{\"term\": \"x\", \"coeff\": -4.761}}]\n"
                f"}}\n"
                f"</submission>"
            )
        if verbosity >= 2:
            print(mock_output, flush=True)
        return mock_output

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    data = {"model": model, "messages": messages, "stream": True}
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, stream=True)
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
    
    x, x_dot = sp.symbols('x x_dot')
    run_id = 1
    
    for turn in range(max_turns):
        if model_id != "no_agent":
            time.sleep(3)
        if verbosity >= 1:
            print(f"\n--- [Trial {trial_id} - Turn {turn}] Agent is thinking... ---")
        
        response = call_openrouter_stream(messages, model_id, verbosity, openrouter_api_key)
        messages.append({"role": "assistant", "content": response})

        # 1. Check for Submission
        sub_match = re.search(r"<submission>(.*?)</submission>", response, re.DOTALL)
        if sub_match:
            try:
                stat_file_path = os.path.join(os.path.dirname(measurements_dir), "latest_stat_validation.json")
                if not os.path.exists(stat_file_path):
                    raise ValueError(
                        "You MUST run the ensemble Chi-Squared Goodness-of-Fit test using your Python sandbox "
                        "and evaluate the results before submitting to determine if they are satisfactory. If "
                        "they are not, you should conduct more experiments and data analysis."
                    )

                if verbosity >= 1:
                    print(f"\n[System] Parsing agent submission block...")
                submission_data = json.loads(sub_match.group(1))
                
                if not isinstance(submission_data, dict) or "discovered_terms" not in submission_data:
                    raise ValueError("Submission must be a JSON object containing the 'discovered_terms' key.")
                
                discovered_terms = submission_data["discovered_terms"]

                k_pred = None
                for entry in discovered_terms:
                    expr = sp.parse_expr(entry['term'])
                    if sp.simplify(expr - x) == 0:
                        k_pred = float(entry['coeff'])
                        break
                
                if k_pred is None:
                    raise ValueError("Could not find a term equivalent to 'x' in discovered_terms.")
                
                true_k0 = dynamic_coeffs.get('k_0', config.TRUE_COEFFS.get('k_0'))
                error = (k_pred - true_k0) / true_k0
                
                if verbosity >= 1:
                    print(f"\nSUCCESS! Agent predicted k_0 = {k_pred}. True = {true_k0}. Relative Error = {error}")
                return submission_data, error, messages
                
            except Exception as e:
                error_msg = f"Submission failed: {e}. {prompts.ERROR_PROMPT}"
                if verbosity >= 3:
                    print(f"\n[System Error] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue
       
        # 2. Check for Experiment
        exp_match = re.search(r"<run_experiment>(.*?)</run_experiment>", response, re.DOTALL)
        if exp_match:
            try:
                params = json.loads(exp_match.group(1))
                t_start = float(params.get('t_start', 0.0))
                delta_t = float(params.get('delta_t', 0.1))
                steps = int(params.get('steps', 50))
                
                has_custom_ic = ("x0" in params and "v0" in params)
                if has_custom_ic and not allow_custom_ic:
                    raise ValueError("Custom initial conditions are disabled for this benchmark run.")
                
                if verbosity >= 3:
                    print(f"\n[System Experiment Triggered] Run #{run_id}")
                    print(f" -> Agent Parameters: {json.dumps(params, indent=2)}")
                
                t_eval = np.linspace(t_start, t_start + delta_t * steps, steps + 1)
                t_end = t_eval[-1]
                
                current_run_data = {
                    "run_id": run_id,
                    "t_start": t_start,
                    "delta_t": delta_t,
                    "steps_per_trajectory": steps,
                    "trajectories": []
                }
                
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
                    
                    if hasattr(config, "apply_actuator_bias"):
                        noisy_initial = config.apply_actuator_bias({"x": x0_requested, "v": v0_requested}, noise_config)
                    else:
                        noisy_initial = config.apply_input_noise({"x": x0_requested, "v": v0_requested}, noise_config)
                        
                    x0_noisy = noisy_initial["x"]
                    v0_noisy = noisy_initial["v"] 
                    
                    fun = lambda t, y: config.hidden_diffeq(t, y, coeffs=dynamic_coeffs)
                    sol = solve_ivp(fun, [t_start, max(t_end, 1e-5)], [x0_noisy, v0_noisy], t_eval=t_eval)
                    
                    measurements = []
                    for i, t_val in enumerate(sol.t):
                        x_true = sol.y[0, i]
                        v_true = sol.y[1, i]
                        noisy_x, noisy_v = config.apply_measurement_noise([x_true, v_true], noise_config)
                        measurements.append({
                            "t": float(t_val),
                            "x_measured": float(noisy_x),
                            "v_measured": float(noisy_v)
                        })
                        
                    current_run_data["trajectories"].append({
                        "trajectory_id": traj_idx + 1,
                        "initial_conditions": {"x0": float(x0_noisy), "v0": float(v0_noisy)},
                        "measurements": measurements
                    })
                    
                    trigger_plot = (traj_idx == 0 and plotting_mode > 0)
                    if trigger_plot:
                        plot_filename = f"plot_turn_{turn}_run_{run_id}.png"
                        output_plot_path = os.path.join(measurements_dir, plot_filename)
                        plot_cmd = [
                            sys.executable, "plot_trajectory.py",
                            "--config", config_path,
                            "--x_init", str(x0_noisy),
                            "--v_init", str(v0_noisy),
                            "--t_start", str(t_start),
                            "--t_end", str(t_end),
                            "--output_path", output_plot_path,
                            "--measurements_json", json.dumps(measurements)
                        ]
                        subprocess.run(plot_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                run_file = os.path.join(measurements_dir, f"run_{run_id}_experiment_data.json")
                with open(run_file, "w") as f:
                    json.dump(current_run_data, f, indent=4)
                    
                master_file = os.path.join(measurements_dir, "all_compiled_experiments.json")
                master_data = {}
                if os.path.exists(master_file):
                    with open(master_file, "r") as f:
                        master_data = json.load(f)
                master_data[f"run_{run_id}"] = current_run_data
                with open(master_file, "w") as f:
                    json.dump(master_data, f, indent=4)
                
                exp_output = (
                    f"<experiment_output>\n"
                    f"{{\n"
                    f"  \"status\": \"success\",\n"
                    f"  \"message\": \"Successfully executed Experiment Run #{run_id}. Generated {num_trajectories} trajectory/trajectories.\",\n"
                    f"  \"metadata\": {{\"run_id\": {run_id}, \"t_start\": {t_start}, \"delta_t\": {delta_t}, \"steps_per_trajectory\": {steps}, \"max_polynomial_order\": 3}},\n"
                    f"  \"current_batch_file\": \"./results_output/measurements/run_{run_id}_experiment_data.json\",\n"
                    f"  \"master_history_file\": \"./results_output/measurements/all_compiled_experiments.json\",\n"
                    f"  \"instructions\": \"Load the master history file via Python to analyze initial state responses and trajectories.\"\n"
                    f"}}\n"
                    f"</experiment_output>"
                )
                
                if verbosity >= 3:
                    print(f" -> File saved successfully to {master_file}")
                messages.append({"role": "user", "content": exp_output})
                run_id += 1
                continue
            except Exception as e:
                error_msg = f"Experiment failed: {e}."
                if verbosity >= 3:
                    print(f"\n[System Error] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue

        # 3. Check for Python Sandbox Execution
        py_match = re.search(r"<run_python>(.*?)</run_python>", response, re.DOTALL)
        if py_match:
            code = py_match.group(1).strip()
            max_install_retries = 3
            py_output = ""
            
            for attempt in range(max_install_retries):
                try:
                    if verbosity >= 3:
                        print(f"\n[System Executing Python Code Sandbox] (Attempt {attempt+1})")
                    
                    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
                    
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

            messages.append({"role": "user", "content": py_output})
            continue
    return None, None, messages
