import os
import sys
import json
import time
import argparse
import importlib.util
import re
import requests
import sympy as sp
import numpy as np
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from scipy.integrate import solve_ivp

# Load env variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

import prompts

print_lock = threading.Lock()

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def call_openrouter_stream(messages, model, verbosity):
    if model == "no_agent":
        assistant_turns = len([m for m in messages if m["role"] == "assistant"])
        if assistant_turns == 0:
            mock_output = (
                "Let me test the sandbox setup first to verify everything is functional.\n"
                "<run_python>\n"
                "print('Python works')\n"
                "</run_python>"
            )
        elif assistant_turns == 1:
            mock_output = (
                "Sandbox works. Now I'll fire an experimental measurement sequence.\n"
                "<run_experiment>\n"
                "{\n"
                "  \"t_start\": 0.0,\n"
                "  \"delta_t\": 0.1,\n"
                "  \"steps\": 50\n"
                "}\n"
                "</run_experiment>"
            )
        else:
            true_k0 = -4.761
            variance_k1 = abs(-1.234)
            sampled_k0 = float(np.random.normal(loc=true_k0, scale=variance_k1))
            mock_output = (
                f"Based on the empirical tracking profile data, we extract the linear coefficient.\n"
                f"<submission>\n"
                f"[\n"
                f"  {{\"term\": \"x\", \"coeff\": {sampled_k0:.6f}}}\n"
                f"]\n"
                f"</submission>"
            )
        if verbosity >= 2:
            print(mock_output, flush=True)
        return mock_output

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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

def run_trial(trial_id, config, model_id, verbosity, plotting_mode, config_path, measurements_dir, noise_config, dynamic_coeffs):
    system_prompt = prompts.get_system_prompt(config.ENV_SCHEMA)
    messages = [{"role": "system", "content": system_prompt}]
    
    if verbosity == 4:
        print(f"\n{'#'*60}\n[SYSTEM PROMPT LOADED FOR TRIAL {trial_id}]\n{'#'*60}")
        print(system_prompt)
        print(f"{'#'*60}\n")
    
    x, x_dot = sp.symbols('x x_dot')
    run_id = 1
    
    for turn in range(25):
        if model_id != "no_agent":
            time.sleep(3)
        if verbosity >= 1:
            print(f"\n--- [Trial {trial_id} - Turn {turn}] Agent is thinking... ---")
        
        response = call_openrouter_stream(messages, model_id, verbosity)
        messages.append({"role": "assistant", "content": response})

        # 1. Check for Submission
        sub_match = re.search(r"<submission>(.*?)</submission>", response, re.DOTALL)
        if sub_match:
            try:
                if verbosity >= 1:
                    print(f"\n[System] Parsing agent submission block...")
                submission_data = json.loads(sub_match.group(1))
                k_pred = None
                
                for entry in submission_data:
                    expr = sp.parse_expr(entry['term'])
                    if sp.simplify(expr - x) == 0:
                        k_pred = float(entry['coeff'])
                        break
                
                if k_pred is None:
                    raise ValueError("Could not find a term equivalent to 'x' in submission.")
                
                # Check against the dynamic coefficients assigned to this specific trial
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
                
                num_trajectories = getattr(config, "NUM_TRAJECTORIES", 100)
                
                for traj_idx in range(num_trajectories):
                    x_min, x_max = config.ENV_SCHEMA['x']['range']
                    v_min, v_max = config.ENV_SCHEMA['v']['range']
                    x0_true = np.random.uniform(x_min, x_max)
                    v0_true = np.random.uniform(v_min, v_max)
                    
                    # FIXED: Support both old 'apply_input_noise' and new 'apply_actuator_bias' names
                    if hasattr(config, "apply_actuator_bias"):
                        noisy_initial = config.apply_actuator_bias({"x": x0_true, "v": v0_true}, noise_config)
                    else:
                        noisy_initial = config.apply_input_noise({"x": x0_true, "v": v0_true}, noise_config)
                        
                    x0_noisy = noisy_initial["x"]
                    v0_noisy = noisy_initial["v"] 
                    
                    # Wrap the diff eq to inject the specific dynamic parameters for this trial
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
                    f"  \"message\": \"Successfully executed Experiment Run #{run_id}. Generated {num_trajectories} new trajectories.\",\n"
                    f"  \"metadata\": {{\"run_id\": {run_id}, \"t_start\": {t_start}, \"delta_t\": {delta_t}, \"steps_per_trajectory\": {steps}, \"max_polynomial_order\": 3}},\n"
                    f"  \"current_batch_file\": \"./results_output/measurements/run_{run_id}_experiment_data.json\",\n"
                    f"  \"master_history_file\": \"./results_output/measurements/all_compiled_experiments.json\",\n"
                    f"  \"instructions\": \"Load the master history file via Python. Loop over the trajectories to extract parameters and calculate the mean and variance of your results.\"\n"
                    f"}}\n"
                    f"</experiment_output>"
                )
                
                if verbosity >= 3:
                    print(f" -> File saved successfully to {master_file}")
                messages.append({"role": "user", "content": exp_output})
                run_id += 1
                continue
            except Exception as e:
                error_msg = f"Experiment failed: {e}. Ensure inputs match JSON schema: {{\"t_start\": 0.0, \"delta_t\": 0.1, \"steps\": 50}}"
                if verbosity >= 3:
                    print(f"\n[System Error] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue

        # 3. Check for Python Code Execution
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
                        pip_package = missing_module
                        if pip_package == "sklearn":
                            pip_package = "scikit-learn"
                        elif pip_package == "cv2":
                            pip_package = "opencv-python"
                            
                        if verbosity >= 3:
                            print(f"\n[Auto-Installer] Missing module '{missing_module}' caught. Running pip install {pip_package}...")
                            
                        install_result = subprocess.run([sys.executable, "-m", "pip", "install", pip_package], capture_output=True, text=True)
                        
                        if install_result.returncode == 0:
                            if verbosity >= 3:
                                print(f"[Auto-Installer] Successfully installed '{pip_package}'. Retrying script execution...")
                            continue 
                        else:
                            if verbosity >= 3:
                                print(f"[Auto-Installer] Error: Failed to install '{pip_package}': {install_result.stderr.strip()}")
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

def main():
    parser = argparse.ArgumentParser(description="Run LLM DiffEq Benchmark with Dynamic Noise Framework")
    parser.add_argument("--model", type=str, required=True, help="Agent shorthand")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--diff_eq_config", type=str, required=True, help="Path to config file")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3, 4], default=4)
    parser.add_argument("--plotting", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--max_workers", type=int, default=1)
    
    parser.add_argument("--input_const_noise", type=float, default=None, help="Input constant noise level override")
    parser.add_argument("--input_lin_noise", type=float, default=None, help="Input linear noise scale override")
    parser.add_argument("--meas_const_noise", type=float, default=None, help="Measurement constant noise floor override")
    parser.add_argument("--meas_lin_noise", type=float, default=None, help="Measurement linear noise scale override")
    
    parser.add_argument("--internal_trial_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal_output_dir", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = load_config(args.diff_eq_config)

    noise_config = {
        "input_const_noise": args.input_const_noise if args.input_const_noise is not None else config.DEFAULT_NOISE_CONFIG["input_const_noise"],
        "input_lin_noise": args.input_lin_noise if args.input_lin_noise is not None else config.DEFAULT_NOISE_CONFIG["input_lin_noise"],
        "meas_const_noise": args.meas_const_noise if args.meas_const_noise is not None else config.DEFAULT_NOISE_CONFIG["meas_const_noise"],
        "meas_lin_noise": args.meas_lin_noise if args.meas_lin_noise is not None else config.DEFAULT_NOISE_CONFIG["meas_lin_noise"]
    }

    # =========================================================================
    # INTERNAL CONTAINER EXECUTION (Runs within Docker)
    # =========================================================================
    if args.internal_trial_id is not None:
        measurements_dir = os.path.join(args.internal_output_dir, "measurements")
        os.makedirs(measurements_dir, exist_ok=True)
        
        # Pull dynamic true coefficients prepared by the host orchestrator
        dynamic_coeffs = getattr(config, "TRUE_COEFFS", {}).copy()
        coeffs_path = os.path.join(args.internal_output_dir, "true_coeffs.json")
        if os.path.exists(coeffs_path):
            with open(coeffs_path, "r") as f:
                dynamic_coeffs = json.load(f)

        submission, error, chat_log = run_trial(
            args.internal_trial_id, config, args.model, args.verbosity, 
            args.plotting, args.diff_eq_config, measurements_dir, noise_config, dynamic_coeffs
        )
        
        trial_data = {"trial_id": args.internal_trial_id, "status": "success" if submission else "failed", "chat_history": chat_log}
        with open(os.path.join(args.internal_output_dir, f"trial_{args.internal_trial_id}.json"), "w") as f:
            json.dump(trial_data, f, indent=4)
            
        with open(os.path.join(args.internal_output_dir, "summary.json"), "w") as f:
            json.dump({
                "submission": submission, 
                "error": error,
                "true_k0": dynamic_coeffs.get('k_0')
            }, f, indent=4)
        sys.exit(0)

    # =========================================================================
    # HOST ORCHESTRATION ENGINE
    # =========================================================================
    if args.model == "no_agent":
        model_id = "no_agent"
    else:
        if not OPENROUTER_API_KEY:
            print("Error: OPENROUTER_API_KEY environment variable is missing.")
            sys.exit(1)
        with open("agents.json", "r") as f:
            agents = json.load(f)
        model_id = agents.get(args.model, args.model)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_dir_name = re.sub(r'[\\/*?:"<>|]', "_", args.model)
    run_dir = os.path.join("results", agent_dir_name, f"run_{timestamp}")
    trials_dir = os.path.join(run_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)
    
    command_log = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_terminal_command": f"python {' '.join(sys.argv)}",
        "parsed_arguments": vars(args),
        "applied_noise_configuration": noise_config
    }
    with open(os.path.join(run_dir, "command_args.json"), "w") as f:
        json.dump(command_log, f, indent=4)

    if args.verbosity >= 1:
        print(f"Starting Host Orchestration Engine. Model: {model_id}")
        print(f"Applied Noise Floor Matrix Parameters: {json.dumps(noise_config)}")

    all_submissions = []
    all_errors = []

    # Prepare dynamic coefficients matrix for all runs
    if hasattr(config, "generate_trial_coefficients"):
        trial_parameter_sets = config.generate_trial_coefficients(args.num_trials)
    else:
        trial_parameter_sets = [getattr(config, "TRUE_COEFFS", {}) for _ in range(args.num_trials)]

    def launch_container_worker(trial_num):
        trial_folder = os.path.join(trials_dir, f"trial_{trial_num}")
        os.makedirs(os.path.join(trial_folder, "measurements"), exist_ok=True)
        
        # Inject the unique ground-truth dict into the trial folder so the Docker container can read it
        assigned_coeffs = trial_parameter_sets[trial_num - 1]
        with open(os.path.join(trial_folder, "true_coeffs.json"), "w") as f:
            json.dump(assigned_coeffs, f, indent=4)
        
        if args.verbosity >= 1:
            with print_lock:
                print(f"[Host Orchestrator] Spawning Docker Container for Trial {trial_num}...")
        # Your original, intact Docker command structure
        docker_cmd = [
            "docker", "run", "--rm",
            "-e", f"OPENROUTER_API_KEY={OPENROUTER_API_KEY}",
            "-v", f"{os.path.abspath('prompts.py')}:/app/prompts.py:ro",
            "-v", f"{os.path.abspath('run_benchmark.py')}:/app/run_benchmark.py:ro",
            "-v", f"{os.path.abspath('plot_trajectory.py')}:/app/plot_trajectory.py:ro",
            "-v", f"{os.path.abspath('agents.json')}:/app/agents.json:ro",
            "-v", f"{os.path.abspath(args.diff_eq_config)}:/app/{args.diff_eq_config}:ro",
            "-v", f"{os.path.abspath(trial_folder)}:/app/results_output",
            "diffeq-benchmark",
            "python", "run_benchmark.py",
            "--model", model_id,
            "--diff_eq_config", args.diff_eq_config,
            "--verbosity", str(args.verbosity),
            "--plotting", str(args.plotting),
            "--internal_trial_id", str(trial_num),
            "--internal_output_dir", "/app/results_output",
            "--input_const_noise", str(noise_config["input_const_noise"]),
            "--input_lin_noise", str(noise_config["input_lin_noise"]),
            "--meas_const_noise", str(noise_config["meas_const_noise"]),
            "--meas_lin_noise", str(noise_config["meas_lin_noise"])
        ]
        
        is_sequential = (args.max_workers == 1 or args.num_trials == 1)
        
        if is_sequential:
            subprocess.run(docker_cmd)
        else:
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            if args.verbosity >= 1:
                with print_lock:
                    print(f"\n--- [Consolidated Logs from Trial {trial_num} Container Room] ---")
                    if result.stdout.strip(): print(result.stdout.strip())
                    if result.stderr.strip(): print(result.stderr.strip())

        # =====================================================================
        # CALL EMPIRICAL UNCERTAINTY QUANTIFICATION SCRIPT
        # =====================================================================
        try:
            if args.verbosity >= 1:
                with print_lock:
                    print(f"[Host Orchestrator] Executing empirical error calculation for: {trial_folder}")
            
            script_result = subprocess.run(
                [sys.executable, "calculate_empirical_error.py", trial_folder],
                capture_output=True,
                text=True,
                check=True
            )
            if args.verbosity >= 2:
                with print_lock:
                    if script_result.stdout.strip(): print(script_result.stdout.strip())
        except subprocess.CalledProcessError as e:
            with print_lock:
                print(f"[Error] calculate_empirical_error.py failed with exit code {e.returncode}")
                print(f"Stderr: {e.stderr}")
        except Exception as e:
            with print_lock:
                print(f"[Warning] Failed to launch empirical error script: {e}")
        # =====================================================================

        summary_path = os.path.join(trial_folder, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            return trial_num, summary["submission"], summary["error"], assigned_coeffs
        return trial_num, None, None, assigned_coeffs

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_container = {executor.submit(launch_container_worker, i + 1): i + 1 for i in range(args.num_trials)}
        for future in as_completed(future_to_container):
            t_num, submission, error, assigned_coeffs = future.result()
            if submission is not None:
                all_submissions.append({"trial": t_num, "submission": submission, "true_coeffs": assigned_coeffs})
                all_errors.append({"trial": t_num, "error": error, "true_coeffs": assigned_coeffs})

    all_submissions.sort(key=lambda x: x["trial"])
    all_errors.sort(key=lambda x: x["trial"])

    errors_log_path = os.path.join(run_dir, "errors_log.json")
    with open(os.path.join(run_dir, "submissions_log.json"), "w") as f:
        json.dump(all_submissions, f, indent=4)
    with open(errors_log_path, "w") as f:
        json.dump(all_errors, f, indent=4)

    histogram_output_path = os.path.join(run_dir, "error_distribution_histogram.png")
    hist_cmd = [sys.executable, "plot_errors_histogram.py", "--errors_path", errors_log_path, "--config", args.diff_eq_config, "--output_path", histogram_output_path]
    subprocess.run(hist_cmd)
   
    submissions_log_path = os.path.join(run_dir, "submissions_log.json")
    accuracy_output_path = os.path.join(run_dir, "accuracy_band_plot.png")
    accuracy_cmd = [
        sys.executable, "plot_accuracy_band.py", 
        "--submissions_path", submissions_log_path, 
        "--config", args.diff_eq_config, 
        "--output_path", accuracy_output_path
    ]
    print(f"Generating accuracy band plot...")
    subprocess.run(accuracy_cmd)

    if args.verbosity >= 1:
        print(f"\n{'='*70}")
        print(f"[Benchmark Complete] Orchestration engine finished.")
        print(f"All tracking data, trajectory files, and plots have been saved to:\n  -> {os.path.abspath(run_dir)}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
