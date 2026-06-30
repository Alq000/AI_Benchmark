import os
import sys
import json
import time
import argparse
import importlib.util
import re
import requests
import sympy as sp
import subprocess
from dotenv import load_dotenv
from scipy.integrate import solve_ivp

# Load env variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

import prompts

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def call_openrouter_stream(messages, model, verbosity):
    """Calls OpenRouter with streaming enabled and conditionally prints chunks to the terminal."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model, 
        "messages": messages,
        "stream": True  # Enable streaming
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, stream=True)
    response.raise_for_status()
    
    full_response = ""
    
    # Process the Server-Sent Events (SSE) stream
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                json_str = line[6:]  # Strip the "data: " prefix
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
        print()  # Add a final newline when the stream finishes
    return full_response

def run_trial(trial_id, config, model_id, verbosity, plotting_mode, config_path, measurements_dir):
    system_prompt = prompts.get_system_prompt(config.ENV_SCHEMA)
    messages = [{"role": "system", "content": system_prompt}]
    
    if verbosity == 4:
        print(f"\n{'#'*60}\n[SYSTEM PROMPT LOADED FOR TRIAL {trial_id}]\n{'#'*60}")
        print(system_prompt)
        print(f"{'#'*60}\n")
    
    x, x_dot = sp.symbols('x x_dot')
    plotted_count = 0  # Tracker for mode 1 constraints
    
    for turn in range(25):
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
                
                true_k0 = config.TRUE_COEFFS['k_0']
                error = abs(k_pred - true_k0)
                
                if verbosity >= 1:
                    print(f"\nSUCCESS! Agent predicted k_0 = {k_pred}. True = {true_k0}. Error = {error}")
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
                points = json.loads(exp_match.group(1))
                
                if verbosity >= 3:
                    print(f"\n[System Experiment Triggered]")
                    print(f" -> Querying Input Parameters: {json.dumps(points, indent=2)}")
                
                results = []
                extra_keys = [k for k in config.ENV_SCHEMA.keys() if k not in ['x', 'v', 't']]
                
                for pt in points:
                    t_end = pt['t']
                    x_init = pt.get('x', 1.0)
                    v_init = pt.get('v', 0.0)
                    
                    args_tuple = tuple(pt[k] for k in extra_keys)
                    sol = solve_ivp(config.hidden_diffeq, [0, max(t_end, 1e-5)], [x_init, v_init], args=args_tuple)
                    
                    x_final = float(sol.y[0, -1])
                    v_final = float(sol.y[1, -1])
                    results.append([x_final, v_final])
                    
                    trigger_plot = False
                    if plotting_mode == 2:
                        trigger_plot = True
                    elif plotting_mode == 1 and plotted_count == 0:
                        trigger_plot = True
                        
                    if trigger_plot:
                        extra_params_dict = {k: pt[k] for k in extra_keys if k in pt}
                        plot_filename = f"plot_turn_{turn}_idx_{len(results)-1}.png"
                        output_plot_path = os.path.join(measurements_dir, plot_filename)
                        
                        plot_cmd = [
                            sys.executable, "plot_trajectory.py",
                            "--config", config_path,
                            "--x_init", str(x_init),
                            "--v_init", str(v_init),
                            "--t", str(t_end),
                            "--extra_params", json.dumps(extra_params_dict),
                            "--output_path", output_plot_path
                        ]
                        subprocess.Popen(plot_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        plotted_count += 1
                
                exp_output = f"[Experiment Results]\n<experiment_output>\n{results}\n</experiment_output>"
                
                if verbosity >= 3:
                    print(f" -> Returned State Variable Vector Quantities [x, v]: {results}")
                
                messages.append({"role": "user", "content": exp_output})
                continue
            except Exception as e:
                error_msg = f"Experiment failed: {e}. {prompts.ERROR_PROMPT}"
                if verbosity >= 3:
                    print(f"\n[System Error] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue

        # 3. Check for Python Code Execution
        py_match = re.search(r"<run_python>(.*?)</run_python>", response, re.DOTALL)
        if py_match:
            code = py_match.group(1).strip()
            try:
                if verbosity >= 3:
                    print(f"\n[System Executing Python Code Sandbox]")
                    print(f"{'-'*40}\n{code}\n{'-'*40}")
                
                result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=10)
                py_output = f"[Python Execution Results]\n{result.stdout}\n{result.stderr}"
                
                if verbosity >= 3:
                    if result.stdout:
                        print(f"[STDOUT]:\n{result.stdout.strip()}")
                    if result.stderr:
                        print(f"[STDERR]:\n{result.stderr.strip()}")
                    print(f"{'-'*40}")
                
                messages.append({"role": "user", "content": py_output})
                continue
            except subprocess.TimeoutExpired:
                timeout_msg = "Python execution timed out (>10s)."
                if verbosity >= 3:
                    print(f"\n[System Error] {timeout_msg}")
                messages.append({"role": "user", "content": timeout_msg})
                continue

        if verbosity >= 3:
            print(f"\n[System Warning] Agent syntax format mismatch. Forwarding error template.")
        messages.append({"role": "user", "content": prompts.ERROR_PROMPT})

    if verbosity >= 1:
        print(f"Trial {trial_id} failed: Max turns reached.")
    return None, None, messages

def main():
    parser = argparse.ArgumentParser(description="Run LLM DiffEq Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Agent shorthand (e.g., o4m)")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--diff_eq_config", type=str, required=True, help="Path to config file")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3, 4], default=4, 
                        help="Output levels: 0=silent, 1=meta-only, 2=+agent, 3=+system-chat, 4=everything")
    parser.add_argument("--plotting", type=int, choices=[0, 1, 2], default=0,
                        help="Plotting mode: 0=No plots (Default), 1=One plot per trial, 2=Plot every measurement")
    args = parser.parse_args()

    config = load_config(args.diff_eq_config)
    
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
        "target_model_endpoint": model_id
    }
    with open(os.path.join(run_dir, "command_args.json"), "w") as f:
        json.dump(command_log, f, indent=4)

    if args.verbosity >= 1:
        print(f"Starting Benchmark. Model: {model_id}, Config: {args.diff_eq_config}, Plotting Level: {args.plotting}")
        print(f"Saving workspace output records cleanly into: {run_dir}\n")

    all_submissions = []
    all_errors = []

    for i in range(args.num_trials):
        trial_num = i + 1
        if args.verbosity >= 1:
            print(f"\n{'='*60}\nSTARTING TRIAL {trial_num}/{args.num_trials}\n{'='*60}")
        
        trial_folder = os.path.join(trials_dir, f"trial_{trial_num}")
        measurements_dir = os.path.join(trial_folder, "measurements")
        os.makedirs(measurements_dir, exist_ok=True)
        
        submission, error, chat_log = run_trial(
            trial_num, config, model_id, args.verbosity, 
            args.plotting, args.diff_eq_config, measurements_dir
        )
        
        trial_data = {
            "trial_id": trial_num,
            "status": "success" if submission else "failed",
            "chat_history": chat_log
        }
        
        trial_file_path = os.path.join(trial_folder, f"trial_{trial_num}.json")
        with open(trial_file_path, "w") as f:
            json.dump(trial_data, f, indent=4)
        
        if submission is not None:
            all_submissions.append({"trial": trial_num, "submission": submission})
            all_errors.append({"trial": trial_num, "error": error})

    errors_log_path = os.path.join(run_dir, "errors_log.json")

    with open(os.path.join(run_dir, "submissions_log.json"), "w") as f:
        json.dump(all_submissions, f, indent=4)
        
    with open(errors_log_path, "w") as f:
        json.dump(all_errors, f, indent=4)

    # --- AUTOMATIC HISTOGRAM TRIGGERING SEQUENCE ---
    if args.verbosity >= 1:
        print("\nGenerating final aggregate evaluation error histogram...")
    
    histogram_output_path = os.path.join(run_dir, "error_distribution_histogram.png")
    hist_cmd = [
        sys.executable, "plot_errors_histogram.py",
        "--errors_path", errors_log_path,
        "--config", args.diff_eq_config,
        "--output_path", histogram_output_path
    ]
    subprocess.run(hist_cmd)

    if args.verbosity >= 1:
        print("\nBenchmark Execution Sequence Terminated Successfully!")

if __name__ == "__main__":
    main()
