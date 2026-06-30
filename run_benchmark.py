import os
import sys
import json
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

def call_openrouter_stream(messages, model):
    """Calls OpenRouter with streaming enabled and prints chunks to the terminal in real-time."""
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
                            # Print to terminal immediately without adding a newline
                            print(content, end="", flush=True)
                            full_response += content
                except json.JSONDecodeError:
                    pass
                    
    print()  # Add a final newline when the stream finishes
    return full_response

def run_trial(trial_id, config, model_id):
    messages = [{"role": "system", "content": prompts.get_system_prompt(config.ENV_SCHEMA)}]
    
    x, x_dot = sp.symbols('x x_dot')
    
    for turn in range(15):
        print(f"\n--- [Trial {trial_id} - Turn {turn}] Agent is thinking... ---")
        
        # The agent's response will now stream directly to the terminal here
        response = call_openrouter_stream(messages, model_id)
        messages.append({"role": "assistant", "content": response})

        # 1. Check for Submission
        sub_match = re.search(r"<submission>(.*?)</submission>", response, re.DOTALL)
        if sub_match:
            try:
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
                print(f"\nSUCCESS! Agent predicted k_0 = {k_pred}. True = {true_k0}. Error = {error}")
                return submission_data, error, messages
                
            except Exception as e:
                error_msg = f"Submission failed: {e}. {prompts.ERROR_PROMPT}"
                print(f"\n[System] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue

        # 2. Check for Experiment
        exp_match = re.search(r"<run_experiment>(.*?)</run_experiment>", response, re.DOTALL)
        if exp_match:
            try:
                points = json.loads(exp_match.group(1))
                results = []
                extra_keys = [k for k in config.ENV_SCHEMA.keys() if k not in ['x', 'v', 't']]
                
                for pt in points:
                    t_end = pt['t']
                    args_tuple = tuple(pt[k] for k in extra_keys)
                    sol = solve_ivp(config.hidden_diffeq, [0, max(t_end, 1e-5)], [1.0, 0.0], args=args_tuple)
                    results.append(sol.y[0, -1])
                
                exp_output = f"[Experiment Results]\n<experiment_output>\n{results}\n</experiment_output>"
                print(f"\n[System returning experiment results...]")
                messages.append({"role": "user", "content": exp_output})
                continue
            except Exception as e:
                error_msg = f"Experiment failed: {e}. {prompts.ERROR_PROMPT}"
                print(f"\n[System] {error_msg}")
                messages.append({"role": "user", "content": error_msg})
                continue

        # 3. Check for Python Code Execution
        py_match = re.search(r"<run_python>(.*?)</run_python>", response, re.DOTALL)
        if py_match:
            code = py_match.group(1).strip()
            try:
                print(f"\n[System executing Python code...]")
                result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=10)
                py_output = f"[Python Execution Results]\n{result.stdout}\n{result.stderr}"
                messages.append({"role": "user", "content": py_output})
                continue
            except subprocess.TimeoutExpired:
                timeout_msg = "Python execution timed out (>10s)."
                print(f"\n[System] {timeout_msg}")
                messages.append({"role": "user", "content": timeout_msg})
                continue

        # If agent outputs none of the above
        print(f"\n[System] Agent format error. Sending reminder.")
        messages.append({"role": "user", "content": prompts.ERROR_PROMPT})

    print(f"Trial {trial_id} failed: Max turns reached.")
    return None, None, messages

def main():
    parser = argparse.ArgumentParser(description="Run LLM DiffEq Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Agent shorthand (e.g., o4m)")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--diff_eq_config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.diff_eq_config)
    
    # Load agent shorthand
    with open("agents.json", "r") as f:
        agents = json.load(f)
    model_id = agents.get(args.model, args.model)
    
    print(f"Starting Benchmark. Model: {model_id}, Config: {args.diff_eq_config}")

    all_submissions = []
    all_errors = []
    all_chat_logs = []

    for i in range(args.num_trials):
        print(f"\n{'='*60}\nSTARTING TRIAL {i+1}/{args.num_trials}\n{'='*60}")
        
        submission, error, chat_log = run_trial(i+1, config, model_id)
        
        # Save chat logs regardless of success or failure
        all_chat_logs.append({
            "trial": i+1, 
            "status": "success" if submission else "failed",
            "log": chat_log
        })
        
        if submission is not None:
            all_submissions.append({"trial": i+1, "submission": submission})
            all_errors.append({"trial": i+1, "error": error})

    # Save outputs to JSON
    with open("submissions_log.json", "w") as f:
        json.dump(all_submissions, f, indent=4)
        
    with open("errors_log.json", "w") as f:
        json.dump(all_errors, f, indent=4)
        
    with open("chat_logs.json", "w") as f:
        json.dump(all_chat_logs, f, indent=4)

    print("\nBenchmark Complete!")
    print(" - submissions_log.json (Final answers)")
    print(" - errors_log.json (Absolute errors)")
    print(" - chat_logs.json (Complete conversational history and prompts)")

if __name__ == "__main__":
    main()
