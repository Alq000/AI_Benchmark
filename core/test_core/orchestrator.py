import os
import sys
import json
import time
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from test_core.compile_benchmark_data import compile_data

print_lock = threading.Lock()

def run_orchestration(args, config, noise_config, openrouter_api_key):
    if args.model == "no_agent":
        model_id = "no_agent"
    else:
        if not openrouter_api_key:
            print("Error: OPENROUTER_API_KEY environment variable is missing.")
            sys.exit(1)
        agents_path = os.path.abspath("core/configs/agents.json")
        with open(agents_path, "r") as f:
            agents = json.load(f)
        model_id = agents.get(args.model, args.model)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_dir_name = re.sub(r'[\\/*?:"<>|]', "_", args.model)
    run_dir = os.path.join("results", agent_dir_name, f"run_{timestamp}")
    trials_dir = os.path.join(run_dir, "trials")
    plots_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
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

    if getattr(args, "vary_params", False) and hasattr(config, "generate_trial_coefficients"):
        trial_parameter_sets = config.generate_trial_coefficients(num_trials=args.num_trials, grouping=args.grouping)
    else:
        trial_parameter_sets = [getattr(config, "TRUE_COEFFS", {}) for _ in range(args.num_trials)]

    def launch_container_worker(trial_num):
        trial_folder = os.path.join(trials_dir, f"trial_{trial_num}")
        os.makedirs(os.path.join(trial_folder, "measurements"), exist_ok=True)
        
        assigned_coeffs = trial_parameter_sets[trial_num - 1]
        with open(os.path.join(trial_folder, "true_coeffs.json"), "w") as f:
            json.dump(assigned_coeffs, f, indent=4)
        
        if args.verbosity >= 1:
            with print_lock:
                print(f"[Host Orchestrator] Spawning Docker Container for Trial {trial_num}...")
                
        docker_cmd = [
            "docker", "run", "--rm",
            "-e", f"OPENROUTER_API_KEY={openrouter_api_key}",
            "-v", f"{os.path.abspath('core/configs/prompts.py')}:/app/core/configs/prompts.py:ro",
            "-v", f"{os.path.abspath('run_benchmark.py')}:/app/run_benchmark.py:ro",
            "-v", f"{os.path.abspath('core')}:/app/core:ro",
            "-v", f"{os.path.abspath('core/configs/agents.json')}:/app/core/configs/agents.json:ro",
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

        if getattr(args, "vary_params", False):
            docker_cmd.append("--vary_params")
        else:
            docker_cmd.append("--no_vary_params")

        if getattr(args, "custom_initial_conditions", False):
            docker_cmd.extend(["--custom_initial_conditions", "True"])
        else:
           docker_cmd.extend(["--custom_initial_conditions", "False"]) 
        
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

        try:
            if args.verbosity >= 1:
                with print_lock:
                    print(f"[Host Orchestrator] Executing empirical error calculation for: {trial_folder}")
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "calculate_empirical_error.py"), trial_folder], capture_output=True, text=True, check=True)
        except Exception as e:
            with print_lock:
                print(f"[Warning] Failed to launch empirical error script: {e}")

        # Judge Container Step
        if args.judge:
            judge_model_key = args.judge
            if args.verbosity >= 1:
                with print_lock:
                    print(f"[Host Orchestrator] Spawning Isolated Judge Container ({judge_model_key}) for Trial {trial_num}...")

            judge_docker_cmd = [
                "docker", "run", "--rm",
                "-e", f"OPENROUTER_API_KEY={openrouter_api_key}",
                "-v", f"{os.path.abspath('core/configs/prompts.py')}:/app/core/configs/prompts.py:ro",
                "-v", f"{os.path.abspath('core/configs/agents.json')}:/app/core/configs/agents.json:ro",
                "-v", f"{os.path.abspath('core/configs/rubric_config.py')}:/app/core/configs/rubric_config.py:ro",
                "-v", f"{os.path.abspath('core/judge_core')}:/app/core/judge_core:ro",
                "-v", f"{os.path.abspath(args.diff_eq_config)}:/app/{args.diff_eq_config}:ro",
                "-v", f"{os.path.abspath(trial_folder)}:/app/results_output:rw",
                "diffeq-benchmark",
                "python", "-m", "core.judge_core.judge_trial",
                "--trial_dir", "/app/results_output",
                "--diff_eq_config", args.diff_eq_config,
                "--judge_model", judge_model_key
            ]

            if is_sequential:
                subprocess.run(judge_docker_cmd)
            else:
                j_result = subprocess.run(judge_docker_cmd, capture_output=True, text=True)
                if args.verbosity >= 1:
                    with print_lock:
                        print(f"\n--- [Consolidated Judge Logs from Trial {trial_num} Container Room] ---")
                        if j_result.stdout.strip(): print(j_result.stdout.strip())
                        if j_result.stderr.strip(): print(j_result.stderr.strip())
        
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
    submissions_log_path = os.path.join(run_dir, "submissions_log.json")
    with open(submissions_log_path, "w") as f:
        json.dump(all_submissions, f, indent=4)
    with open(errors_log_path, "w") as f:
        json.dump(all_errors, f, indent=4)

    # Plots execution
    histogram_output_path = os.path.join(plots_dir, "error_distribution_histogram.png")
    hist_cmd = [sys.executable, os.path.join("core", "plot_core", "plot_errors_histogram.py"), "--errors_path", errors_log_path, "--config", args.diff_eq_config, "--output_path", histogram_output_path]
    if getattr(args, "vary_params", False): hist_cmd.append("--vary_params")
    subprocess.run(hist_cmd)
   
    accuracy_output_path = os.path.join(plots_dir, "accuracy_band_plot.png")
    print(f"Generating accuracy band plot...")
    subprocess.run([sys.executable, os.path.join("core", "plot_core", "plot_accuracy_band.py"), "--submissions_path", submissions_log_path, "--config", args.diff_eq_config, "--output_path", accuracy_output_path])

    sindy_diff_output_path = os.path.join(plots_dir, "sindy_agent_diff_plot.png")
    print(f"Generating SINDy vs Agent relative error difference plot...")
    diff_cmd = [sys.executable, os.path.join("core", "plot_core", "plot_sindy_agent_diff.py"), "--errors_path", errors_log_path, "--config", args.diff_eq_config, "--output_path", sindy_diff_output_path]
    if getattr(args, "vary_params", False): diff_cmd.append("--vary_params")
    subprocess.run(diff_cmd)

    ratio_output_path = os.path.join(plots_dir, "agent_error_uncertainty_ratio_plot.png")
    print(f"Generating Agent error to empirical uncertainty ratio plot...")
    ratio_cmd = [sys.executable, os.path.join("core", "plot_core", "plot_agent_error_uncertainty_ratio.py"), "--errors_path", errors_log_path, "--config", args.diff_eq_config, "--output_path", ratio_output_path]
    if getattr(args, "vary_params", False): ratio_cmd.append("--vary_params")
    subprocess.run(ratio_cmd)

    if args.judge:
        from judge_core.judge_run_summary import summarize_run
        if args.verbosity >= 1:
            print(f"\n[Host Orchestrator] Executing Run-Level Judge Summary & Sampling Analysis ({args.judge})...")
        summarize_run(run_dir, args.judge, openrouter_api_key) 

    try:
        if args.verbosity >= 1:
            print(f"\n[Host Orchestrator] Auto-compiling benchmark data for run: {run_dir}...")
        compile_data(base_dir=run_dir)
    except Exception as e:
        print(f"[Warning] Automatic data compilation failed: {e}")

    if args.verbosity >= 1:
        print(f"\n{'='*70}")
        print(f"[Benchmark Complete] Orchestration engine finished.")
        print(f"All tracking data, trajectory files, and plots saved to:\n  -> {os.path.abspath(run_dir)}")
        print(f"{'='*70}\n")
