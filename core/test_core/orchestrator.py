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

def process_division_analysis(div_dir, args, noise_config):
    """Generates logs and plots for a specific division directory."""
    errors_log_path = os.path.join(div_dir, "errors_log.json")
    submissions_log_path = os.path.join(div_dir, "submissions_log.json")
    plots_dir = os.path.join(div_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Only compute shared MMD metrics & plot heatmap videos if flag is enabled
    if getattr(args, "plot_mmd_heatmap", False):
        try:
            from test_core.config_loader import load_config
            config = load_config(args.diff_eq_config)
            from plot_core.plot_mmd_heatmap import process_division_mmd
            process_division_mmd(div_dir, args, config, noise_config)
        except Exception as e:
            print(f"[Warning] MMD processing failed for division {div_dir}: {e}")

    if not os.path.exists(errors_log_path) or not os.path.exists(submissions_log_path):
        return

    # Gather full summary objects across all trials in this division
    trial_summaries = []
    trial_dirs = [os.path.join(div_dir, d) for d in os.listdir(div_dir) if d.startswith("trial_")]
    for t_dir in sorted(trial_dirs):
        s_file = os.path.join(t_dir, "summary.json")
        if os.path.exists(s_file):
            try:
                with open(s_file, "r") as sf:
                    trial_summaries.append(json.load(sf))
            except Exception:
                pass

    # Save consolidated division summary for plotters that need SINDy and submission data
    consolidated_summary_path = os.path.join(div_dir, "compiled_division_summary.json")
    with open(consolidated_summary_path, "w") as f:
        json.dump(trial_summaries, f, indent=4)

    def run_plot_cmd(cmd_list, name):
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Plotting Error - {name}]\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    # 1. Histogram
    hist_cmd = [sys.executable, os.path.join("core", "plot_core", "plot_errors_histogram.py"), "--errors_path", errors_log_path, "--config", args.diff_eq_config, "--output_path", os.path.join(plots_dir, "error_distribution_histogram.png")]
    if getattr(args, "vary_params", False): hist_cmd.append("--vary_params")
    run_plot_cmd(hist_cmd, "plot_errors_histogram")

    # 2. Accuracy Band
    accuracy_cmd = [sys.executable, os.path.join("core", "plot_core", "plot_accuracy_band.py"), "--submissions_path", consolidated_summary_path, "--config", args.diff_eq_config, "--output_path", os.path.join(plots_dir, "accuracy_band_plot.png")]
    run_plot_cmd(accuracy_cmd, "plot_accuracy_band")

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

    # =========================================================================
    # DYNAMIC SETUP GENERATION FOR 3x3 ALPHA/BETA NOISE GRID
    # =========================================================================
    setup_data = None
    if getattr(args, "vary_setup", None) and os.path.exists(args.vary_setup):
        with open(args.vary_setup, "r") as f:
            setup_data = json.load(f)
    elif getattr(args, "use_alpha_beta_grid", False):
        # Automatically sweep across 3x3 grid pairs: (beta, alpha) in {0.1, 1.0, 5.0}^2
        from core.configs.diffeq_config import BETA_VALUES, ALPHA_VALUES
        divisions = []
        div_id = 1
        for b in BETA_VALUES:
            for a in ALPHA_VALUES:
                divisions.append({
                    "division_id": div_id,
                    "params": {
                        "use_alpha_beta_grid": True,
                        "alpha": a,
                        "beta": b
                    }
                })
                div_id += 1
        setup_data = {
            "num_divisions": len(divisions),
            "divisions": divisions
        }
        if getattr(args, "verbosity", 0) >= 1:
            print(f"[Host Orchestrator] Automatically generated 3x3 variance grid across {len(divisions)} divisions.")

    num_divisions = setup_data.get("num_divisions", 1) if setup_data else 1
    divisions = setup_data.get("divisions", []) if setup_data else []

    # Prepare single run directory
    os.makedirs(run_dir, exist_ok=True)

    command_log = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_terminal_command": f"python {' '.join(sys.argv)}",
        "parsed_arguments": vars(args),
        "applied_noise_configuration": noise_config,
        "auto_grid_divisions": setup_data
    }
    with open(os.path.join(run_dir, "command_args.json"), "w") as f:
        json.dump(command_log, f, indent=4)

    if args.verbosity >= 1:
        print(f"Starting Host Orchestration Engine. Model: {model_id}")
        print(f"Run Output Directory: {os.path.abspath(run_dir)}")

    if getattr(args, "vary_params", False) and hasattr(config, "generate_trial_coefficients"):
        trial_parameter_sets = config.generate_trial_coefficients(num_trials=args.num_trials, grouping=args.grouping)
    else:
        trial_parameter_sets = [getattr(config, "TRUE_COEFFS", {}) for _ in range(args.num_trials)]

    # Map trials across divisions
    division_tasks = {}
    if num_divisions > 1 and divisions:
        base_count = args.num_trials // num_divisions
        remainder = args.num_trials % num_divisions
        
        trial_counter = 1
        for d_idx, div in enumerate(divisions):
            div_id = div.get("division_id", d_idx + 1)
            div_params = div.get("params", {})
            count = base_count + (remainder if d_idx == num_divisions - 1 else 0)
            
            div_dir = os.path.join(run_dir, f"Division_{div_id}")
            os.makedirs(div_dir, exist_ok=True)
            
            division_tasks[div_id] = {
                "dir": div_dir,
                "params": div_params,
                "trials": [],
                "submissions": [],
                "errors": []
            }
            
            for _ in range(count):
                division_tasks[div_id]["trials"].append((trial_counter, trial_parameter_sets[trial_counter - 1]))
                trial_counter += 1
    else:
        trials_dir = os.path.join(run_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)
        division_tasks[0] = {
            "dir": run_dir,
            "params": {},
            "trials": [(i + 1, trial_parameter_sets[i]) for i in range(args.num_trials)],
            "submissions": [],
            "errors": []
        }

    def launch_container_worker(div_id, trial_num, assigned_coeffs):
        div_info = division_tasks[div_id]
        div_params = div_info["params"]
        
        if div_id > 0:
            trial_folder = os.path.join(div_info["dir"], f"trial_{trial_num}")
        else:
            trial_folder = os.path.join(run_dir, "trials", f"trial_{trial_num}")

        os.makedirs(os.path.join(trial_folder, "measurements"), exist_ok=True)

        with open(os.path.join(trial_folder, "true_coeffs.json"), "w") as f:
            json.dump(assigned_coeffs, f, indent=4)

        if div_params:
            with open(os.path.join(trial_folder, "override_params.json"), "w") as f:
                json.dump(div_params, f, indent=4)

        # Inside launch_container_worker in host_orchestrator.py
        from core.configs.diffeq_config import DEFAULT_NOISE_CONFIG

        # Merge DEFAULT_NOISE_CONFIG with noise_config and division parameters
        trial_noise = DEFAULT_NOISE_CONFIG.copy()
        trial_noise.update(noise_config)
        trial_noise.update(div_params)
        if args.verbosity >= 1:
            with print_lock:
                print(f"[Host Orchestrator] Spawning Docker Container for Trial {trial_num} (Division {div_id}: alpha={trial_noise.get('alpha')}, beta={trial_noise.get('beta')})...")

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
            "--input_const_noise", str(trial_noise["input_const_noise"]),
            "--input_lin_noise", str(trial_noise["input_lin_noise"]),
            "--meas_const_noise", str(trial_noise["meas_const_noise"]),
            "--meas_lin_noise", str(trial_noise["meas_lin_noise"])
        ]

        if trial_noise.get("use_alpha_beta_grid"):
            docker_cmd.append("--use_alpha_beta_grid")
        if trial_noise.get("alpha") is not None:
            docker_cmd.extend(["--alpha", str(trial_noise["alpha"])])
        if trial_noise.get("beta") is not None:
            docker_cmd.extend(["--beta", str(trial_noise["beta"])])

        if getattr(args, "vary_params", False):
            docker_cmd.append("--vary_params")
        else:
            docker_cmd.append("--no_vary_params")
            
        if getattr(args, "plot_mmd_heatmap", False):
            docker_cmd.append("--plot_mmd_heatmap")

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
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "calculate_empirical_error.py"), trial_folder], capture_output=True, text=True, check=True)
        except Exception as e:
            with print_lock:
                print(f"[Warning] Failed to launch empirical error script: {e}")

        # Judge Container Step
        if args.judge:
            judge_model_key = args.judge
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

        # Summary Consolidation
        summary_path = os.path.join(trial_folder, "summary.json")
        summary_data = {}
        sindy_path = os.path.join(trial_folder, "sindy_report.json")
        
        timeout_seconds = 30
        poll_interval = 2
        elapsed = 0

        while elapsed < timeout_seconds:
            if os.path.exists(sindy_path):
                try:
                    with open(sindy_path, "r") as f:
                        sindy_content = json.load(f)
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(poll_interval)
            elapsed += poll_interval

        sindy_data = {}
        if os.path.exists(sindy_path):
            try:
                with open(sindy_path, "r") as f:
                    sindy_data = json.load(f)
                os.remove(sindy_path)
            except Exception as e:
                print(f"[Warning] Failed to read/remove {sindy_path}: {e}")

        judge_path = os.path.join(trial_folder, "trial_judge.json")
        judge_data = {}
        if os.path.exists(judge_path):
            try:
                with open(judge_path, "r") as f:
                    judge_data = json.load(f)
                os.remove(judge_path)
            except Exception as e:
                print(f"[Warning] Failed to merge/remove {judge_path}: {e}")

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as sf:
                    summary_data = json.load(sf)
            except Exception as e:
                print(f"[Warning] Failed to read base summary.json: {e}")

        if sindy_data:
            summary_data["sindy_report"] = sindy_data
        if judge_data:
            summary_data["judge_evaluation"] = judge_data

        for temp_file in ["true_coeffs.json", "override_params.json"]:
            host_temp_path = os.path.join(trial_folder, temp_file)
            if os.path.exists(host_temp_path):
                try:
                    os.remove(host_temp_path)
                except Exception:
                    pass

        with open(summary_path, "w") as sf:
            json.dump(summary_data, sf, indent=4)

        if summary_data:
            return div_id, trial_num, summary_data.get("submission"), summary_data.get("error"), assigned_coeffs
        return div_id, trial_num, None, None, assigned_coeffs

    # Launch worker pool
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for div_id, div_data in division_tasks.items():
            for t_num, a_coeffs in div_data["trials"]:
                futures.append(executor.submit(launch_container_worker, div_id, t_num, a_coeffs))

        for future in as_completed(futures):
            div_id, t_num, submission, error, assigned_coeffs = future.result()
            if submission is not None:
                division_tasks[div_id]["submissions"].append({"trial": t_num, "submission": submission, "true_coeffs": assigned_coeffs})
                division_tasks[div_id]["errors"].append({"trial": t_num, "error": error, "true_coeffs": assigned_coeffs})

    # Save and summarize per division
    for div_id, div_data in division_tasks.items():
        div_data["submissions"].sort(key=lambda x: x["trial"])
        div_data["errors"].sort(key=lambda x: x["trial"])

        errors_log_path = os.path.join(div_data["dir"], "errors_log.json")
        submissions_log_path = os.path.join(div_data["dir"], "submissions_log.json")

        with open(submissions_log_path, "w") as f:
            json.dump(div_data["submissions"], f, indent=4)
        with open(errors_log_path, "w") as f:
            json.dump(div_data["errors"], f, indent=4)

        process_division_analysis(div_data["dir"], args, noise_config)

    # =========================================================================
    # TOP-LEVEL RUN PLOTS (3x3 ALPHA/BETA NOISE GRID PLOTS)
    # =========================================================================
    if getattr(args, "use_alpha_beta_grid", False) or num_divisions > 1:
        top_plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(top_plots_dir, exist_ok=True)

        all_run_summaries = []
        for div_id, div_data in division_tasks.items():
            div_dir = div_data["dir"]
            
            # Search both div_dir and div_dir/trials for trial directories
            search_dirs = [div_dir]
            trials_subdir = os.path.join(div_dir, "trials")
            if os.path.exists(trials_subdir):
                search_dirs.append(trials_subdir)

            trial_dirs = []
            for s_dir in search_dirs:
                if os.path.exists(s_dir):
                    for d in os.listdir(s_dir):
                        full_path = os.path.join(s_dir, d)
                        if d.startswith("trial_") and os.path.isdir(full_path):
                            trial_dirs.append(full_path)

            for t_dir in sorted(trial_dirs):
                s_file = os.path.join(t_dir, "summary.json")
                if os.path.exists(s_file):
                    try:
                        with open(s_file, "r") as sf:
                            s_data = json.load(sf)
                            override = s_data.get("override_params", {})
                            
                            s_data["alpha"] = override["alpha"]
                            s_data["beta"] = override["beta"]
                            
                            all_run_summaries.append(s_data)
                    except Exception as e:
                        print(f"[Warning] Failed to parse summary at {s_file}: {e}")

        print(f"[Host Orchestrator] Diagnostic: Found {len(all_run_summaries)} total trial summaries across all divisions.")

        compiled_run_summary_path = os.path.join(run_dir, "compiled_run_summary.json")
        with open(compiled_run_summary_path, "w") as f:
            json.dump(all_run_summaries, f, indent=4)

        plot_3x3_script = os.path.join("core", "plot_core", "plot_3x3.py")
        if os.path.exists(plot_3x3_script):
            print(f"[Host Orchestrator] Executing 3x3 plot generator script: {plot_3x3_script}")
            grid_cmd = [
                sys.executable, plot_3x3_script,
                "--summary_path", compiled_run_summary_path,
                "--output_dir", top_plots_dir
            ]
            grid_res = subprocess.run(grid_cmd, capture_output=True, text=True)
            
            if grid_res.stdout.strip():
                print(f"[3x3 Plot Generator Output]\n{grid_res.stdout.strip()}")
            
            if grid_res.returncode != 0:
                print(f"[Plotting Error - 3x3 Grid Plots]\nSTDOUT: {grid_res.stdout}\nSTDERR: {grid_res.stderr}")
            else:
                print(f"[Host Orchestrator] Successfully generated 3x3 grid plots in: {top_plots_dir}")

    try:
        if args.verbosity >= 1:
            print(f"\n[Host Orchestrator] Auto-compiling benchmark data for run: {run_dir}...")
        
        compiled_df = compile_data(base_dir=run_dir)
        if compiled_df is not None and not compiled_df.empty:
            output_csv_path = os.path.join(run_dir, "compiled_data.csv")
            compiled_df.to_csv(output_csv_path, index=False)
            if args.verbosity >= 1:
                print(f"  -> Compiled CSV saved to: {output_csv_path}")
        else:
            print("[Warning] Compiled DataFrame is empty. No CSV generated.")
    except Exception as e:
        print(f"[Warning] Automatic data compilation failed: {e}")

    if args.verbosity >= 1:
        print(f"\n{'='*70}")
        print(f"[Benchmark Complete] Orchestration engine finished.")
        print(f"All tracking data, trajectory files, and plots saved to:\n  -> {os.path.abspath(run_dir)}")
        print(f"{'='*70}\n")

    return run_dir
