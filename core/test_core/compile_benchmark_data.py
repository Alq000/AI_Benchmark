import json
import os
import re
import argparse
import pandas as pd
from pathlib import Path

def safe_get_dict(obj, key):
    """Safely retrieves a dictionary from a key, preventing NoneType attribute errors."""
    if not isinstance(obj, dict):
        return {}
    res = obj.get(key)
    return res if isinstance(res, dict) else {}

def compile_data(base_dir):
    """
    Crawls the target directory structure and compiles all trial attributes into a DataFrame.
    """
    base_path = Path(base_dir).resolve()
    all_trials_data = []
    
    # Locate all trial directories
    trial_dirs = [p for p in base_path.rglob("trial_*") if p.is_dir()]
    
    if not trial_dirs:
        print(f"Warning: No 'trial_*' directories found inside {base_path}")
        return pd.DataFrame()

    for t_dir in trial_dirs:
        match = re.search(r'trial_(\d+)', t_dir.name)
        if not match:
            continue
        trial_id = int(match.group(1))
        
        # Determine agent_model and run_id based on the directory layout:
        # results / {agent_model} / {run_timestamp} / trials / trial_N
        parts = t_dir.parts
        if 'results' in parts:
            idx = parts.index('results')
            agent_model = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
            run_id = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
        else:
            if t_dir.parent.name == "trials":
                run_id = t_dir.parent.parent.name
                agent_model = t_dir.parent.parent.parent.name
            else:
                run_id = t_dir.parent.name
                agent_model = t_dir.parent.parent.name

        trial_row = {
            "agent_model": agent_model,
            "run_id": run_id,
            "trial_id": trial_id
        }
        
        run_dir = t_dir.parent.parent if t_dir.parent.name == "trials" else t_dir.parent

        # 1. Parse Run-Level Directory: command_args.json
        cmd_file = run_dir / "command_args.json"
        if cmd_file.exists():
            try:
                with open(cmd_file, 'r') as f:
                    cmd_data = json.load(f) or {}
                    trial_row["execution_timestamp"] = cmd_data.get("execution_timestamp")
                    trial_row["raw_terminal_command"] = cmd_data.get("raw_terminal_command")
                    
                    noise = safe_get_dict(cmd_data, "applied_noise_configuration") or cmd_data
                    trial_row["global_input_const_noise"] = noise.get("input_const_noise")
                    trial_row["global_input_lin_noise"] = noise.get("input_lin_noise")
                    trial_row["global_meas_const_noise"] = noise.get("meas_const_noise")
                    trial_row["global_meas_lin_noise"] = noise.get("meas_lin_noise")
            except Exception:
                pass

        # 2. Parse Trial Directory: true_coeffs.json
        true_file = t_dir / "true_coeffs.json"
        if true_file.exists():
            try:
                with open(true_file, 'r') as f:
                    true_data = json.load(f) or {}
                    for k, v in true_data.items():
                        if isinstance(v, (int, float, str, bool)):
                            trial_row[f"true_{k}"] = v
            except Exception:
                pass

        # 3. Parse Trial Directory: calculated_error.json (if present)
        calc_file = t_dir / "calculated_error.json"
        if calc_file.exists():
            try:
                with open(calc_file, 'r') as f:
                    calc_data = json.load(f) or {}
                    trial_row["calc_estimated_k0"] = calc_data.get("estimated_k0", calc_data.get("k0_hat"))
                    trial_row["calc_signed_error"] = calc_data.get("signed_error")
                    trial_row["calc_absolute_error"] = calc_data.get("error", calc_data.get("absolute_error"))
            except Exception:
                pass

        # 4. Parse Trial Directory: summary.json
        sum_file = t_dir / "summary.json"
        if sum_file.exists():
            try:
                with open(sum_file, 'r') as f:
                    sum_data = json.load(f) or {}
                    trial_row["summary_error"] = sum_data.get("error")
                    trial_row["summary_true_k0"] = sum_data.get("true_k0")
                    
                    emp_unc = safe_get_dict(sum_data, "empirical_uncertainty")
                    for k, v in emp_unc.items():
                        trial_row[f"summary_emp_{k}"] = v
                    
                    submission = safe_get_dict(sum_data, "submission")
                    
                    # Extract estimated_k0 from discovered terms if term is "x"
                    discovered_terms = submission.get("discovered_terms", [])
                    trial_row["summary_num_discovered_terms"] = len(discovered_terms) if isinstance(discovered_terms, list) else 0
                    
                    estimated_k0 = None
                    if isinstance(discovered_terms, list):
                        for term_dict in discovered_terms:
                            if isinstance(term_dict, dict) and term_dict.get("term") == "x":
                                estimated_k0 = term_dict.get("coeff")
                                break
                    trial_row["summary_estimated_k0"] = estimated_k0

                    # Extract the 7 statistical validation fields from latest run
                    stat_val = safe_get_dict(sum_data, "latest_statistical_validation")
                    trial_row["summary_stat_calc_const_noise"] = stat_val.get("calc_const_noise")
                    trial_row["summary_stat_calc_lin_noise"] = stat_val.get("calc_lin_noise")
                    trial_row["summary_stat_chi2_statistic"] = stat_val.get("chi2_statistic")
                    trial_row["summary_stat_degrees_of_freedom"] = stat_val.get("degrees_of_freedom")
                    trial_row["summary_stat_reduced_chi2"] = stat_val.get("reduced_chi2")
                    trial_row["summary_stat_p_value"] = stat_val.get("p_value")
                    trial_row["summary_stat_null_hypothesis_accepted"] = stat_val.get("null_hypothesis_accepted")
                        
            except Exception:
                pass
        # 5. Parse Trial Directory: trial_judge.json
        judge_file = t_dir / "trial_judge.json"
        if not judge_file.exists(): # Fallback to older naming if needed
            judge_file = t_dir / "judge_evaluation.json"
            
        if judge_file.exists():
            try:
                with open(judge_file, 'r') as f:
                    j_data = json.load(f) or {}
                    trial_row["judge_rubric_version"] = j_data.get("rubric_version")
                    trial_row["judge_model"] = j_data.get("judge_model")
                    trial_row["judge_final_weighted_score"] = j_data.get("final_weighted_score")
                    
                    cat_scores = safe_get_dict(j_data, "category_scores")
                    for cat, score in cat_scores.items():
                        trial_row[f"judge_score_cat_{cat}"] = score
                        
                    checkpoints = safe_get_dict(j_data, "checkpoint_details")
                    for cp_name, cp_data in checkpoints.items():
                        safe_cp_name = str(cp_name).replace(".", "_")
                        if isinstance(cp_data, dict):
                            trial_row[f"judge_cp_selection_{safe_cp_name}"] = cp_data.get("selection")
                            trial_row[f"judge_cp_points_{safe_cp_name}"] = cp_data.get("points")
                            
                    qual = safe_get_dict(j_data, "qualitative_summary")
                    trial_row["judge_qualitative_what_went_right"] = qual.get("what_went_right")
                    trial_row["judge_qualitative_what_went_wrong"] = qual.get("what_went_wrong")
            except Exception:
                pass

        # 6. Parse Trial Directory: sindy_report.json
        sindy_file = t_dir / "sindy_report.json"
        if sindy_file.exists():
            try:
                with open(sindy_file, 'r') as f:
                    sindy_data = json.load(f) or {}
                    for k, v in sindy_data.items():
                        trial_row[f"sindy_{k}"] = v
            except Exception:
                pass

        # 7. Parse Measurements: all_compiled_experiments.json
        measurements_file = t_dir / "measurements" / "all_compiled_experiments.json"
        if measurements_file.exists():
            try:
                with open(measurements_file, 'r') as f:
                    meas_data = json.load(f) or {}
                    
                    num_exp_calls = len(meas_data.keys())
                    t_starts = []
                    delta_ts = []
                    steps_list = []
                    total_data_points = 0
                    
                    for run_key, run_info in meas_data.items():
                        if not isinstance(run_info, dict):
                            continue
                            
                        t_starts.append(run_info.get("t_start", 0.0))
                        delta_ts.append(run_info.get("delta_t", 0.0))
                        
                        steps = run_info.get("steps_per_trajectory", 0)
                        steps_list.append(steps)
                        
                        trajectories = run_info.get("trajectories", [])
                        total_data_points += (steps * len(trajectories))
                    
                    trial_row["num_experiment_calls"] = num_exp_calls
                    trial_row["exp_t_start_list"] = str(t_starts)
                    trial_row["exp_delta_t_list"] = str(delta_ts)
                    trial_row["exp_steps_per_trajectory_list"] = str(steps_list)
                    
                    # Integer averages as requested
                    trial_row["exp_avg_t_start"] = int(sum(t_starts) / len(t_starts)) if t_starts else 0
                    trial_row["exp_avg_delta_t"] = int(sum(delta_ts) / len(delta_ts)) if delta_ts else 0
                    trial_row["exp_avg_steps_per_trajectory"] = int(sum(steps_list) / len(steps_list)) if steps_list else 0
                    
                    trial_row["total_data_points"] = total_data_points
            except Exception:
                pass

        # 8. Parse Trial Chat History File (trial_X.json or trail_X.json)
        # Check both naming conventions just in case
        chat_files = list(t_dir.glob(f"trial_{trial_id}.json")) + list(t_dir.glob(f"trail_{trial_id}.json"))
        if chat_files:
            chat_file = chat_files[0]
            try:
                with open(chat_file, 'r') as f:
                    chat_data = json.load(f) or {}
                    
                    trial_row["trial_status"] = chat_data.get("status", trial_row.get("trial_status"))
                    
                    history = chat_data.get("chat_history", chat_data.get("history", []))
                    if isinstance(history, list):
                        num_assistant_messages = sum(1 for msg in history if isinstance(msg, dict) and msg.get("role") == "assistant")
                        trial_row["num_assistant_messages"] = num_assistant_messages
            except Exception:
                pass

        all_trials_data.append(trial_row)

    df = pd.DataFrame(all_trials_data)
    
    if not df.empty and 'run_id' in df.columns and 'trial_id' in df.columns:
        df = df.sort_values(['agent_model', 'run_id', 'trial_id']).reset_index(drop=True)
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile exhaustive benchmark data into a CSV.")
    parser.add_argument("--run-dir", type=str, required=True, 
                        help="Path to a single run directory or a parent directory containing multiple runs.")
    parser.add_argument("--output", type=str, default="EXHAUSTIVE_benchmark_compiled.csv",
                        help="Output CSV filename.")
    
    args = parser.parse_args()
    
    print(f"Compiling exhaustive benchmark data from: {args.run_dir}...")
    final_df = compile_data(args.run_dir)
    
    if not final_df.empty:
        final_df.to_csv(args.output, index=False)
        print(f"Done. Extracted {len(final_df.columns)} attributes across {len(final_df)} trials.")
        print(f"Saved to {args.output}")
    else:
        print("Compilation failed. No valid data found to save.")
