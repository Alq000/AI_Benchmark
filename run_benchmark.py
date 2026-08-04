import argparse
import json
import os
import sys
import subprocess
import sympy as sp
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath("core"))

from test_core.config_loader import load_config
from test_core.agent_runner import run_trial
from test_core.orchestrator import run_orchestration

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_VARY_PARAMS = False
DEFAULT_ALLOW_CUSTOM_INITIAL_CONDITIONS = False


def str2bool(v):
    if v is None or isinstance(v, bool):
        return v
    if v.lower() in ("true"):
        return True
    elif v.lower() in ("false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def safe_read_and_remove_json(file_path):
    """Reads JSON content from file_path if present, then safely deletes the file."""
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            os.remove(file_path)
        except Exception as e:
            print(f"[Warning] Failed to read/remove {file_path}: {e}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM DiffEq Benchmark with Dynamic Noise Framework"
    )
    parser.add_argument("--model", type=str, required=True, help="Agent shorthand")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--diff_eq_config", type=str, default="core/configs/diffeq_config.py", help="Path to config file")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3, 4], default=4)
    parser.add_argument("--plotting", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--max_workers", type=int, default=1)

    parser.add_argument("--vary_params", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--custom_initial_conditions", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--judge", nargs="?", const="gpt5nano", default=None)
    parser.add_argument("--grouping", type=int, default=0)
    parser.add_argument("--vary_setup", type=str, default=None)

    parser.add_argument("--input_const_noise", type=float, default=None)
    parser.add_argument("--input_lin_noise", type=float, default=None)
    parser.add_argument("--meas_const_noise", type=float, default=None)
    parser.add_argument("--meas_lin_noise", type=float, default=None)

    parser.add_argument("--internal_trial_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal_output_dir", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    config = load_config(args.diff_eq_config)
    config.GROUPING = args.grouping

    # Resolution hierarchy
    if args.vary_params is not None:
        final_vary_params = args.vary_params
    elif getattr(config, "VARY_PARAMS", None) is not None:
        final_vary_params = config.VARY_PARAMS
    else:
        final_vary_params = DEFAULT_VARY_PARAMS

    if args.custom_initial_conditions is not None:
        final_allow_custom_ic = args.custom_initial_conditions
    elif getattr(config, "ALLOW_CUSTOM_INITIAL_CONDITIONS", None) is not None:
        final_allow_custom_ic = config.ALLOW_CUSTOM_INITIAL_CONDITIONS
    else:
        final_allow_custom_ic = DEFAULT_ALLOW_CUSTOM_INITIAL_CONDITIONS

    args.custom_initial_conditions = final_allow_custom_ic
    args.vary_params = final_vary_params

    noise_config = {
        "input_const_noise": args.input_const_noise if args.input_const_noise is not None else config.DEFAULT_NOISE_CONFIG["input_const_noise"],
        "input_lin_noise": args.input_lin_noise if args.input_lin_noise is not None else config.DEFAULT_NOISE_CONFIG["input_lin_noise"],
        "meas_const_noise": args.meas_const_noise if args.meas_const_noise is not None else config.DEFAULT_NOISE_CONFIG["meas_const_noise"],
        "meas_lin_noise": args.meas_lin_noise if args.meas_lin_noise is not None else config.DEFAULT_NOISE_CONFIG["meas_lin_noise"]
    }

    # =========================================================================
    # INTERNAL CONTAINER EXECUTION (Runs inside Docker per trial)
    # =========================================================================
    if args.internal_trial_id is not None:
        target_dir = args.internal_output_dir
        measurements_dir = os.path.join(target_dir, "measurements")
        os.makedirs(measurements_dir, exist_ok=True)

        # 1. Handle Override Params File
        override_path = os.path.join(target_dir, "override_params.json")
        override_params_data = safe_read_and_remove_json(override_path)
        for k, v in override_params_data.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif k in noise_config:
                noise_config[k] = v
            else:
                setattr(config, k, v)

        # 2. Handle True Coefficients File
        coeffs_path = os.path.join(target_dir, "true_coeffs.json")
        true_coeffs_data = safe_read_and_remove_json(coeffs_path)
        if not true_coeffs_data:
            true_coeffs_data = getattr(config, "TRUE_COEFFS", {}).copy()

        max_turns = getattr(config, "MAX_TURNS", 25)

        # Run agent
        submission, error, chat_log = run_trial(
            args.internal_trial_id, config, args.model, args.verbosity,
            args.plotting, args.diff_eq_config, measurements_dir, noise_config, true_coeffs_data, OPENROUTER_API_KEY,
            args.custom_initial_conditions,
            max_turns=max_turns
        )

        # 3. Handle Stat Validation File
        stat_val_path = os.path.join(target_dir, "latest_stat_validation.json")
        stat_val_data = safe_read_and_remove_json(stat_val_path)

        # 4. Handle SINDy Report File
        sindy_path = os.path.join(target_dir, "sindy_report.json")
        sindy_data = safe_read_and_remove_json(sindy_path)

        # Write trial execution history
        trial_data = {"trial_id": args.internal_trial_id, "status": "success" if submission else "failed", "chat_history": chat_log}
        with open(os.path.join(target_dir, f"trial_{args.internal_trial_id}.json"), "w") as f:
            json.dump(trial_data, f, indent=4)


        # Write merged summary.json
        summary_payload = {
            "submission": submission,
            "error": error,
            "true_coeffs": true_coeffs_data,
            "override_params": override_params_data,
            "latest_statistical_validation": stat_val_data
        }

        with open(os.path.join(target_dir, "summary.json"), "w") as f:
            json.dump(summary_payload, f, indent=4)

        sys.exit(0)
    # Host execution
    run_dir = run_orchestration(args, config, noise_config, OPENROUTER_API_KEY)


if __name__ == "__main__":
    main()
