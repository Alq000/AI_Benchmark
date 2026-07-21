import argparse
import json
import os
import sys
from dotenv import load_dotenv

from benchmark_core.agent_runner import run_trial
from benchmark_core.config_loader import load_config
from benchmark_core.orchestrator import run_orchestration
from benchmark_core.statistical_validation import compute_ensemble_chi_squared

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# =========================================================================
# SCRIPT DEFAULTS (Bottom of the Hierarchy)
# =========================================================================
DEFAULT_VARY_PARAMS = False
DEFAULT_ALLOW_CUSTOM_INITIAL_CONDITIONS = False


def str2bool(v):
    """Converts standard string inputs to boolean for CLI flags."""
    if v is None or isinstance(v, bool):
        return v
    if v.lower() in ("true"):
        return True
    elif v.lower() in ("false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (e.g., True/False, 1/0, yes/no).")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM DiffEq Benchmark with Dynamic Noise Framework"
    )
    parser.add_argument("--model", type=str, required=True, help="Agent shorthand")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--diff_eq_config", type=str, required=True, help="Path to config file")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3, 4], default=4)
    parser.add_argument("--plotting", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--max_workers", type=int, default=1)

    # CLI Flags allowing explicit boolean values (nargs="?" allows omitting the value to default to True)
    parser.add_argument(
        "--vary_params",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        metavar="True/False",
        help="Vary coefficients using LHS across trials (e.g., --vary_params True/False)",
    )
    parser.add_argument(
        "--custom_initial_conditions",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        metavar="True/False",
        help="Allow agent to specify x0/v0 (e.g., --custom_initial_conditions True/False)",
    )

    parser.add_argument(
        "--judge",
        nargs="?",
        const="gpt5nano",
        default=None,
        help="Enable LLM Judge evaluation.",
    )
    parser.add_argument("--input_const_noise", type=float, default=None)
    parser.add_argument("--input_lin_noise", type=float, default=None)
    parser.add_argument("--meas_const_noise", type=float, default=None)
    parser.add_argument("--meas_lin_noise", type=float, default=None)

    parser.add_argument("--internal_trial_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal_output_dir", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    config = load_config(args.diff_eq_config)

    # =========================================================================
    # RESOLUTION HIERARCHY: CLI > Config File > run_benchmark Default
    # =========================================================================

    # 1. Resolve VARY_PARAMS
    if args.vary_params is not None:
        final_vary_params = args.vary_params
    elif getattr(config, "VARY_PARAMS", None) is not None:
        final_vary_params = config.VARY_PARAMS
    else:
        final_vary_params = DEFAULT_VARY_PARAMS
    # 2. Resolve CUSTOM_INITIAL_CONDITIONS
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
    # INTERNAL CONTAINER EXECUTION (Runs within Docker)
    # =========================================================================
    if args.internal_trial_id is not None:
        measurements_dir = os.path.join(args.internal_output_dir, "measurements")
        os.makedirs(measurements_dir, exist_ok=True)
        
        dynamic_coeffs = getattr(config, "TRUE_COEFFS", {}).copy()
        coeffs_path = os.path.join(args.internal_output_dir, "true_coeffs.json")
        if os.path.exists(coeffs_path):
            with open(coeffs_path, "r") as f:
                dynamic_coeffs = json.load(f)

        max_turns = getattr(config, "MAX_TURNS", 25)

        submission, error, chat_log = run_trial(
            args.internal_trial_id, config, args.model, args.verbosity, 
            args.plotting, args.diff_eq_config, measurements_dir, noise_config, dynamic_coeffs, OPENROUTER_API_KEY,
            args.custom_initial_conditions,
            max_turns=max_turns
        )
        
        # --- NEW: Inject Statistical Evaluation Here ---
        if submission and "discovered_terms" in submission:
            try:
                x_sym, v_sym = sp.symbols('x v')
                accel_expr = 0
                for entry in submission["discovered_terms"]:
                    term_expr = sp.parse_expr(entry['term']).subs(sp.Symbol('x_dot'), v_sym) 
                    accel_expr += float(entry['coeff']) * term_expr
                
                accel_func = sp.lambdify((x_sym, v_sym), accel_expr, modules='numpy')
                
                def discovered_ode(t, y):
                    return [y[1], float(accel_func(y[0], y[1]))]
                
                master_file = os.path.join(measurements_dir, "all_compiled_experiments.json")
                if os.path.exists(master_file):
                    system_stats = compute_ensemble_chi_squared(
                        master_file, discovered_ode, noise_config, len(submission["discovered_terms"])
                    )
                    submission["statistical_validation"] = system_stats
                    
                    if args.verbosity >= 1:
                        print(f"\n[System] Independently computed statistical validation: {system_stats}")
            except Exception as e:
                if args.verbosity >= 1:
                    print(f"\n[System] Failed to compute internal stats: {e}")

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
    run_orchestration(args, config, noise_config, OPENROUTER_API_KEY)

if __name__ == "__main__":
    main()
