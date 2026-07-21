import hashlib
import json

RUBRIC_VERSION = "1.1.0"

RUBRIC_CATEGORIES = {
    "choice_of_measurements": {
        "name": "Choice of Measurements & Experimental Design",
        "weight": 1.0,
        "checkpoints": {
            "varied_timescales": {
                "prompt": "Did the agent systematically vary delta_t across micro and macro timescales rather than staying at a fixed step?",
                "options": {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
            },
            "decoupled_state_variables": {
                "prompt": "Did the agent design specific experiments to isolate velocity or displacement boundary conditions?",
                "options": {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
            }
        }
    },
    "use_of_tools_and_packages": {
        "name": "Correct Use of Packages & Python Sandbox",
        "weight": 1.0,
        "checkpoints": {
            "acceleration_estimation": {
                "prompt": "Did the agent estimate acceleration via numerical derivatives/filtering rather than regressing velocity against displacement?",
                "options": {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
            },
            "advanced_regression_methods": {
                "prompt": "Did the agent execute Python code using regression methods (OLS, SINDy, Splines, etc.) to extract coefficients?",
                "options": {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
            }
        }
    },
    "reasoning_and_discipline": {
        "name": "Reasoning Process & Scientific Discipline",
        "weight": 1.0,
        "checkpoints": {
            "out_of_sample_validation": {
                "prompt": "Did the agent test its discovered equation against unqueried state points before submission?",
                "options": {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
            },
            "loop_avoidance": {
                "prompt": "Did the agent avoid getting trapped in repetitive textual loops or unverified guesses?",
                "options": {"YES": 1.0, "NO": 0.0}
            }
        }
    },
    # PROGRAMMATIC CATEGORIES (Calculated via Python, not LLM)
    "statistical_goodness_of_fit": {
        "name": "Statistical Rigor & Chi-Squared Validation",
        "weight": 1.0,
        "checkpoints": {
            "chi2_acceptance_band": {
                "prompt": "PROGRAMMATIC: Did the agent compute an ensemble Chi-Squared test where the reduced Chi-Squared falls within the valid statistical acceptance band (0.5 to 2.0)?",
                "options": {"YES": 1.0, "NO": 0.0}
            }
        }
    },
    "structural_term_discovery": {
        "name": "Correct Discovered Equation Structure",
        "weight": 1.0,
        "checkpoints": {
            "exact_term_match": {
                "prompt": "PROGRAMMATIC: Did the agent discover the exact set of active forcing terms in the differential equation?",
                "options": {"YES": 1.0, "NO": 0.0}
            }
        }
    },
    "empirical_uncertainty_accuracy": {
        "name": "Target Coefficient (k_0) Accuracy within Uncertainty",
        "weight": 1.0,
        "checkpoints": {
            "within_empirical_uncertainty": {
                "prompt": "PROGRAMMATIC: Is the relative error of predicted k_0 within the calculated empirical noise threshold?",
                "options": {"YES": 1.0, "NO": 0.0}
            }
        }
    }
}

def get_rubric_hash():
    rubric_str = json.dumps(RUBRIC_CATEGORIES, sort_keys=True)
    return hashlib.sha256(rubric_str.encode("utf-8")).hexdigest()[:10]
