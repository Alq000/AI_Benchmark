import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure parent directory is in Python path for importing configs
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from core.configs.diffeq_config import (
        DEFAULT_NOISE_CONFIG,
        SYSTEMATIC_BIAS,
        true_sigma_x,  # CHANGED: Imported true_sigma_x
        true_sigma_v,  # CHANGED: Imported true_sigma_v
    )
except ImportError:
    try:
        from configs.diffeq_config import (
            DEFAULT_NOISE_CONFIG,
            SYSTEMATIC_BIAS,
            true_sigma_x,  # CHANGED: Imported true_sigma_x
            true_sigma_v,  # CHANGED: Imported true_sigma_v
        )
    except ImportError:
        DEFAULT_NOISE_CONFIG = {
            "sigma_0_x": 0.1,
            "sigma_1_x": 0.05,
            "sigma_0_v": 0.1,
            "sigma_1_v": 0.05,
        }
        SYSTEMATIC_BIAS = {
            "x_c_0": 0.8,
            "x_c_1": 0.1,
            "v_c_0": 0.2,
            "v_c_1": 0.03,
        }

        # CHANGED: Added fallback dummy definitions if import fails entirely
        def true_sigma_x(x, v, t, noise_config=None):
            cfg = noise_config or DEFAULT_NOISE_CONFIG
            alpha, beta = cfg.get("alpha", 1.0), cfg.get("beta", 1.0)
            return np.sqrt(alpha * (cfg["sigma_0_x"]**2) + beta * (cfg["sigma_1_x"]**2) * (np.asarray(x)**2))

        def true_sigma_v(x, v, t, noise_config=None):
            cfg = noise_config or DEFAULT_NOISE_CONFIG
            alpha, beta = cfg.get("alpha", 1.0), cfg.get("beta", 1.0)
            return np.sqrt(alpha * (cfg["sigma_0_v"]**2) + beta * (cfg["sigma_1_v"]**2) * (np.asarray(v)**2))

def load_division_summary(summary_path):
    print(f"[Diagnostic] Loading summary data from: {summary_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary path does not exist: {summary_path}")
    with open(summary_path, "r") as f:
        data = json.load(f)
    print(f"[Diagnostic] Successfully loaded {len(data)} trial summary record(s).")
    return data


def extract_alpha_beta(trial):
    """Extract alpha and beta safely from either top-level or override_params."""
    if "alpha" in trial and "beta" in trial:
        return float(trial["alpha"]), float(trial["beta"])
    override = trial.get("override_params", {})
    alpha = float(override["alpha"])
    beta  = float(override["beta"])
    return alpha, beta


def safe_eval_sigma_func(func, state_domain, is_v_domain=False):
    """Helper to safely evaluate vectorized or scalar sigma functions."""
    if not callable(func):
        return np.full_like(state_domain, np.nan)

    zeros = np.zeros_like(state_domain)
    try:
        if is_v_domain:
            res = func(zeros, state_domain, zeros)
        else:
            res = func(state_domain, zeros, zeros)

        res_arr = np.asarray(res, dtype=float)
        if res_arr.ndim == 0:
            res_arr = np.full_like(state_domain, res_arr)
        return res_arr
    except Exception as vec_err:
        print(
            f"[Diagnostic] Vectorized call failed ({vec_err}). Trying scalar"
            " loop..."
        )
        sigma_list = []
        for val in state_domain:
            try:
                x_val = 0.0 if is_v_domain else val
                v_val = val if is_v_domain else 0.0
                res_scalar = float(func(x_val, v_val, 0.0))
                sigma_list.append(res_scalar)
            except Exception:
                sigma_list.append(np.nan)
        return np.array(sigma_list, dtype=float)


# ← NEW FUNCTION ─────────────────────────────────────────────────────────────
def safe_eval_bias_func(func, domain, mode):
    """Helper to safely evaluate vectorized or scalar bias functions.

    mode 'x': calls func(domain, zeros, zeros)   – vary x, v=0, t=0
    mode 'v': calls func(zeros, domain, zeros)   – vary v, x=0, t=0
    mode 't': calls func(zeros, zeros, domain)   – vary t, x=0, v=0
    """
    if not callable(func):
        return np.full_like(domain, np.nan)

    zeros = np.zeros_like(domain)
    try:
        if mode == "x":
            res = func(domain, zeros, zeros)
        elif mode == "v":
            res = func(zeros, domain, zeros)
        elif mode == "t":
            res = func(zeros, zeros, domain)
        else:
            raise ValueError(f"Unknown bias eval mode: {mode}")

        res_arr = np.asarray(res, dtype=float)
        if res_arr.ndim == 0:
            res_arr = np.full_like(domain, res_arr)
        return res_arr
    except Exception as vec_err:
        print(
            f"[Diagnostic] Bias vectorized call failed ({vec_err}). Trying"
            " scalar loop..."
        )
        result = []
        for val in domain:
            try:
                if mode == "x":
                    r = float(func(val, 0.0, 0.0))
                elif mode == "v":
                    r = float(func(0.0, val, 0.0))
                elif mode == "t":
                    r = float(func(0.0, 0.0, val))
                else:
                    r = np.nan
                result.append(r)
            except Exception:
                result.append(np.nan)
        return np.array(result, dtype=float)
# ─────────────────────────────────────────────────────────────────────────────


def compute_trial_evaluations(trial, domain, t_domain_bias, v_domain_bias):   # ← CHANGED: added t_domain_bias, v_domain_bias (was: def compute_trial_evaluations(trial, domain))
    sub         = trial.get("submission", {})
    code_string = sub.get("code", "")

    exec_scope = {"np": np, "__builtins__": __builtins__}
    try:
        exec(code_string, exec_scope)
    except Exception as e:
        print(f"[Warning] Failed to execute code string for trial: {e}")

    sigma_x_func = exec_scope.get("sigma_x")
    sigma_v_func = exec_scope.get("sigma_v")

    # CHANGED: Constructed noise configuration dictionary for true_sigma imported functions
    alpha, beta = extract_alpha_beta(trial)
    trial_noise_config = {
        **DEFAULT_NOISE_CONFIG,
        "alpha": alpha,
        "beta": beta,
        "use_alpha_beta_grid": True,
    }

    # --- EVALUATIONS FOR X (sigma) ---
    # CHANGED: Evaluated true_sigma_x directly using imported function
    sigma_true_x  = true_sigma_x(domain, 0.0, 0.0, noise_config=trial_noise_config)
    sigma_true_x0 = float(true_sigma_x(0.0, 0.0, 0.0, noise_config=trial_noise_config))
    sigma_agent_x = safe_eval_sigma_func(sigma_x_func, domain, is_v_domain=False)

    mid_idx = len(domain) // 2
    sigma_agent_x0 = (
        float(sigma_agent_x[mid_idx]) if len(sigma_agent_x) > mid_idx else np.nan
    )

    # --- EVALUATIONS FOR V (sigma) ---
    # CHANGED: Evaluated true_sigma_v directly using imported function
    sigma_true_v  = true_sigma_v(0.0, domain, 0.0, noise_config=trial_noise_config)
    sigma_true_v0 = float(true_sigma_v(0.0, 0.0, 0.0, noise_config=trial_noise_config))
    sigma_agent_v = safe_eval_sigma_func(sigma_v_func, domain, is_v_domain=True)

    sigma_agent_v0 = (
        float(sigma_agent_v[mid_idx]) if len(sigma_agent_v) > mid_idx else np.nan
    )
    # ── NEW BLOCK: bias evaluations ───────────────────────────────────────────
    bias_x_func = exec_scope["bias_x"]                                          # ← NEW
    bias_v_func = exec_scope["bias_v"]                                          # ← NEW

    c_x_0 = SYSTEMATIC_BIAS["x_c_0"]                                           # ← NEW
    c_x_1 = SYSTEMATIC_BIAS["x_c_1"]                                           # ← NEW
    c_v_0 = SYSTEMATIC_BIAS["v_c_0"]                                           # ← NEW
    c_v_1 = SYSTEMATIC_BIAS["v_c_1"]                                           # ← NEW

    # True bias arrays over each domain (at the zero-value of the other vars)
    true_bias_x_t = c_x_0 + c_x_1 * t_domain_bias                             # ← NEW
    true_bias_x_x = np.full_like(domain,        c_x_0)                         # ← NEW (no x-dep at t=0)
    true_bias_x_v = np.full_like(v_domain_bias, c_x_0)                         # ← NEW (no v-dep at t=0)
    true_bias_x_ref = float(c_x_0)                                              # ← NEW (at x=0,v=0,t=0)

    true_bias_v_t = c_v_0 + c_v_1 * t_domain_bias                             # ← NEW
    true_bias_v_x = np.full_like(domain,        c_v_0)                         # ← NEW
    true_bias_v_v = np.full_like(v_domain_bias, c_v_0)                         # ← NEW
    true_bias_v_ref = float(c_v_0)                                              # ← NEW

    # Agent bias arrays
    agent_bias_x_t = safe_eval_bias_func(bias_x_func, t_domain_bias, "t")      # ← NEW
    agent_bias_x_x = safe_eval_bias_func(bias_x_func, domain,        "x")      # ← NEW
    agent_bias_x_v = safe_eval_bias_func(bias_x_func, v_domain_bias, "v")      # ← NEW
    _rx = safe_eval_bias_func(bias_x_func, np.array([0.0]), "t")               # ← NEW
    agent_bias_x_ref = float(_rx[0])                                            # ← NEW

    agent_bias_v_t = safe_eval_bias_func(bias_v_func, t_domain_bias, "t")      # ← NEW
    agent_bias_v_x = safe_eval_bias_func(bias_v_func, domain,        "x")      # ← NEW
    agent_bias_v_v = safe_eval_bias_func(bias_v_func, v_domain_bias, "v")      # ← NEW
    _rv = safe_eval_bias_func(bias_v_func, np.array([0.0]), "t")               # ← NEW
    agent_bias_v_ref = float(_rv[0])                                            # ← NEW
    # ── END NEW BLOCK ─────────────────────────────────────────────────────────

    return {
        # existing sigma fields (unchanged)
        "sigma_true_x":   sigma_true_x,
        "sigma_agent_x":  sigma_agent_x,
        "sigma_true_x0":  sigma_true_x0,
        "sigma_agent_x0": sigma_agent_x0,
        "sigma_true_v":   sigma_true_v,
        "sigma_agent_v":  sigma_agent_v,
        "sigma_true_v0":  sigma_true_v0,
        "sigma_agent_v0": sigma_agent_v0,
        # ← NEW: bias fields
        "true_bias_x_t":   true_bias_x_t,     # ← NEW
        "true_bias_x_x":   true_bias_x_x,     # ← NEW
        "true_bias_x_v":   true_bias_x_v,     # ← NEW
        "true_bias_x_ref": true_bias_x_ref,   # ← NEW
        "agent_bias_x_t":   agent_bias_x_t,   # ← NEW
        "agent_bias_x_x":   agent_bias_x_x,   # ← NEW
        "agent_bias_x_v":   agent_bias_x_v,   # ← NEW
        "agent_bias_x_ref": agent_bias_x_ref, # ← NEW
        "true_bias_v_t":   true_bias_v_t,     # ← NEW
        "true_bias_v_x":   true_bias_v_x,     # ← NEW
        "true_bias_v_v":   true_bias_v_v,     # ← NEW
        "true_bias_v_ref": true_bias_v_ref,   # ← NEW
        "agent_bias_v_t":   agent_bias_v_t,   # ← NEW
        "agent_bias_v_x":   agent_bias_v_x,   # ← NEW
        "agent_bias_v_v":   agent_bias_v_v,   # ← NEW
        "agent_bias_v_ref": agent_bias_v_ref, # ← NEW
    }


def get_padded_limits(data_min, data_max, default_pad=0.1, pad_percent=0.10):
    if np.isnan(data_min) or np.isnan(data_max):
        return -default_pad, default_pad
    if np.isclose(data_min, data_max):
        span = abs(data_min) if data_min != 0 else default_pad
        return data_min - span * 0.1, data_max + span * 0.1
    span = data_max - data_min
    return data_min - span * pad_percent, data_max + pad_percent * span


def generate_3x3_grid_plots(summary_data, output_dir):
    print("[Diagnostic] Initializing 3x3 grid plot generation for x and v.")
    print(f"[Diagnostic] Target output directory: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    extracted_alphas = set()
    extracted_betas  = set()

    for trial in summary_data:
        a, b = extract_alpha_beta(trial)
        extracted_alphas.add(a)
        extracted_betas.add(b)

    alpha_vals = sorted(list(extracted_alphas))[:3]
    beta_vals  = sorted(list(extracted_betas))[:3]

    grid_trials = {(b, a): [] for b in beta_vals for a in alpha_vals}

    for trial in summary_data:
        alpha, beta = extract_alpha_beta(trial)
        a_closest = min(alpha_vals, key=lambda x: abs(x - alpha))
        b_closest = min(beta_vals,  key=lambda x: abs(x - beta))
        grid_trials[(b_closest, a_closest)].append(trial)

    domain        = np.linspace(-10, 10, 100)
    t_domain_bias = np.linspace(0,  10, 100)    # ← NEW
    v_domain_bias = np.linspace(-20, 20, 100)   # ← NEW

    evaluated_trials = {}
    for key, trials in grid_trials.items():
        evaluated_trials[key] = [
            (t, compute_trial_evaluations(t, domain, t_domain_bias, v_domain_bias))  # ← CHANGED: added t_domain_bias, v_domain_bias
            for t in trials
        ]

    cmap = plt.get_cmap("tab10")

    # ── existing render_grid_plot (UNCHANGED) ─────────────────────────────────
    def render_grid_plot(mode, plot_type, filename):
        fig, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=False, sharey=False)

        state_var    = "x" if mode == "x" else "v"
        agent_label  = r"\sigma_{" + mode + r", \text{agent}}"
        true_label   = r"\sigma_{" + mode + r", \text{true}}"

        if plot_type == "parity":
            fig.suptitle(
                rf"Parity Plot (${agent_label}$ vs. ${true_label}$ at"
                rf" ${state_var}=0$)",
                fontsize=16,
            )
        elif plot_type == "profile":
            fig.suptitle(
                rf"Noise Profile Plot (${agent_label}$ vs. ${true_label}$ over"
                rf" ${state_var} \in [-10, 10]$)",
                fontsize=16,
            )
        elif plot_type == "relative":
            fig.suptitle(
                rf"Relative Noise Error (${agent_label} / {true_label}$)",
                fontsize=16,
            )

        for i, beta in enumerate(reversed(beta_vals)):
            for j, alpha in enumerate(alpha_vals):
                ax     = axes[i, j]
                trials = evaluated_trials[(beta, alpha)]

                if plot_type == "parity":
                    x_vals, y_vals = [], []
                    for t_idx, (trial, eval_data) in enumerate(trials):
                        val_true  = eval_data[f"sigma_true_{mode}0"]
                        val_agent = eval_data[f"sigma_agent_{mode}0"]
                        if not np.isnan(val_true) and not np.isnan(val_agent):
                            x_vals.append(val_true)
                            y_vals.append(val_agent)
                            ax.scatter(val_true, val_agent, color=cmap(t_idx % 10), alpha=0.8, s=50)

                    if x_vals and y_vals:
                        x_min, x_max = get_padded_limits(np.min(x_vals), np.max(x_vals))
                        y_min, y_max = get_padded_limits(np.min(y_vals), np.max(y_vals))
                        diag_min, diag_max = min(x_min, y_min), max(x_max, y_max)
                        ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", alpha=0.7, label="y = x")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)

                    if j == 0:
                        ax.set_ylabel(rf"${agent_label}$ at ${state_var}=0$")
                    if i == 2:
                        ax.set_xlabel(rf"${true_label}$ at ${state_var}=0$")

                elif plot_type == "profile":
                    all_st, all_sa = [], []
                    for t_idx, (trial, eval_data) in enumerate(trials):
                        sig_true  = eval_data[f"sigma_true_{mode}"]
                        sig_agent = eval_data[f"sigma_agent_{mode}"]
                        if sig_agent is not None and len(sig_true) == len(sig_agent):
                            mask = ~np.isnan(sig_agent)
                            st_v, sa_v = sig_true[mask], sig_agent[mask]
                            if len(st_v) > 0:
                                ax.scatter(st_v, sa_v, color=cmap(t_idx % 10), s=15, alpha=0.7)
                                all_st.extend(st_v)
                                all_sa.extend(sa_v)

                    if all_st and all_sa:
                        x_min, x_max = get_padded_limits(np.min(all_st), np.max(all_st))
                        y_min, y_max = get_padded_limits(np.min(all_sa), np.max(all_sa))
                        diag_min, diag_max = min(x_min, y_min), max(x_max, y_max)
                        ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", alpha=0.7, label="y = x")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)

                    if j == 0:
                        ax.set_ylabel(rf"${agent_label}({state_var})$")
                    if i == 2:
                        ax.set_xlabel(rf"${true_label}({state_var})$")

                elif plot_type == "relative":
                    all_rel = []
                    for t_idx, (trial, eval_data) in enumerate(trials):
                        sa = eval_data[f"sigma_agent_{mode}"]
                        st = eval_data[f"sigma_true_{mode}"]
                        if sa is not None and len(sa) == len(domain):
                            rel_error = sa / np.maximum(st, 1e-12)
                            mask = ~np.isnan(rel_error)
                            if np.any(mask):
                                ax.plot(domain[mask], rel_error[mask], color=cmap(t_idx % 10), alpha=0.8)
                                all_rel.extend(rel_error[mask])

                    if all_rel:
                        y_min, y_max = get_padded_limits(np.min(all_rel), np.max(all_rel), default_pad=0.2)
                        ax.set_ylim(y_min, y_max)

                    ax.axhline(1.0, color="k", linestyle="--", alpha=0.8)
                    if j == 0:
                        ax.set_ylabel(rf"${agent_label} / {true_label}$")
                    if i == 2:
                        ax.set_xlabel(rf"State ${state_var}$")

                ax.set_title(rf"$\beta={beta}, \alpha={alpha}$")
                ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f" -> Generated: {save_path}")
    # ─────────────────────────────────────────────────────────────────────────

    # ── NEW FUNCTION ──────────────────────────────────────────────────────────
    def render_bias_grid_plot(mode, plot_type, profile_domain, filename):       # ← NEW
        """Generate a 3x3 grid plot for submitted bias_x / bias_v functions.

        mode:           'x' or 'v'
        plot_type:      'parity' | 'profile' | 'relative'
        profile_domain: 't' | 'x' | 'v'  (only used for plot_type=='profile')
        filename:       output PNG filename
        """
        fig, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=False, sharey=False)  # ← NEW

        agent_label = r"\hat{b}_{" + mode + r"}"                               # ← NEW
        true_label  = r"b_{" + mode + r"}"                                      # ← NEW

        if plot_type == "parity":                                                # ← NEW
            fig.suptitle(                                                        # ← NEW
                rf"Bias Parity (${agent_label}$ vs. ${true_label}$"             # ← NEW
                rf" at $x=0,\,v=0,\,t=0$)",                                    # ← NEW
                fontsize=16,                                                     # ← NEW
            )                                                                    # ← NEW
        elif plot_type == "profile":                                             # ← NEW
            domain_sym = {"t": "t", "x": "x", "v": "v"}[profile_domain]       # ← NEW
            fig.suptitle(                                                        # ← NEW
                rf"Bias Profile (${agent_label}$ vs. ${true_label}$"           # ← NEW
                rf" over ${domain_sym}$)",                                      # ← NEW
                fontsize=16,                                                     # ← NEW
            )                                                                    # ← NEW
        elif plot_type == "relative":                                            # ← NEW
            fig.suptitle(                                                        # ← NEW
                rf"Relative Bias Error (${agent_label} / {true_label}$"        # ← NEW
                rf" over $t$)",                                                  # ← NEW
                fontsize=16,                                                     # ← NEW
            )                                                                    # ← NEW

        for i, beta in enumerate(reversed(beta_vals)):                          # ← NEW
            for j, alpha in enumerate(alpha_vals):                              # ← NEW
                ax     = axes[i, j]                                             # ← NEW
                trials = evaluated_trials[(beta, alpha)]                        # ← NEW

                if plot_type == "parity":                                       # ← NEW
                    x_vals, y_vals = [], []                                     # ← NEW
                    for t_idx, (trial, eval_data) in enumerate(trials):        # ← NEW
                        val_true  = eval_data[f"true_bias_{mode}_ref"]         # ← NEW
                        val_agent = eval_data[f"agent_bias_{mode}_ref"]        # ← NEW
                        if not np.isnan(val_true) and not np.isnan(val_agent): # ← NEW
                            x_vals.append(val_true)                            # ← NEW
                            y_vals.append(val_agent)                           # ← NEW
                            ax.scatter(                                         # ← NEW
                                val_true, val_agent,                           # ← NEW
                                color=cmap(t_idx % 10), alpha=0.8, s=50,      # ← NEW
                            )                                                   # ← NEW

                    if x_vals and y_vals:                                       # ← NEW
                        x_min, x_max = get_padded_limits(np.min(x_vals), np.max(x_vals))  # ← NEW
                        y_min, y_max = get_padded_limits(np.min(y_vals), np.max(y_vals))  # ← NEW
                        diag_min = min(x_min, y_min)                           # ← NEW
                        diag_max = max(x_max, y_max)                           # ← NEW
                        ax.plot([diag_min, diag_max], [diag_min, diag_max],    # ← NEW
                                "k--", alpha=0.7, label="y = x")               # ← NEW
                        ax.set_xlim(x_min, x_max)                              # ← NEW
                        ax.set_ylim(y_min, y_max)                              # ← NEW

                    if j == 0:                                                  # ← NEW
                        ax.set_ylabel(rf"${agent_label}$ at $x=0,v=0,t=0$")   # ← NEW
                    if i == 2:                                                  # ← NEW
                        ax.set_xlabel(rf"${true_label}$ at $x=0,v=0,t=0$")    # ← NEW

                elif plot_type == "profile":                                    # ← NEW
                    true_key  = f"true_bias_{mode}_{profile_domain}"           # ← NEW
                    agent_key = f"agent_bias_{mode}_{profile_domain}"          # ← NEW
                    domain_sym = {"t": "t", "x": "x", "v": "v"}[profile_domain]  # ← NEW
                    all_true, all_agent = [], []                                # ← NEW

                    for t_idx, (trial, eval_data) in enumerate(trials):        # ← NEW
                        b_true  = eval_data[true_key]                          # ← NEW
                        b_agent = eval_data[agent_key]                         # ← NEW
                        if b_agent is not None and len(b_true) == len(b_agent):  # ← NEW
                            mask = ~np.isnan(b_agent)                          # ← NEW
                            bt_v, ba_v = b_true[mask], b_agent[mask]          # ← NEW
                            if len(bt_v) > 0:                                  # ← NEW
                                ax.scatter(bt_v, ba_v,                         # ← NEW
                                           color=cmap(t_idx % 10), s=15, alpha=0.7)  # ← NEW
                                all_true.extend(bt_v)                          # ← NEW
                                all_agent.extend(ba_v)                         # ← NEW

                    if all_true and all_agent:                                  # ← NEW
                        x_min, x_max = get_padded_limits(np.min(all_true), np.max(all_true))   # ← NEW
                        y_min, y_max = get_padded_limits(np.min(all_agent), np.max(all_agent)) # ← NEW
                        diag_min = min(x_min, y_min)                           # ← NEW
                        diag_max = max(x_max, y_max)                           # ← NEW
                        ax.plot([diag_min, diag_max], [diag_min, diag_max],    # ← NEW
                                "k--", alpha=0.7, label="y = x")               # ← NEW
                        ax.set_xlim(x_min, x_max)                              # ← NEW
                        ax.set_ylim(y_min, y_max)                              # ← NEW

                    if j == 0:                                                  # ← NEW
                        ax.set_ylabel(rf"${agent_label}({domain_sym})$")       # ← NEW
                    if i == 2:                                                  # ← NEW
                        ax.set_xlabel(rf"${true_label}({domain_sym})$")        # ← NEW

                elif plot_type == "relative":                                   # ← NEW
                    all_rel = []                                                # ← NEW
                    for t_idx, (trial, eval_data) in enumerate(trials):        # ← NEW
                        b_agent = eval_data[f"agent_bias_{mode}_t"]            # ← NEW
                        b_true  = eval_data[f"true_bias_{mode}_t"]             # ← NEW
                        if b_agent is not None and len(b_agent) == len(b_true):  # ← NEW
                            # guard against near-zero denominator              # ← NEW
                            denom = np.where(                                   # ← NEW
                                np.abs(b_true) > 1e-12,                        # ← NEW
                                b_true,                                         # ← NEW
                                np.sign(b_true + 1e-15) * 1e-12,               # ← NEW
                            )                                                   # ← NEW
                            rel_error = b_agent / denom                        # ← NEW
                            mask = ~np.isnan(rel_error)                        # ← NEW
                            if np.any(mask):                                    # ← NEW
                                ax.plot(t_domain_bias[mask], rel_error[mask],  # ← NEW
                                        color=cmap(t_idx % 10), alpha=0.8)     # ← NEW
                                all_rel.extend(rel_error[mask])                # ← NEW

                    if all_rel:                                                 # ← NEW
                        y_min, y_max = get_padded_limits(                      # ← NEW
                            np.min(all_rel), np.max(all_rel), default_pad=0.2  # ← NEW
                        )                                                       # ← NEW
                        ax.set_ylim(y_min, y_max)                              # ← NEW

                    ax.axhline(1.0, color="k", linestyle="--", alpha=0.8)      # ← NEW
                    if j == 0:                                                  # ← NEW
                        ax.set_ylabel(rf"${agent_label} / {true_label}$")      # ← NEW
                    if i == 2:                                                  # ← NEW
                        ax.set_xlabel(r"Time $t$")                             # ← NEW

                ax.set_title(rf"$\beta={beta},\,\alpha={alpha}$")              # ← NEW
                ax.grid(True, linestyle=":", alpha=0.6)                        # ← NEW

        plt.tight_layout()                                                      # ← NEW
        save_path = os.path.join(output_dir, filename)                         # ← NEW
        fig.savefig(save_path, dpi=300)                                        # ← NEW
        plt.close(fig)                                                          # ← NEW
        print(f" -> Generated: {save_path}")                                   # ← NEW
    # ── END NEW FUNCTION ──────────────────────────────────────────────────────

    # Generate all 6 sigma images (UNCHANGED)
    render_grid_plot("x", "parity",   "image1_sigma_x0_parity.png")
    render_grid_plot("x", "profile",  "image2_sigma_x_profile.png")
    render_grid_plot("x", "relative", "image3_sigma_x_relative_errors.png")
    render_grid_plot("v", "parity",   "image4_sigma_v0_parity.png")
    render_grid_plot("v", "profile",  "image5_sigma_v_profile.png")
    render_grid_plot("v", "relative", "image6_sigma_v_relative_errors.png")

    # ── NEW: Generate all 10 bias images ─────────────────────────────────────
    render_bias_grid_plot("x", "parity",   None, "image7_bias_x_parity.png")           # ← NEW
    render_bias_grid_plot("x", "profile",  "t",  "image8_bias_x_profile_t.png")        # ← NEW
    render_bias_grid_plot("x", "profile",  "x",  "image9_bias_x_profile_x.png")        # ← NEW
    render_bias_grid_plot("x", "profile",  "v",  "image10_bias_x_profile_v.png")       # ← NEW
    render_bias_grid_plot("x", "relative", None, "image11_bias_x_relative_t.png")      # ← NEW
    render_bias_grid_plot("v", "parity",   None, "image12_bias_v_parity.png")           # ← NEW
    render_bias_grid_plot("v", "profile",  "t",  "image13_bias_v_profile_t.png")        # ← NEW
    render_bias_grid_plot("v", "profile",  "x",  "image14_bias_v_profile_x.png")        # ← NEW
    render_bias_grid_plot("v", "profile",  "v",  "image15_bias_v_profile_v.png")        # ← NEW
    render_bias_grid_plot("v", "relative", None, "image16_bias_v_relative_t.png")      # ← NEW
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate 6 sigma + 10 bias 3x3 grid plots dynamically from"
            " JSON summary."
        )
    )
    parser.add_argument("--summary_path", type=str, required=True,
                        help="Path to compiled division summary JSON.")
    parser.add_argument("--output_dir",   type=str, default="./plots_3x3",
                        help="Output directory for generated plot images.")
    args = parser.parse_args()

    summary_data = load_division_summary(args.summary_path)
    generate_3x3_grid_plots(summary_data, args.output_dir)
