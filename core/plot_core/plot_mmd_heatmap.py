import os
import json
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def compute_1d_mmd(P, Q, gamma=None):
    """
    Computes Maximum Mean Discrepancy (MMD) between 1D sample arrays P and Q using an RBF kernel.
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 1)
    Q = np.asarray(Q, dtype=np.float64).reshape(-1, 1)

    dist_PP = (P - P.T) ** 2
    dist_QQ = (Q - Q.T) ** 2
    dist_PQ = (P - Q.T) ** 2

    if gamma is None:
        all_dists = np.concatenate([dist_PP.ravel(), dist_QQ.ravel(), dist_PQ.ravel()])
        med = np.median(all_dists)
        gamma = 1.0 / (2.0 * (med + 1e-8)) if med > 0 else 1.0

    K_PP = np.exp(-gamma * dist_PP)
    K_QQ = np.exp(-gamma * dist_QQ)
    K_PQ = np.exp(-gamma * dist_PQ)

    mmd_sq = np.mean(K_PP) + np.mean(K_QQ) - 2.0 * np.mean(K_PQ)
    return float(np.sqrt(max(0.0, mmd_sq)))


def get_true_epsilon_samples(config, noise_config, x, v, t, var_name="x", num_samples=100, rng=None):
    """
    Generates sample realizations of true_epsilon_x or true_epsilon_v at state (x, v, t).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    func_attr = f"true_epsilon_{var_name}"
    if hasattr(config, func_attr) and callable(getattr(config, func_attr)):
        fn = getattr(config, func_attr)
        return np.array([fn(x, v, t) for _ in range(num_samples)])
    elif hasattr(config, "true_epsilon") and callable(config.true_epsilon):
        return np.array([config.true_epsilon(x, v, t) for _ in range(num_samples)])
    else:
        c_noise = noise_config.get("meas_const_noise", 0.1) if noise_config else 0.1
        l_noise = noise_config.get("meas_lin_noise", 0.1) if noise_config else 0.1
        state_val = x if var_name == "x" else v
        std = c_noise + l_noise * abs(state_val)
        return rng.normal(0, std, size=num_samples)


def get_agent_epsilon_samples(agent_mod, x, v, t, var_name="x", num_samples=100):
    """
    Generates sample realizations of agent_epsilon_x or agent_epsilon_v at state (x, v, t).
    """
    func_attr = f"epsilon_{var_name}"
    if agent_mod is None or not hasattr(agent_mod, func_attr) or not callable(getattr(agent_mod, func_attr)):
        return np.zeros(num_samples)

    fn = getattr(agent_mod, func_attr)
    samples = []
    for _ in range(num_samples):
        try:
            val = fn(x, v, t)
            samples.append(float(val))
        except Exception:
            samples.append(0.0)
    return np.array(samples)


def precompute_true_mmd_grid(config, noise_config, x_grid, v_grid, t_frames, var_name="x", num_samples=100):
    """
    Precomputes true_epsilon sample array P once for all trials in a division.
    Shape: (num_frames, Nv, Nx, num_samples)
    """
    num_frames = len(t_frames)
    Nv = len(v_grid)
    Nx = len(x_grid)

    P_samples = np.zeros((num_frames, Nv, Nx, num_samples))
    rng = np.random.default_rng(42)

    for k, t_val in enumerate(t_frames):
        for j, v_val in enumerate(v_grid):
            for i, x_val in enumerate(x_grid):
                P_samples[k, j, i, :] = get_true_epsilon_samples(
                    config, noise_config, x_val, v_val, t_val, var_name=var_name, num_samples=num_samples, rng=rng
                )
    return P_samples


def create_mmd_heatmap_video(A, x_grid, v_grid, t_frames, var_name, output_video_path, target_duration_sec=10.0):
    """
    Renders 2D heatmap animation across time frames t for variable var_name ('x' or 'v').
    Enforces a constant total video duration set by target_duration_sec (10.0 seconds).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    vmin = 0.0
    vmax = float(np.max(A)) if np.max(A) > 0 else 1.0
    extent = [x_grid[0], x_grid[-1], v_grid[0], v_grid[-1]]

    im = ax.imshow(
        A[0],
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Local MMD Discrepancy A_{var_name}(x, v, t)")

    ax.set_xlabel("Position x")
    ax.set_ylabel("Velocity v")
    title = ax.set_title(f"Local MMD Heatmap [\\epsilon_{var_name}] (t = {t_frames[0]:.2f}s)")

    num_frames = len(t_frames)
    fps = num_frames / target_duration_sec
    interval_ms = (target_duration_sec / num_frames) * 1000.0

    def update(frame_idx):
        im.set_data(A[frame_idx])
        title.set_text(f"Local MMD Heatmap [\\epsilon_{var_name}] (t = {t_frames[frame_idx]:.2f}s)")
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval_ms, blit=True
    )

    try:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output_video_path, writer=writer)
    except Exception:
        gif_path = os.path.splitext(output_video_path)[0] + ".gif"
        try:
            anim.save(gif_path, writer="pillow", fps=fps)
        except Exception as e:
            print(f"[Warning] Could not save MMD heatmap animation for {var_name}: {e}")

    plt.close(fig)


def process_division_mmd(div_dir, args, config, noise_config):
    """
    Computes shared P matrix for x and v across division, evaluates trial MMDs for both,
    updates summary.json with single scalar MMD values, and renders videos per trial if requested.
    """
    # Guard clause: Exit immediately if MMD heatmap flag is not passed
    if not getattr(args, "plot_mmd_heatmap", False):
        return

    if os.path.exists(os.path.join(div_dir, "trials")):
        trials_parent = os.path.join(div_dir, "trials")
    else:
        trials_parent = div_dir

    trial_dirs = [
        os.path.join(trials_parent, d)
        for d in sorted(os.listdir(trials_parent))
        if d.startswith("trial_") and os.path.isdir(os.path.join(trials_parent, d))
    ]

    if not trial_dirs:
        return

    # 1. State space grid & time frames setup
    if hasattr(config, "ENV_SCHEMA") and "x" in config.ENV_SCHEMA and "v" in config.ENV_SCHEMA:
        x_min, x_max = config.ENV_SCHEMA["x"]["range"]
        v_min, v_max = config.ENV_SCHEMA["v"]["range"]
    else:
        x_min, x_max = -10.0, 10.0
        v_min, v_max = -20.0, 20.0

    Nx, Nv = 20, 20
    x_grid = np.linspace(x_min, x_max, Nx)
    v_grid = np.linspace(v_min, v_max, Nv)
    t_frames = np.linspace(0.0, 10.0, 15)

    # 2. Precompute P (true_epsilon_x and true_epsilon_v) once for the whole division
    P_samples_x = precompute_true_mmd_grid(config, noise_config, x_grid, v_grid, t_frames, var_name="x", num_samples=200)
    P_samples_v = precompute_true_mmd_grid(config, noise_config, x_grid, v_grid, t_frames, var_name="v", num_samples=200)

    do_plot = getattr(args, "plot_mmd_heatmap", False)

    # 3. Process each trial using shared P matrices
    for t_dir in trial_dirs:
        submission_script_path = os.path.join(t_dir, "submission_model.py")
        agent_mod = None

        if os.path.exists(submission_script_path):
            try:
                spec = importlib.util.spec_from_file_location("agent_submission_mmd", submission_script_path)
                agent_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agent_mod)
            except Exception:
                agent_mod = None

        A_x = np.zeros((len(t_frames), Nv, Nx))
        A_v = np.zeros((len(t_frames), Nv, Nx))

        for k, t_val in enumerate(t_frames):
            for j, v_val in enumerate(v_grid):
                for i, x_val in enumerate(x_grid):
                    # Evaluate x
                    P_ij_x = P_samples_x[k, j, i, :]
                    Q_ij_x = get_agent_epsilon_samples(agent_mod, x_val, v_val, t_val, var_name="x", num_samples=500)
                    A_x[k, j, i] = compute_1d_mmd(P_ij_x, Q_ij_x)

                    # Evaluate v
                    P_ij_v = P_samples_v[k, j, i, :]
                    Q_ij_v = get_agent_epsilon_samples(agent_mod, x_val, v_val, t_val, var_name="v", num_samples=500)
                    A_v[k, j, i] = compute_1d_mmd(P_ij_v, Q_ij_v)

        # Update summary.json with ONLY the single mean MMD scalar values
        summary_path = os.path.join(t_dir, "summary.json")
        summary_data = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as sf:
                    summary_data = json.load(sf)
            except Exception:
                summary_data = {}

        summary_data["mmd_x"] = float(np.mean(A_x))
        summary_data["mmd_v"] = float(np.mean(A_v))

        with open(summary_path, "w") as sf:
            json.dump(summary_data, sf, indent=4)

        # Plot videos if requested
        if do_plot:
            video_path_x = os.path.join(t_dir, "mmd_heatmap_x.mp4")
            video_path_v = os.path.join(t_dir, "mmd_heatmap_v.mp4")
            create_mmd_heatmap_video(A_x, x_grid, v_grid, t_frames, "x", video_path_x, target_duration_sec=5.0)
            create_mmd_heatmap_video(A_v, x_grid, v_grid, t_frames, "v", video_path_v, target_duration_sec=5.0)
