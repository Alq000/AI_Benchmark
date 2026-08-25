import json
import numpy as np

def load_trajectories(experiments_file_path):
    """
    Parses history JSON files directly into structured, clean NumPy arrays[cite: 18].
    """
    with open(experiments_file_path, "r") as f:
        data = json.load(f)

    trajectories = []

    for run_key, run_data in data.items():
        for traj in run_data.get("trajectories", []):
            ic = traj["initial_conditions"]
            x0 = ic["x0"]
            v0 = ic["v0"]

            measurements = traj["measurements"]
            if not measurements:
                continue

            t_arr = np.array([m["t"] for m in measurements])
            x_obs = np.array([m["x_measured"] for m in measurements])
            v_obs = np.array([m["v_measured"] for m in measurements])
            
            trajectories.append({
                "trajectory_id": f"{run_key}_{traj.get('id', 'unknown')}",
                "x0": x0,
                "v0": v0,
                "t": t_arr,
                "x": x_obs,
                "v": v_obs
            })

    return trajectories
