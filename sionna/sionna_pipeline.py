"""
ResQNet - Person 2 (Sionna RT & Wireless Propagation Module)
Full-featured production pipeline supporting:
 1. Visualization with UAV & User overlays
 2. Robust scene loading (.obj / .ply / built-in fallback)
 3. Dead-zone tracking & unbiased mean path gain (handling -200 dB cells)
 4. Engine 1: Grid-Sweep Dataset Generator (15 Environments x 507 grid points)
 5. Engine 2: Temporal Earthquake Trajectory Generator (Clustered survivors + UAV repositioning over time)
 6. Dual Export: Structured JSON + Flattened CSV + Stats Summary + Trajectory plots
 7. Complete JSON handoff interface for Person 3 (ML) and Person 4 (ns-3)
"""

import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import sionna
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray,
    PathSolver, RadioMapSolver
)

def safe_load_scene(scene_name):
    """
    Robustly loads a scene from a custom .obj/.ply file or built-in preset.
    Handles missing files, unassigned materials, and syntax errors gracefully.
    """
    if scene_name is None:
        return load_scene(sionna.rt.scene.simple_street_canyon)
        
    if isinstance(scene_name, str):
        if os.path.exists(scene_name):
            try:
                scene = load_scene(scene_name)
                for name, obj in scene.objects.items():
                    if obj.radio_material is None:
                        obj.radio_material = "itu_concrete"
                return scene
            except Exception as e:
                print(f"[ERROR] Failed to load custom mesh '{scene_name}': {e}. Falling back to default.")
                return load_scene(sionna.rt.scene.simple_street_canyon)
        else:
            return load_scene(sionna.rt.scene.simple_street_canyon)
            
    try:
        return load_scene(scene_name)
    except Exception as e:
        return load_scene(sionna.rt.scene.simple_street_canyon)


def run_uav_propagation_sim(
    uav_positions,
    user_positions,
    scene_name=None,
    freq=2.1e9,
    tx_power_dbm=30.0,
    samples_per_tx=10**5,
    save_plot=False,
    plot_filename="coverage_heatmap.png"
):
    """
    Runs single-scenario Sionna RT simulation.
    """
    # 1. Load Scene
    scene = safe_load_scene(scene_name)
    scene.frequency = freq
    
    # 2. Antenna Array Configuration
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    
    # 3. Add Transmitters (UAVs)
    tx_names = []
    for idx, pos in enumerate(uav_positions):
        name = f"uav_{idx+1}"
        tx = Transmitter(name=name, position=pos)
        scene.add(tx)
        tx_names.append(name)
        
    # 4. Add Receivers (Ground Users)
    rx_names = []
    for idx, pos in enumerate(user_positions):
        name = f"user_{idx+1}"
        rx = Receiver(name=name, position=pos)
        scene.add(rx)
        rx_names.append(name)

    # 5. Ray-Tracing Path Solver
    p_solver = PathSolver()
    paths = p_solver(scene)
    
    a_real, a_imag = paths.a
    a_complex = a_real.numpy() + 1j * a_imag.numpy()
    taus = paths.tau.numpy()
    
    # 6. Extract Per-User Metrics for ns-3
    user_results = []
    tx_power_watts = 10 ** ((tx_power_dbm - 30) / 10)
    
    for u_idx, u_pos in enumerate(user_positions):
        user_a = a_complex[u_idx, 0, :, 0, :]
        user_tau = taus[u_idx, :, :]
        
        path_gains_linear = np.abs(user_a) ** 2
        total_path_gain = np.sum(path_gains_linear)
        rx_power_linear = total_path_gain * tx_power_watts
        rx_power_dbm = 10 * np.log10(rx_power_linear * 1000 + 1e-20)
        
        valid_delays = user_tau[user_tau > 0]
        min_delay = float(np.min(valid_delays)) if len(valid_delays) > 0 else None
        
        user_results.append({
            "user_id": f"user_{u_idx+1}",
            "position": u_pos,
            "received_power_dbm": round(float(rx_power_dbm), 2),
            "path_gain_linear": float(total_path_gain),
            "shortest_delay_sec": min_delay,
            "num_paths": int(np.count_nonzero(path_gains_linear))
        })
        
    # 7. Radio Map Generation & Dead Zone (-200 dB) Handling
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene=scene, max_depth=3, cell_size=[3, 3], samples_per_tx=samples_per_tx)
    path_gain_grid = rm.path_gain.numpy()
    
    # Identify Dead Zones
    no_coverage_mask = path_gain_grid < 1e-18
    no_coverage_count = int(np.sum(no_coverage_mask))
    total_cells = path_gain_grid.size
    no_coverage_pct = round((no_coverage_count / total_cells) * 100.0, 2)
    
    path_gain_db = 10 * np.log10(path_gain_grid + 1e-20)
    
    valid_path_gains_db = path_gain_db[~no_coverage_mask]
    if len(valid_path_gains_db) > 0:
        unbiased_mean_gain_db = float(np.mean(valid_path_gains_db))
        max_gain_db = float(np.max(valid_path_gains_db))
        min_valid_gain_db = float(np.min(valid_path_gains_db))
    else:
        unbiased_mean_gain_db = -200.0
        max_gain_db = -200.0
        min_valid_gain_db = -200.0

    coverage_threshold_db = -100.0
    covered_cells = np.sum(path_gain_db > coverage_threshold_db)
    coverage_percentage = round((covered_cells / total_cells) * 100.0, 2)
    
    output_data = {
        "simulation_config": {
            "num_uavs": len(uav_positions),
            "uav_positions": uav_positions,
            "carrier_frequency_hz": freq,
            "tx_power_dbm": tx_power_dbm
        },
        "ml_summary_metrics": {
            "coverage_percentage": coverage_percentage,
            "mean_path_gain_db": round(unbiased_mean_gain_db, 2),
            "min_valid_path_gain_db": round(min_valid_gain_db, 2),
            "max_path_gain_db": round(max_gain_db, 2),
            "no_coverage_cell_count": no_coverage_count,
            "no_coverage_percentage": no_coverage_pct
        },
        "user_metrics_for_ns3": user_results
    }
    
    if save_plot:
        plot_coverage_heatmap(rm, path_gain_db, uav_positions, user_positions, save_path=plot_filename)
        
    return output_data, path_gain_db


def plot_coverage_heatmap(rm, path_gain_db, uav_positions, user_positions, save_path="coverage_heatmap.png"):
    """
    Plots the radio map coverage grid as a 2D heatmap with UAV (Tx) and User (Rx) overlays.
    """
    center = rm.center.numpy() if hasattr(rm.center, 'numpy') else rm.center
    size = rm.size.numpy() if hasattr(rm.size, 'numpy') else rm.size
    
    xmin = center[0] - size[0] / 2.0
    xmax = center[0] + size[0] / 2.0
    ymin = center[1] - size[1] / 2.0
    ymax = center[1] + size[1] / 2.0
    extent = [xmin, xmax, ymin, ymax]
    
    plt.figure(figsize=(10, 8))
    grid_to_plot = path_gain_db[0] if path_gain_db.ndim == 3 else path_gain_db
    
    im = plt.imshow(grid_to_plot, cmap="viridis", origin="lower", extent=extent, vmin=-140, vmax=-60)
    cbar = plt.colorbar(im)
    cbar.set_label("Path Gain (dB)")
    
    # Overlay UAV Transmitters (Red Triangles)
    uav_x = [pos[0] for pos in uav_positions]
    uav_y = [pos[1] for pos in uav_positions]
    plt.scatter(uav_x, uav_y, c="red", marker="^", s=140, edgecolor="black", linewidth=1.5, label="UAV (Tx)")
    for i, pos in enumerate(uav_positions):
        plt.annotate(f"UAV {i+1}", (pos[0], pos[1]), textcoords="offset points", xytext=(0, 10), ha='center', color="white", weight="bold", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.7))
        
    # Overlay Ground Users (Cyan Circles)
    user_x = [pos[0] for pos in user_positions]
    user_y = [pos[1] for pos in user_positions]
    plt.scatter(user_x, user_y, c="cyan", marker="o", s=90, edgecolor="black", linewidth=1.5, label="User (Rx)")
    for i, pos in enumerate(user_positions):
        plt.annotate(f"User {i+1}", (pos[0], pos[1]), textcoords="offset points", xytext=(0, -15), ha='center', color="black", weight="bold", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="cyan", alpha=0.8))

    plt.title("ResQNet Sionna RT Coverage Heatmap\n(UAV & Ground User Overlay)", fontsize=13, fontweight="bold")
    plt.xlabel("X Position (meters)", fontsize=11)
    plt.ylabel("Y Position (meters)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved coverage heatmap to '{save_path}'")


# =====================================================================
# ENGINE 1: GRID-SWEEP DATASET GENERATOR
# =====================================================================

def generate_environments(num_environments=15, user_count_range=(8, 15), x_bounds=(-50.0, 70.0), y_bounds=(-50.0, 70.0), seed=42):
    """
    Generates N fixed disaster environments with reproducible ground user layouts.
    """
    rng = np.random.default_rng(seed)
    environments = []
    for idx in range(num_environments):
        num_users = int(rng.integers(user_count_range[0], user_count_range[1] + 1))
        users = []
        for u in range(num_users):
            x = round(float(rng.uniform(x_bounds[0], x_bounds[1])), 2)
            y = round(float(rng.uniform(y_bounds[0], y_bounds[1])), 2)
            users.append([x, y, 1.5])
            
        environments.append({
            "env_id": f"env_{idx+1}",
            "seed": seed + idx,
            "num_users": num_users,
            "user_positions": users
        })
    return environments


def run_grid_sweep_dataset(
    num_environments=15,
    x_grid=np.arange(-50.0, 70.0 + 1e-5, 10.0).tolist(),
    y_grid=np.arange(-50.0, 70.0 + 1e-5, 10.0).tolist(),
    z_grid=[20.0, 30.0, 40.0],
    scene_name=None,
    seed=42,
    output_json_file="grid_sweep_dataset.json",
    output_csv_file="grid_sweep_dataset.csv"
):
    """
    Sweeps UAV positions across a dense 3D grid for multiple fixed user environments.
    """
    environments = generate_environments(num_environments=num_environments, seed=seed)
    
    grid_points = []
    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                grid_points.append([round(float(x), 2), round(float(y), 2), round(float(z), 2)])
                
    points_per_env = len(grid_points)
    total_target_runs = num_environments * points_per_env
    
    print(f"\n==================================================")
    print(f"   STARTING ENGINE 1: GRID-SWEEP DATASET GENERATION")
    print(f"   Environments: {num_environments} | Grid Points / Env: {points_per_env}")
    print(f"   Total Target Runs: {total_target_runs}")
    print(f"==================================================")
    
    dataset_environments = []
    csv_rows = []
    start_total_time = time.time()
    total_completed_runs = 0
    
    for env_idx, env in enumerate(environments):
        env_id = env["env_id"]
        users = env["user_positions"]
        print(f"\n---> Env {env_idx+1}/{num_environments} ({env_id}): {len(users)} Users | Sweeping {points_per_env} grid points...")
        
        env_runs = []
        env_start_time = time.time()
        
        for g_idx, pt in enumerate(grid_points):
            uav_pos = [pt]
            t0 = time.time()
            try:
                sim_data, _ = run_uav_propagation_sim(
                    uav_positions=uav_pos,
                    user_positions=users,
                    scene_name=scene_name,
                    samples_per_tx=10**5,
                    save_plot=False
                )
                elapsed = time.time() - t0
                total_completed_runs += 1
                
                run_record = {
                    "run_id": g_idx + 1,
                    "uav_x": pt[0],
                    "uav_y": pt[1],
                    "uav_z": pt[2],
                    "coverage_percentage": sim_data["ml_summary_metrics"]["coverage_percentage"],
                    "mean_path_gain_db": sim_data["ml_summary_metrics"]["mean_path_gain_db"],
                    "no_coverage_percentage": sim_data["ml_summary_metrics"]["no_coverage_percentage"],
                    "user_results": sim_data["user_metrics_for_ns3"]
                }
                env_runs.append(run_record)
                
                for u in sim_data["user_metrics_for_ns3"]:
                    csv_rows.append({
                        "env_id": env_id,
                        "run_id": g_idx + 1,
                        "uav_x": pt[0],
                        "uav_y": pt[1],
                        "uav_z": pt[2],
                        "num_users": len(users),
                        "coverage_percentage": sim_data["ml_summary_metrics"]["coverage_percentage"],
                        "mean_path_gain_db": sim_data["ml_summary_metrics"]["mean_path_gain_db"],
                        "no_coverage_percentage": sim_data["ml_summary_metrics"]["no_coverage_percentage"],
                        "user_id": u["user_id"],
                        "user_x": u["position"][0],
                        "user_y": u["position"][1],
                        "user_z": u["position"][2],
                        "received_power_dbm": u["received_power_dbm"],
                        "path_gain_linear": u["path_gain_linear"],
                        "shortest_delay_sec": u["shortest_delay_sec"] if u["shortest_delay_sec"] is not None else "",
                        "num_paths": u["num_paths"]
                    })
                    
            except Exception as e:
                print(f"[ERROR] Env {env_id} Grid {g_idx+1} failed: {e}. Skipping.")

            if (g_idx + 1) % 100 == 0 or (g_idx + 1) == points_per_env:
                env_elapsed = time.time() - env_start_time
                print(f"     [{g_idx+1}/{points_per_env} grid points done in {env_elapsed:.1f}s]")
                
        dataset_environments.append({
            "env_id": env_id,
            "num_users": len(users),
            "user_positions": users,
            "grid_runs_count": len(env_runs),
            "runs": env_runs
        })

    total_elapsed = time.time() - start_total_time

    dataset_output = {
        "schema_version": "1.0",
        "notes": f"Grid-sweep dataset across {len(dataset_environments)} environments ({total_completed_runs} total runs). Single UAV swept over 3D grid.",
        "num_environments": len(dataset_environments),
        "total_runs": total_completed_runs,
        "environments": dataset_environments
    }
    with open(output_json_file, "w") as f:
        json.dump(dataset_output, f, indent=4)
        
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(output_csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
            
    print(f"[SUCCESS] Saved Grid-Sweep Dataset ({total_completed_runs} runs) to '{output_json_file}' and '{output_csv_file}' in {total_elapsed:.2f}s.")
    return dataset_output


# =====================================================================
# ENGINE 2: TEMPORAL EARTHQUAKE TRAJECTORY GENERATOR
# =====================================================================

def generate_earthquake_users(
    num_clusters_range=(2, 4),
    users_per_cluster_range=(3, 6),
    std_dev=6.0,
    x_bounds=(-40.0, 60.0),
    y_bounds=(-40.0, 60.0),
    rng=None
):
    """
    Requirement 1: Generates earthquake survivor groups clustered around 2-4 collapsed building centers.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    num_clusters = int(rng.integers(num_clusters_range[0], num_clusters_range[1] + 1))
    cluster_centers = []
    for _ in range(num_clusters):
        cx = round(float(rng.uniform(x_bounds[0], x_bounds[1])), 2)
        cy = round(float(rng.uniform(y_bounds[0], y_bounds[1])), 2)
        cluster_centers.append([cx, cy])
        
    users = []
    user_idx = 1
    for c_id, center in enumerate(cluster_centers):
        num_cluster_users = int(rng.integers(users_per_cluster_range[0], users_per_cluster_range[1] + 1))
        for _ in range(num_cluster_users):
            ux = round(float(rng.normal(center[0], std_dev)), 2)
            uy = round(float(rng.normal(center[1], std_dev)), 2)
            # Clip to scene bounds
            ux = np.clip(ux, x_bounds[0] - 10, x_bounds[1] + 10)
            uy = np.clip(uy, y_bounds[0] - 10, y_bounds[1] + 10)
            
            users.append({
                "user_id": f"user_{user_idx}",
                "position": [round(float(ux), 2), round(float(uy), 2), 1.5],
                "cluster_id": c_id + 1
            })
            user_idx += 1
            
    return cluster_centers, users


def generate_uav_trajectory(start_pos, target_centroid, trajectory_type="toward_cluster", timesteps=8, max_step=15.0, rng=None):
    """
    Requirement 3: Generates a sequence of UAV positions over time.
    - 'greedy_exploration': Bounded random walk (max step 15m).
    - 'toward_cluster': UAV moves step-by-step toward survivor cluster centroid.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    trajectory = [start_pos]
    curr_pos = np.array(start_pos, dtype=float)
    target = np.array([target_centroid[0], target_centroid[1], start_pos[2]], dtype=float)
    
    for t in range(1, timesteps):
        if trajectory_type == "toward_cluster":
            # Direction vector toward centroid
            direction = target - curr_pos
            dist = np.linalg.norm(direction[:2])
            if dist > 1.0:
                step_dist = min(max_step, dist * 0.4)
                unit_dir = direction / np.linalg.norm(direction)
                next_pos = curr_pos + unit_dir * step_dist
            else:
                # Small hover jitter around centroid
                jitter = rng.uniform(-2.0, 2.0, size=3)
                jitter[2] = 0.0
                next_pos = curr_pos + jitter
        else:  # greedy_exploration (bounded random walk)
            step_xy = rng.uniform(-max_step, max_step, size=2)
            step_z = rng.uniform(-3.0, 3.0)
            next_pos = curr_pos + np.array([step_xy[0], step_xy[1], step_z])
            next_pos[2] = np.clip(next_pos[2], 15.0, 50.0)  # Bound altitude
            
        next_pos[0] = round(float(next_pos[0]), 2)
        next_pos[1] = round(float(next_pos[1]), 2)
        next_pos[2] = round(float(next_pos[2]), 2)
        trajectory.append(next_pos.tolist())
        curr_pos = next_pos
        
    return trajectory


def run_earthquake_temporal_dataset(
    num_disaster_states=25,
    timesteps_per_state=8,
    scene_name=None,
    seed=100,
    output_json_file="earthquake_trajectory_dataset.json",
    output_csv_file="earthquake_trajectory_dataset.csv",
    output_stats_file="earthquake_stats_summary.json"
):
    """
    Requirements 4, 5, 6, 8: Generates temporal earthquake dataset with clustered survivor states
    and UAV repositioning trajectories over time.
    """
    print(f"\n==================================================")
    print(f"   STARTING ENGINE 2: TEMPORAL EARTHQUAKE DATASET")
    print(f"   Disaster States: {num_disaster_states} | Timesteps / State: {timesteps_per_state}")
    print(f"   Total Timestep Runs: {num_disaster_states * timesteps_per_state}")
    print(f"==================================================")
    
    rng = np.random.default_rng(seed)
    disaster_states = []
    csv_rows = []
    
    t0_coverages = []
    t_final_coverages = []
    
    start_time = time.time()
    total_runs = 0
    
    for s_idx in range(num_disaster_states):
        state_seed = seed + s_idx * 10
        state_rng = np.random.default_rng(state_seed)
        
        # 1. Generate earthquake survivor clusters (fixed environment for this state)
        centers, users_info = generate_earthquake_users(rng=state_rng)
        user_positions_3d = [u["position"] for u in users_info]
        
        # Compute centroid of survivor clusters
        centroid = np.mean([u["position"][:2] for u in users_info], axis=0).tolist()
        
        # Alternate trajectory types (50% toward_cluster, 50% greedy_exploration)
        traj_type = "toward_cluster" if (s_idx % 2 == 0) else "greedy_exploration"
        
        # Random initial UAV position
        start_uav = [
            round(float(state_rng.uniform(-40, 60)), 2),
            round(float(state_rng.uniform(-40, 60)), 2),
            round(float(state_rng.uniform(20, 35)), 2)
        ]
        
        trajectory = generate_uav_trajectory(
            start_pos=start_uav,
            target_centroid=centroid,
            trajectory_type=traj_type,
            timesteps=timesteps_per_state,
            rng=state_rng
        )
        
        print(f"\n---> State {s_idx+1}/{num_disaster_states} (ID: {s_idx+1}) | Type: {traj_type} | Users: {len(users_info)} | Timesteps: {timesteps_per_state}")
        
        timesteps_data = []
        
        for t_idx, uav_pos in enumerate(trajectory):
            t_sim_0 = time.time()
            try:
                sim_data, _ = run_uav_propagation_sim(
                    uav_positions=[uav_pos],
                    user_positions=user_positions_3d,
                    scene_name=scene_name,
                    samples_per_tx=10**5,
                    save_plot=False
                )
                dt_sim = time.time() - t_sim_0
                total_runs += 1
                
                cov_pct = sim_data["ml_summary_metrics"]["coverage_percentage"]
                if t_idx == 0:
                    t0_coverages.append(cov_pct)
                if t_idx == timesteps_per_state - 1:
                    t_final_coverages.append(cov_pct)
                    
                timestep_record = {
                    "t": t_idx,
                    "uav_position": uav_pos,
                    "coverage_percentage": cov_pct,
                    "mean_path_gain_db": sim_data["ml_summary_metrics"]["mean_path_gain_db"],
                    "no_coverage_percentage": sim_data["ml_summary_metrics"]["no_coverage_percentage"],
                    "user_results": sim_data["user_metrics_for_ns3"]
                }
                timesteps_data.append(timestep_record)
                
                # CSV Row Export
                for u in sim_data["user_metrics_for_ns3"]:
                    csv_rows.append({
                        "state_id": s_idx + 1,
                        "trajectory_type": traj_type,
                        "t": t_idx,
                        "uav_x": uav_pos[0],
                        "uav_y": uav_pos[1],
                        "uav_z": uav_pos[2],
                        "num_users": len(users_info),
                        "coverage_percentage": cov_pct,
                        "mean_path_gain_db": sim_data["ml_summary_metrics"]["mean_path_gain_db"],
                        "user_id": u["user_id"],
                        "user_x": u["position"][0],
                        "user_y": u["position"][1],
                        "received_power_dbm": u["received_power_dbm"],
                        "shortest_delay_sec": u["shortest_delay_sec"] if u["shortest_delay_sec"] is not None else ""
                    })
                    
            except Exception as e:
                print(f"[ERROR] State {s_idx+1} t={t_idx} failed: {e}. Skipping.")

        disaster_states.append({
            "state_id": s_idx + 1,
            "cluster_centers": centers,
            "fixed_users": users_info,
            "trajectory_type": traj_type,
            "timesteps": timesteps_data
        })
        
        print(f"     [State {s_idx+1} finished: t=0 Coverage = {timesteps_data[0]['coverage_percentage']}% -> t={timesteps_per_state-1} Coverage = {timesteps_data[-1]['coverage_percentage']}%]")

    total_elapsed = time.time() - start_time
    
    # 5. Requirement 5 Output Structure (Schema 3.0)
    dataset_output = {
        "schema_version": "3.0",
        "notes": "Temporal earthquake-scenario dataset: clustered survivor distributions, UAV repositioning trajectories over time. Placeholder scene (simple_street_canyon).",
        "num_disaster_states": len(disaster_states),
        "total_timesteps_simulated": total_runs,
        "disaster_states": disaster_states
    }
    
    with open(output_json_file, "w") as f:
        json.dump(dataset_output, f, indent=4)
        
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(output_csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
            
    # Requirement 8: Stats Summary
    mean_t0_cov = float(np.mean(t0_coverages)) if t0_coverages else 0.0
    mean_tfinal_cov = float(np.mean(t_final_coverages)) if t_final_coverages else 0.0
    avg_improvement = round(mean_tfinal_cov - mean_t0_cov, 2)
    
    stats_summary = {
        "num_disaster_states": len(disaster_states),
        "total_runs": total_runs,
        "mean_coverage_t0_pct": round(mean_t0_cov, 2),
        "mean_coverage_final_pct": round(mean_tfinal_cov, 2),
        "average_coverage_improvement_pct": avg_improvement,
        "hypothesis_validated": bool(avg_improvement > 0)
    }
    
    with open(output_stats_file, "w") as f:
        json.dump(stats_summary, f, indent=4)
        
    print(f"\n[SUCCESS] Temporal Earthquake Dataset ({total_runs} runs) saved!")
    print(f"          JSON: '{output_json_file}'")
    print(f"          CSV: '{output_csv_file}'")
    print(f"          Stats: '{output_stats_file}' (Mean t0={mean_t0_cov:.1f}% -> Final={mean_tfinal_cov:.1f}%, Improv={avg_improvement:+.1f}%)")
    
    # Requirement 7: Generate Trajectory Plots
    plot_earthquake_sample_visualizations(dataset_output)
    
    return dataset_output


def plot_earthquake_sample_visualizations(dataset):
    """
    Requirement 7: Plots sample UAV trajectories over user clusters and coverage over time.
    """
    states = dataset.get("disaster_states", [])
    if not states:
        return
        
    sample_states = states[:min(3, len(states))]
    
    for state in sample_states:
        s_id = state["state_id"]
        traj_type = state["trajectory_type"]
        users = state["fixed_users"]
        clusters = state["cluster_centers"]
        timesteps = state["timesteps"]
        
        # 1. 2D Scatter/Line Trajectory Plot
        plt.figure(figsize=(9, 7))
        
        # Plot Clusters & Users
        ux = [u["position"][0] for u in users]
        uy = [u["position"][1] for u in users]
        plt.scatter(ux, uy, c="cyan", s=70, edgecolor="black", label="Survivor Users", zorder=3)
        
        cx = [c[0] for c in clusters]
        cy = [c[1] for c in clusters]
        plt.scatter(cx, cy, c="gold", marker="X", s=150, edgecolor="black", label="Collapsed Building Clusters", zorder=4)
        
        # Plot UAV Trajectory
        tx = [t["uav_position"][0] for t in timesteps]
        ty = [t["uav_position"][1] for t in timesteps]
        plt.plot(tx, ty, "r--o", linewidth=2.5, markersize=8, label="UAV Trajectory Path", zorder=5)
        
        # Annotate t=0 and t=final
        plt.annotate("Start (t=0)", (tx[0], ty[0]), xytext=(10, 10), textcoords="offset points", fontweight="bold", color="red", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red"))
        plt.annotate(f"Final (t={len(tx)-1})", (tx[-1], ty[-1]), xytext=(10, -15), textcoords="offset points", fontweight="bold", color="green", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green"))
        
        plt.title(f"Earthquake Scenario State #{s_id} ({traj_type.upper()})\nUAV Repositioning Trajectory Over Survivor Clusters", fontsize=12, fontweight="bold")
        plt.xlabel("X Position (meters)", fontsize=11)
        plt.ylabel("Y Position (meters)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"earthquake_trajectory_state_{s_id}.png", dpi=200)
        plt.close()
        
        # 2. Coverage % over Timesteps Line Chart
        plt.figure(figsize=(8, 5))
        times = [t["t"] for t in timesteps]
        coverages = [t["coverage_percentage"] for t in timesteps]
        
        plt.plot(times, coverages, "b-o", linewidth=2.5, markersize=8)
        plt.title(f"Coverage % Over Time — State #{s_id} ({traj_type})", fontsize=12, fontweight="bold")
        plt.xlabel("Timestep (t)", fontsize=11)
        plt.ylabel("Coverage Percentage (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.ylim(0, 100)
        for t, cov in zip(times, coverages):
            plt.annotate(f"{cov}%", (t, cov), xytext=(0, 7), textcoords="offset points", ha='center', fontsize=9, fontweight="bold")
            
        plt.tight_layout()
        plt.savefig(f"earthquake_coverage_over_time_state_{s_id}.png", dpi=200)
        plt.close()
        
        print(f"[INFO] Saved trajectory & coverage plots for State #{s_id} -> 'earthquake_trajectory_state_{s_id}.png'")


if __name__ == "__main__":
    # 1. Single Scenario Simulation with Heatmap Plot
    uavs = [[0.0, 0.0, 30.0], [40.0, 30.0, 25.0]]
    users = [[10.0, 5.0, 1.5], [-20.0, 15.0, 1.5], [35.0, 25.0, 1.5], [60.0, -10.0, 1.5]]
    
    data, grid = run_uav_propagation_sim(uavs, users, plot_filename="coverage_heatmap.png")
    with open("sionna_output_interface.json", "w") as f:
        json.dump(data, f, indent=4)
    print("\n[SUCCESS] Single simulation complete. Handoff interface saved to 'sionna_output_interface.json'")
    
    # 2. Production Run of Engine 2: Temporal Earthquake Trajectory Dataset (25 Disaster States, 8 Timesteps = 200 runs)
    print("\n--- RUNNING PRODUCTION TEMPORAL EARTHQUAKE DATASET (25 STATES, 8 TIMESTEPS) ---")
    run_earthquake_temporal_dataset(num_disaster_states=25, timesteps_per_state=8, seed=100)
    
    # 3. Production Run of Engine 1: Recommended Grid-Sweep Dataset (6 Environments, 15m Grid, 588 runs)
    print("\n--- RUNNING RECOMMENDED GRID SWEEP DATASET (6 ENVIRONMENTS, 588 RUNS) ---")
    run_grid_sweep_dataset(
        num_environments=6,
        x_grid=np.arange(-40.0, 60.0 + 1e-5, 15.0).tolist(),
        y_grid=np.arange(-40.0, 60.0 + 1e-5, 15.0).tolist(),
        z_grid=[25.0, 35.0],
        output_json_file="grid_sweep_dataset.json",
        output_csv_file="grid_sweep_dataset.csv"
    )
