# ResQNet — Sionna RT Wireless Propagation Module (Person 2)

This directory contains the **Sionna RT (v2.0.1)** wireless ray-tracing module for the **ResQNet** emergency UAV communications pipeline.

---

## 🏗️ Pipeline Overview

```
[ Person 1 ] GIS / 3D Mesh (.obj/.ply)
     │
     ▼
[ Person 2 ] Sionna RT 3D Ray Tracing & Radio Map Solver  <-- (THIS MODULE)
     │
     ├─────────── JSON Handoff Data Contract ───────────┐
     ▼                                                 ▼
[ Person 3 ] ML UAV Placement Model            [ Person 4 ] ns-3 Digital Twin
(Consumes coverage % & path gain)              (Consumes per-user RSS & delays)
```

---

## 📄 JSON Handoff Contract (`sionna_output_interface.json`)

The simulation outputs a structured JSON interface documenting propagation results.

### JSON Field Reference

| Key Path | Type | Unit | Range / Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `simulation_config.num_uavs` | `int` | Count | $\ge 1$ | Number of active UAV base station transmitters. |
| `simulation_config.uav_positions` | `list[list]` | meters | $[x, y, z]$ | 3D coordinates of UAV transmitters. |
| `simulation_config.carrier_frequency_hz` | `float` | Hz | e.g. `2.1e9` | Carrier radio frequency ($2.1\text{ GHz}$ LTE/5G). |
| `simulation_config.tx_power_dbm` | `float` | dBm | e.g. `30.0` | Transmit RF power output ($30\text{ dBm} = 1\text{ W}$). |
| `ml_summary_metrics.coverage_percentage` | `float` | % | `0.0` to `100.0` | Percentage of ground grid cells exceeding $-100\text{ dB}$ path gain threshold. |
| `ml_summary_metrics.mean_path_gain_db` | `float` | dB | $[-200, 0]$ | Unbiased average path gain computed **only** over valid coverage cells (excluding dead zones). |
| `ml_summary_metrics.min_valid_path_gain_db` | `float` | dB | $[-200, 0]$ | Minimum path gain recorded among cells with at least 1 valid ray path. |
| `ml_summary_metrics.max_path_gain_db` | `float` | dB | $[-200, 0]$ | Maximum path gain recorded (closest point to Tx). |
| `ml_summary_metrics.no_coverage_cell_count` | `int` | Count | $\ge 0$ | Number of grid cells with 0 ray paths (dead zones). |
| `ml_summary_metrics.no_coverage_percentage` | `float` | % | `0.0` to `100.0` | Percentage of dead zone grid cells. |
| `user_metrics_for_ns3[i].user_id` | `string` | — | `"user_1"` | Identifier string for ground survivor node. |
| `user_metrics_for_ns3[i].position` | `list` | meters | $[x, y, z]$ | 3D position of ground user. |
| `user_metrics_for_ns3[i].received_power_dbm` | `float` | dBm | $[-150, +30]$ | Total received RF signal power at user antenna. |
| `user_metrics_for_ns3[i].path_gain_linear` | `float` | ratio | $[0.0, 1.0]$ | Total linear power gain ($|a|^2$). |
| `user_metrics_for_ns3[i].shortest_delay_sec` | `float` | seconds | $> 0.0$ | Time delay of the shortest direct line-of-sight ray path. |
| `user_metrics_for_ns3[i].num_paths` | `int` | Count | $\ge 0$ | Number of distinct ray-traced multipath reflections reaching user. |

---

## 🌋 Temporal Earthquake Trajectory Contract (`earthquake_trajectory_dataset.json` / `earthquake_trajectory_dataset.csv` - Schema 3.0)

Simulates dynamic UAV repositioning trajectories over earthquake survivor clusters (collapsed building assembly points):

### Top-Level Metadata
* **`schema_version`**: `"3.0"`
* **`num_disaster_states`**: Number of simulated disaster environment states.
* **`total_timesteps_simulated`**: Total number of evaluated trajectory timesteps.

### Per-Disaster State Schema (`disaster_states[i]`)
* **`state_id`**: Unique disaster state index.
* **`cluster_centers`**: $2D$ coordinates $[[x_1, y_1], [x_2, y_2], \dots]$ of collapsed building survivor clusters.
* **`fixed_users`**: Array of ground survivor objects with `user_id`, `position`, and `cluster_id`.
* **`trajectory_type`**: `"toward_cluster"` (converging on survivor centroid) or `"greedy_exploration"` (bounded random walk).
* **`timesteps`**: Array of per-timestep evaluations (`t`, `uav_position`, `coverage_percentage`, `mean_path_gain_db`, `no_coverage_percentage`, `user_results`).

---

## 🗺️ Grid-Sweep ML Dataset Contract (`grid_sweep_dataset.json` / `grid_sweep_dataset.csv`)

The Grid-Sweep Engine systematically evaluates coverage outcomes across a dense 3D spatial grid for multiple fixed disaster environments:

### Top-Level Metadata
* **`schema_version`**: `"1.0"`
* **`num_environments`**: Number of distinct fixed ground user environments ($N$).
* **`total_runs`**: Total number of evaluated grid points ($N \times \text{Grid Points}$).

### Per-Environment Schema (`environments[i]`)
* **`env_id`**: Unique environment identifier (e.g. `"env_1"`).
* **`num_users`**: Count of ground users (survivors) in this environment (8–15).
* **`user_positions`**: Fixed 3D coordinates $[x, y, 1.5]$ of survivors.
* **`runs`**: Array of grid evaluation records (`run_id`, `uav_x`, `uav_y`, `uav_z`, `coverage_percentage`, `mean_path_gain_db`, `no_coverage_percentage`, `user_results`).

---

## 📊 Batch ML Dataset Contract (`batch_uav_dataset.json`)

The batch scenario generator outputs a dataset format designed specifically for Person 3 (ML Model Training):

### Top-Level Metadata
* **`schema_version`**: `"1.0"`
* **`notes`**: Informational disclaimer regarding placeholder geometry vs final OBJ/PLY mesh.
* **`dataset_size`**: Number of simulated UAV deployment scenarios.

### Per-Scenario Schema (`scenarios[i]`)
* **`scenario_id`**: Unique scenario index.
* **`uav_positions`**: List of 3D UAV coordinates $[x, y, z]$.
* **`coverage_percentage`**: Global spatial coverage percentage ($\% > -100\text{ dB}$).
* **`mean_path_gain_db`**: Unbiased mean path gain ($\text{dB}$) across non-dead-zone cells.
* **`no_coverage_percentage`**: Dead zone percentage ($\%$ cells with 0 rays).
* **`user_results`**: Array of structured user objects containing `user_id`, `position`, `received_power_dbm`, `path_gain_linear`, `shortest_delay_sec`, and `num_paths`.

---

## 🚀 Execution Instructions

### 1. Run Single Scenario Simulation & Heatmap
```powershell
python sionna_pipeline.py
```
* Generates `sionna_output_interface.json` (handoff contract).
* Saves `coverage_heatmap.png` (2D heatmap overlaying red UAV triangles and cyan User circles).
* Generates `batch_uav_dataset.json` (multi-scenario dataset for Person 3's ML training).

---

## 🛠️ Module Functions Reference

* **`run_uav_propagation_sim(uav_positions, user_positions, scene_name, freq, tx_power_dbm)`**:
  Executes ray tracing (`PathSolver`) and grid solver (`RadioMapSolver`), returns metrics dictionary and dB path gain matrix.

* **`safe_load_scene(scene_name)`**:
  Handles custom `.obj` or `.ply` mesh files from Person 1, with automatic fallback to built-in scenes and default material assignment (`itu_concrete`).

* **`batch_run_scenarios(uav_configs_list, user_positions, output_dataset_file)`**:
  Batch executes multiple UAV candidate configurations and aggregates training data for Person 3.
