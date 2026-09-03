import argparse
import sys
from pathlib import Path
import json
from stable_baselines3 import PPO

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

from dynamic_model import UAVDynamicEnv


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_uav_dynamic"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "uav_positions.json"
)


def run_ppo_episode(use_sionna: bool = True, max_steps: int = 8):
    mode_str = "LIVE SIONNA RT (PHYSICS-IN-THE-LOOP)" if use_sionna else "SURROGATE (RANDOM FOREST)"
    print("=" * 75)
    print(f"PPO UAV CLOSED-LOOP EPISODE EXECUTION -- MODE: {mode_str}")
    print("=" * 75)

    # 1. Initialize Environment
    env = UAVDynamicEnv(use_sionna_feedback=use_sionna, scene_name="city_damaged.xml")
    
    # 2. Load trained PPO model (if exists)
    model = None
    if MODEL_PATH.with_suffix(".zip").exists() or MODEL_PATH.exists():
        try:
            model = PPO.load(MODEL_PATH)
            print(f"[INFO] Loaded trained PPO policy from: {MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Failed loading model ({e}). Using deterministic exploration policy.")
            model = None
    else:
        print("[INFO] No pre-trained zip found. Using deterministic exploration policy.")

    # 3. Reset Environment
    observation, info = env.reset(seed=42)
    
    print("\nInitial State (t=0):")
    print(f"  * UAV Position:      [X={observation[0]:.2f}, Y={observation[1]:.2f}, Z={observation[2]:.2f}]")
    print(f"  * Initial Coverage:  {observation[3]:.2f}%")
    print(f"  * Initial Mean RSS:  {observation[4]:.2f} dBm")
    
    if info.get("sionna_metrics"):
        sm = info["sionna_metrics"]["aggregate"]
        print(f"  * Connected Users:   {sm['connected_users_count']} / 10 ({sm['connected_users_percentage']}%)")

    total_reward = 0.0
    step_history = []

    print("\n" + "-" * 75)
    print(f"{'Step':<6} | {'Action':<8} | {'UAV Position (X, Y, Z)':<25} | {'Connected':<11} | {'Coverage %':<11} | {'Reward':<8}")
    print("-" * 75)

    for step_num in range(1, max_steps + 1):
        # 4. Predict PPO action
        if model is not None:
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
        else:
            action = (step_num % 4) + 1 if step_num < 4 else 0

        action_names = {0: "0 (Stay)", 1: "1 (Move -X)", 2: "2 (Move +X)", 3: "3 (Move -Y)", 4: "4 (Move +Y)"}
        act_label = action_names.get(action, str(action))

        # 5. Apply action in environment
        prev_pos = [round(float(p), 2) for p in observation[:3]]
        observation, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        new_pos = [round(float(p), 2) for p in observation[:3]]
        cov = float(observation[3])
        
        print(f"\nStep {step_num}")
        print(f"  * Action:                  {act_label}")
        print(f"  * Previous UAV position:   [X={prev_pos[0]:.2f}, Y={prev_pos[1]:.2f}, Z={prev_pos[2]:.2f}]")
        print(f"  * New UAV position:        [X={new_pos[0]:.2f}, Y={new_pos[1]:.2f}, Z={new_pos[2]:.2f}]")
        
        if step_info.get("sionna_metrics"):
            agg = step_info["sionna_metrics"]["aggregate"]
            print(f"  * Connected users:         {agg['connected_users_count']} / 10 ({agg['connected_users_percentage']}%)")
            print(f"  * Coverage %:              {agg['coverage_percentage']:.1f}%")
            print(f"  * Mean path gain:          {agg['mean_path_gain_db']:.2f} dB")
            print(f"  * Mean RSS:                {agg['mean_rss_dbm']:.2f} dBm")
            snr_str = f"{agg['mean_sinr_db']:.2f} dB (Thermal noise floor = -94 dBm, I = 0)" if agg.get('mean_sinr_db') is not None else "None"
            print(f"  * Mean SNR:                {snr_str}")
        else:
            print(f"  * Coverage % (Surrogate):  {cov:.1f}%")
            print(f"  * Mean RSS (Surrogate):    {float(observation[4]):.2f} dBm")

        print(f"  * Reward:                  {reward:+.4f}")
        
        step_history.append({
            "step": step_num,
            "action": action,
            "previous_position": prev_pos,
            "new_position": new_pos,
            "coverage": cov,
            "reward": reward,
            "sionna_metrics": step_info.get("sionna_metrics")
        })

        if terminated or truncated:
            print(f"\n[INFO] Episode terminated at step {step_num}.")
            break

    final_x, final_y, final_z = round(float(observation[0]), 2), round(float(observation[1]), 2), round(float(observation[2]), 2)
    print("-" * 75)
    print(f"\nEpisode Finished! Total Cumulative Reward: {total_reward:.4f}")
    print(f"Final PPO-Selected UAV Position: [X={final_x}, Y={final_y}, Z={final_z}]")

    # 6. Save final position for NS-3 handoff
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    handoff_data = {
        "num_uavs": 1,
        "uav_positions": [
            [final_x, final_y, final_z]
        ]
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(handoff_data, f, indent=2)

    print(f"\n[SUCCESS] Exported final UAV position to NS-3 handoff file: '{OUTPUT_PATH}'")
    print(f"Content:\n{json.dumps(handoff_data, indent=2)}")

    return step_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO UAV dynamic placement episode")
    parser.add_argument("--mode", choices=["sionna", "surrogate"], default="sionna", help="Simulation mode")
    parser.add_argument("--steps", type=int, default=5, help="Number of episode steps")
    args = parser.parse_args()

    run_ppo_episode(use_sionna=(args.mode == "sionna"), max_steps=args.steps)
