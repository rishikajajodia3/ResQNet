from pathlib import Path
import json
from stable_baselines3 import PPO

from dynamic_model import UAVDynamicEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


print("=" * 60)
print("PPO UAV POSITION SELECTION")
print("=" * 60)

# Load environment
env = UAVDynamicEnv()

# Load trained PPO model
model = PPO.load(MODEL_PATH)

# Reset environment
observation, info = env.reset()

print()
print("Initial UAV position:")
print(
    f"X={observation[0]:.2f}, "
    f"Y={observation[1]:.2f}, "
    f"Z={observation[2]:.2f}"
)

# Let PPO choose an action
action, _ = model.predict(
    observation,
    deterministic=True
)

action = int(action)

print()
print("PPO selected action:", action)

# Apply action
observation, reward, terminated, truncated, info = env.step(action)

x = float(observation[0])
y = float(observation[1])
z = float(observation[2])

print()
print("PPO UAV Position:")
print(f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

print(f"Coverage: {observation[3]:.2f}%")
print(f"Reward: {reward:.2f}")

# Save position for ns-3
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

data = {
    "num_uavs": 1,
    "uav_positions": [
        [x, y, z]
    ]
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print()
print("Saved UAV position to:")
print(OUTPUT_PATH)

print("=" * 60)
print("PPO POSITION SELECTION COMPLETE")
print("=" * 60)
