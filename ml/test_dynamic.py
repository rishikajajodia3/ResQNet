import numpy as np

from dynamic_model import UAVDynamicEnv


print("=" * 60)
print("DYNAMIC UAV ENVIRONMENT")
print("=" * 60)


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = UAVDynamicEnv()

print()
print("Environment loaded successfully")

print(
    "Number of trajectories:",
    env.num_trajectories
)


# =========================================================
# RESET
# =========================================================

observation, info = env.reset()


print()
print("=" * 60)
print("RESET")
print("=" * 60)


print("Initial observation:")
print(observation)

print()
print("Initial info:")
print(info)


# =========================================================
# INITIAL STATE
# =========================================================

previous_coverage = float(
    observation[3]
)


print()
print("=" * 60)
print("INITIAL STATE")
print("=" * 60)


print(
    f"UAV position: "
    f"X={observation[0]:.2f}, "
    f"Y={observation[1]:.2f}, "
    f"Z={observation[2]:.2f}"
)


print(
    f"Coverage: "
    f"{observation[3]:.2f}%"
)


print(
    f"Mean received power: "
    f"{observation[4]:.2f} dBm"
)


print(
    f"Number of users: "
    f"{observation[5]:.0f}"
)


# =========================================================
# ACTION DEFINITIONS
# =========================================================

print()
print("=" * 60)
print("ACTION DEFINITIONS")
print("=" * 60)

print("0 = Stay")
print("1 = Move left  (-X)")
print("2 = Move right (+X)")
print("3 = Move down  (-Y)")
print("4 = Move up    (+Y)")


# =========================================================
# TEST ACTIONS
# =========================================================

actions = [

    1,  # left
    2,  # right
    3,  # down
    4,  # up
    0   # stay

]


print()
print("=" * 60)
print("TESTING DYNAMIC MOVEMENT")
print("=" * 60)


for step_number, action in enumerate(
    actions,
    start=1
):

    old_position = (
        observation[:3].copy()
    )

    old_coverage = float(
        observation[3]
    )

    print()
    print(
        f"Step {step_number}"
    )

    print("-" * 40)

    print(
        f"Action: {action}"
    )

    # =====================================================
    # TAKE ACTION
    # =====================================================

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    # =====================================================
    # NEW STATE
    # =====================================================

    new_position = (
        observation[:3]
    )

    new_coverage = float(
        observation[3]
    )

    mean_power = float(
        observation[4]
    )

    # =====================================================
    # POSITION
    # =====================================================

    print(
        f"Position: "
        f"({old_position[0]:.2f}, "
        f"{old_position[1]:.2f}, "
        f"{old_position[2]:.2f})"
        f" -> "
        f"({new_position[0]:.2f}, "
        f"{new_position[1]:.2f}, "
        f"{new_position[2]:.2f})"
    )

    # =====================================================
    # COVERAGE
    # =====================================================

    print(
        f"Coverage: "
        f"{old_coverage:.2f}% -> "
        f"{new_coverage:.2f}%"
    )

    print(
        f"Coverage change: "
        f"{new_coverage - old_coverage:+.2f}%"
    )

    # =====================================================
    # REWARD
    # =====================================================

    print(
        f"Reward: "
        f"{reward:.2f}"
    )

    # =====================================================
    # POWER
    # =====================================================

    print(
        f"Mean received power: "
        f"{mean_power:.2f} dBm"
    )

    # =====================================================
    # INFO
    # =====================================================

    print(
        f"Action applied: "
        f"{info['action']}"
    )

    print(
        f"Terminated: "
        f"{terminated}"
    )

    print(
        f"Truncated: "
        f"{truncated}"
    )

    # =====================================================
    # STOP IF EPISODE ENDS
    # =====================================================

    if terminated or truncated:

        print()
        print(
            "Episode ended."
        )

        break


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 60)
print("DYNAMIC TEST COMPLETE")
print("=" * 60)


env.close()