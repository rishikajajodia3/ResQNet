from pathlib import Path

from stable_baselines3 import PPO

from dynamic_model import UAVDynamicEnv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_uav_dynamic"
)


def main():

    print("=" * 60)
    print("TRAINING PPO FOR DYNAMIC UAV PLACEMENT")
    print("=" * 60)

    # ==================================================
    # ENVIRONMENT
    # ==================================================

    env = UAVDynamicEnv()

    print()
    print(
        "Environment loaded successfully"
    )

    print(
        "Number of earthquake trajectories:",
        env.num_trajectories
    )

    print(
        "Raw earthquake rows:",
        len(env.df)
    )

    # ==================================================
    # PPO
    # ==================================================

    model = PPO(

        "MlpPolicy",

        env,

        learning_rate=3e-4,

        n_steps=64,

        batch_size=32,

        gamma=0.99,

        gae_lambda=0.95,

        ent_coef=0.01,

        verbose=1,

        seed=42
    )

    # ==================================================
    # TRAIN
    # ==================================================

    print()
    print("=" * 60)
    print("STARTING PPO TRAINING")
    print("=" * 60)

    model.learn(
        total_timesteps=50_000
    )

    # ==================================================
    # SAVE
    # ==================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 60)
    print("PPO TRAINING COMPLETE")
    print("=" * 60)

    print()
    print("Model saved:")
    print(MODEL_PATH)

    env.close()


if __name__ == "__main__":

    main()