"""Train the multi-UAV PPO policy across several disaster scenarios.

The policy is exposed to different scenarios AND different user
populations during training - each episode samples a
(scenario, num_users, num_uavs) combination from the *training* set in
``configs/training.json``. The evaluation scenarios are held out.

Training uses the surrogate (free-space) RF model for speed. Final
validation on real Sionna RT is done separately by
``ml/evaluate_multiuav.py``. Surrogate metrics are never labelled Sionna.

    python ml/train_multiuav.py            # full run from configs/training.json
    python ml/train_multiuav.py --smoke    # ~2000 timesteps, for the test gate
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import scenarios  # noqa: E402
from multiuav_env import MultiUAVEnv  # noqa: E402


def build_sampler(train_cfg: dict, base_seed: int):
    combos = list(
        itertools.product(
            train_cfg["scenarios"],
            train_cfg["user_counts"],
            train_cfg["num_uavs"],
        )
    )

    def sampler(episode_idx: int) -> dict:
        scen, n_users, n_uavs = combos[episode_idx % len(combos)]
        return {
            "scenario": scen,
            "num_users": int(n_users),
            "num_uavs": int(n_uavs),
            "episode_seed": base_seed * 100_000 + episode_idx,
        }

    return sampler, combos


class CsvLogger:
    """Minimal per-episode logger (SB3-agnostic, via a gym wrapper)."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        self._ep_reward = 0.0
        self._ep_len = 0

    def on_step(self, reward, info, done):
        self._ep_reward += float(reward)
        self._ep_len += 1
        if done:
            ev = info.get("eval", {}).get("aggregate", {})
            net = info.get("network", {})
            self.rows.append({
                "episode": len(self.rows) + 1,
                "scenario": info.get("scenario", info.get("eval", {}).get("scene_name", "")),
                "num_uavs": info.get("num_uavs", ev.get("num_uavs", "")),
                "num_users": info.get("num_users", ev.get("num_users", "")),
                "ep_reward": round(self._ep_reward, 4),
                "ep_len": self._ep_len,
                "coverage_pct": ev.get("coverage_percentage", ""),
                "connected_pct": ev.get("connected_users_percentage", ""),
                "mean_best_rss_dbm": ev.get("mean_best_rss_dbm", ""),
                "mean_best_sinr_db": ev.get("mean_best_sinr_db", ""),
                "net_connected_pct": round(100 * net["net_connected_fraction"], 2) if net else "",
                "predicted_pdr": round(net["predicted_pdr"], 4) if net else "",
                "calibrated_pdr": round(info["calibrated"]["calibrated_pdr"], 4)
                if info.get("calibrated") else "",
            })
            self._ep_reward = 0.0
            self._ep_len = 0

    def flush(self):
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


class LoggingEnv(gym.Wrapper):
    """Gymnasium wrapper that records per-episode stats to a CsvLogger."""

    def __init__(self, env: MultiUAVEnv, logger: CsvLogger):
        super().__init__(env)
        self.logger = logger
        self._last_info: dict = {}

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        merged = {**self._last_info, **info}
        self.logger.on_step(reward, merged, terminated or truncated)
        return obs, reward, terminated, truncated, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "training.json"))
    ap.add_argument("--smoke", action="store_true", help="~2000 timesteps for the test gate")
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    scenarios.assert_train_eval_disjoint()
    scenarios.register_scenarios(cfg.get("extra_train_scenarios", {}))
    base_seed = int(cfg.get("seed", 42))
    tr = cfg["train"]
    held_out = cfg.get("eval", {}).get("scenarios", scenarios.eval_scenarios())
    overlap = set(tr["scenarios"]) & set(held_out)
    if overlap:
        raise AssertionError(f"train / eval scenarios overlap: {overlap}")
    reward_mode = tr.get("reward_mode", "legacy")
    model_dir = _ROOT / cfg.get("model_dir", "models/ppo_multiuav")
    if args.smoke:   # never let a smoke run overwrite the real checkpoint
        model_dir = model_dir / "_smoke"
    model_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = args.timesteps or (2000 if args.smoke else int(tr["total_timesteps"]))

    sampler, combos = build_sampler(tr, base_seed)
    logger = CsvLogger(model_dir / "train_log.csv")

    def make_env():
        base = MultiUAVEnv(
            num_uavs=int(tr["num_uavs"][0]),
            num_users=int(tr["user_counts"][0]),
            scenario=tr["scenarios"][0],
            mode=tr.get("mode", "surrogate"),
            seed=base_seed,
            max_steps=int(tr.get("max_steps_per_episode", 12)),
            sampler=sampler,
            reward_mode=reward_mode,
            random_start_prob=float(tr.get("random_start_prob", 0.0)),
        )
        return LoggingEnv(base, logger)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(make_env())

    print("=" * 66)
    print("MULTI-UAV PPO TRAINING (surrogate / free-space RF model)")
    print("=" * 66)
    print(f"train scenarios : {tr['scenarios']}")
    print(f"held-out (eval) : {held_out}")
    print(f"reward mode     : {reward_mode}")
    print(f"user counts     : {tr['user_counts']}   num_uavs: {tr['num_uavs']}")
    print(f"episode combos  : {len(combos)}   total_timesteps: {total_timesteps}")
    print()

    model = PPO(
        "MlpPolicy", env,
        seed=base_seed,
        n_steps=256, batch_size=64, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.01, learning_rate=3e-4, verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    dt = time.time() - t0

    model_path = model_dir / "model.zip"
    model.save(str(model_path))
    logger.flush()

    (model_dir / "train_config.json").write_text(json.dumps({
        "base_seed": base_seed,
        "train": tr,
        "extra_train_scenarios": cfg.get("extra_train_scenarios", {}),
        "reward_mode": reward_mode,
        "eval_scenarios_held_out": held_out,
        "total_timesteps": total_timesteps,
        "smoke": args.smoke,
        "rf_model": "surrogate/free_space",
        "wall_time_sec": round(dt, 1),
        "obs_len": env.observation_space.shape[0],
        "action_space": "MultiDiscrete([5]*%d)" % scenarios.MAX_UAVS,
    }, indent=2))

    print()
    print(f"saved model -> {model_path}")
    print(f"train log   -> {model_dir / 'train_log.csv'}  ({len(logger.rows)} episodes)")
    if logger.rows:
        last = logger.rows[-len(combos):] if len(logger.rows) >= len(combos) else logger.rows
        mr = np.mean([r["ep_reward"] for r in last])
        mc = np.mean([float(r["coverage_pct"]) for r in last if r["coverage_pct"] != ""])
        print(f"last {len(last)} episodes: mean reward {mr:.3f}  mean coverage {mc:.1f}%")
    print(f"wall time   : {dt:.1f}s")


if __name__ == "__main__":
    main()
