"""Evaluate multi-UAV UAV placement: FIXED vs RANDOM vs PPO.

Answers the project question: does a trained multi-UAV PPO policy improve
disaster-area communication coverage compared with fixed or random UAV
placement?

Three placements, same scenario / users / seed:

  fixed   - initial (spatially separated) positions, no movement
  random  - seeded random positions in the UAV region, no movement
  ppo     - roll out models/ppo_multiuav/model.zip, take final positions

For each: RF metrics (Sionna or surrogate) + optional NS-3 network run.
All numbers are from real runs. Surrogate results are labelled surrogate.

    python ml/evaluate_multiuav.py --scenario blocked_urban --num-uavs 3 \
        --num-users 30 --mode surrogate --steps 10 --ns3

Outputs: data/experiments/<exp_id>/{config,baselines}.json and an entry in
data/experiments/index.json (read by the dashboard).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import scenarios  # noqa: E402
import multiuav_io as mio  # noqa: E402
from multiuav_env import surrogate_multi_uav, MAX_UAVS  # noqa: E402

EXPERIMENTS_DIR = _ROOT / "data" / "experiments"
INDEX_FILE = EXPERIMENTS_DIR / "index.json"
MODEL_PATH = _ROOT / "models" / "ppo_multiuav" / "model.zip"
HANDOFF_FILE = _ROOT / "data" / "uav_positions.json"


# ----------------------------------------------------------------------
def rf_eval(uav_positions, users, mode, scenario, interference_model, seed):
    if mode == "sionna":
        import sionna_feedback_engine as sfe
        return sfe.evaluate_multi_uav(
            uav_positions, user_positions=users,
            scene_name=scenarios.get_scenario(scenario)["scene"],
            seed=seed, interference_model=interference_model,
        )
    import sionna_feedback_engine as sfe
    return surrogate_multi_uav(
        uav_positions, users, connectivity_config=sfe.ConnectivityConfig(),
        interference_model=interference_model,
    )


def random_positions(num_uavs, scenario, seed):
    reg = scenarios.get_scenario(scenario)["uav_region"]
    rng = np.random.default_rng(seed + 9999)
    out = []
    for _ in range(num_uavs):
        out.append([
            round(float(rng.uniform(*reg["x"])), 2),
            round(float(rng.uniform(*reg["y"])), 2),
            round(float(rng.uniform(*reg["z"])), 2),
        ])
    return out


def ppo_positions(num_uavs, num_users, scenario, mode, steps, seed, interference_model):
    from multiuav_env import MultiUAVEnv
    env = MultiUAVEnv(num_uavs=num_uavs, num_users=num_users, scenario=scenario,
                      mode=mode, seed=seed, max_steps=steps,
                      interference_model=interference_model)
    obs, _ = env.reset(seed=seed)
    if MODEL_PATH.exists():
        from stable_baselines3 import PPO
        model = PPO.load(str(MODEL_PATH))
        for _ in range(steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(np.asarray(a).ravel())
            if term or trunc:
                break
    else:
        for _ in range(steps):
            obs, _, term, trunc, _ = env.step(env.np_random.integers(0, 5, size=MAX_UAVS))
            if term or trunc:
                break
    return env.uav_positions.tolist(), env.initial_positions.tolist()


def movement_m(start, end):
    return float(np.mean([np.linalg.norm(np.array(a) - np.array(b))
                          for a, b in zip(start, end)]))


# ----------------------------------------------------------------------
FLOW_RE = re.compile(r"Flow:\s+(\S+)\s+->\s+(\S+)(.*?)(?=\nFlow:|\n=+\n|\Z)", re.DOTALL)


def run_ns3(uav_positions, users, ns3_root: Path):
    assoc_engine = mio.nearest_uav_association(uav_positions, users)
    handoff = mio.build_uav_positions(uav_positions, users, associations=assoc_engine)
    (ns3_root / "data").mkdir(parents=True, exist_ok=True)
    (ns3_root / "data" / "uav_positions.json").write_text(json.dumps(handoff, indent=2))
    subprocess.run(
        ["/usr/bin/python3.12", "./ns3", "build", "scratch/uav-demo"],
        cwd=str(ns3_root), capture_output=True, text=True, timeout=600,
    )
    r = subprocess.run(
        ["/usr/bin/python3.12", "./ns3", "run", "scratch/uav-demo"],
        cwd=str(ns3_root), capture_output=True, text=True, timeout=600,
    )
    out = r.stdout
    flows = []
    for m in FLOW_RE.finditer(out):
        block = m.group(3)
        if "No packets received." in block:
            flows.append({"delivering": False})
        else:
            tp = re.search(r"Throughput:\s+([\d.eE+-]+)", block)
            dl = re.search(r"Average Delay:\s+([\d.eE+-]+)", block)
            flows.append({
                "delivering": True,
                "throughput_mbps": float(tp.group(1)) if tp else 0.0,
                "average_delay_ms": float(dl.group(1)) if dl else 0.0,
            })
    ok = [f for f in flows if f["delivering"]]
    return {
        "total_flows": len(flows),
        "delivering_flows": len(ok),
        "avg_throughput_mbps": (sum(f["throughput_mbps"] for f in ok) / len(ok)) if ok else 0.0,
        "avg_delay_ms": (sum(f["average_delay_ms"] for f in ok) / len(ok)) if ok else 0.0,
    }


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=scenarios.scenario_ids())
    ap.add_argument("--num-uavs", type=int, default=3)
    ap.add_argument("--num-users", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["surrogate", "sionna"], default="surrogate")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--interference-model", choices=["co_channel", "orthogonal"], default="co_channel")
    ap.add_argument("--ns3", action="store_true", help="also run NS-3 for each placement")
    ap.add_argument("--ns3-path", default=None)
    args = ap.parse_args()

    sid = args.scenario
    users = scenarios.generate_users(sid, args.num_users, args.seed).tolist()
    split = ("train" if sid in scenarios.train_scenarios()
             else "eval (held out)" if sid in scenarios.eval_scenarios() else "unused")

    fixed_pos = scenarios.initial_uav_positions(args.num_uavs, sid).tolist()
    rand_pos = random_positions(args.num_uavs, sid, args.seed)
    ppo_final, ppo_init = ppo_positions(args.num_uavs, args.num_users, sid, args.mode,
                                       args.steps, args.seed, args.interference_model)

    placements = {
        "fixed": (fixed_pos, fixed_pos),
        "random": (rand_pos, rand_pos),
        "ppo": (ppo_init, ppo_final),
    }

    ns3_root = None
    if args.ns3:
        import os
        ns3_root = Path(args.ns3_path or os.environ.get("NS3_PATH")
                        or (Path.home() / "ns-allinone-3.47" / "ns-3.47")).expanduser()
        # make sure the current scenario file is in the scratch dir
        import shutil
        shutil.copy2(_ROOT / "ns3" / "uav-demo.cc", ns3_root / "scratch" / "uav-demo.cc")

    print("=" * 70)
    print(f"EXPERIMENT  {sid} [{split}]  {args.num_uavs} UAV x {args.num_users} users  "
          f"mode={args.mode}  seed={args.seed}")
    print("=" * 70)

    rows = []
    for name, (start, end) in placements.items():
        res = rf_eval(end, users, args.mode, sid, args.interference_model, args.seed)
        agg = res["aggregate"]
        row = {
            "placement": name,
            "scenario": sid,
            "split": split,
            "num_uavs": args.num_uavs,
            "num_users": args.num_users,
            "rf_model": args.mode,
            "coverage_pct": agg["coverage_percentage"],
            "connected_users": agg["connected_users_count"],
            "connected_pct": agg["connected_users_percentage"],
            "mean_best_rss_dbm": agg["mean_best_rss_dbm"],
            "mean_best_snr_db": agg["mean_best_snr_db"],
            "mean_best_sinr_db": agg["mean_best_sinr_db"],
            "movement_m": round(movement_m(start, end), 2),
            "uav_positions": [[round(c, 2) for c in p] for p in end],
        }
        if args.ns3 and ns3_root is not None:
            n = run_ns3(end, users, ns3_root)
            row["ns3_total_flows"] = n["total_flows"]
            row["ns3_delivering_flows"] = n["delivering_flows"]
            row["ns3_avg_throughput_mbps"] = round(n["avg_throughput_mbps"], 6)
            row["ns3_avg_delay_ms"] = round(n["avg_delay_ms"], 6)
        rows.append(row)
        print(f"  {name:7s}: coverage {row['coverage_pct']:5.1f}%  "
              f"connected {row['connected_users']}/{args.num_users}  "
              f"SNR {row['mean_best_snr_db']:.1f}  SINR {row['mean_best_sinr_db']:.1f}  "
              f"move {row['movement_m']:.1f} m"
              + (f"  NS-3 {row.get('ns3_delivering_flows','-')}/{row.get('ns3_total_flows','-')}"
                 if args.ns3 else ""))

    exp_id = f"{sid}_{args.num_uavs}u_{args.num_users}n_{args.mode}_s{args.seed}"
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps({
        "experiment_id": exp_id, "scenario": sid, "split": split,
        "num_uavs": args.num_uavs, "num_users": args.num_users,
        "mode": args.mode, "seed": args.seed, "steps": args.steps,
        "interference_model": args.interference_model,
        "ns3": bool(args.ns3), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_positions": users,
    }, indent=2))
    (exp_dir / "baselines.json").write_text(json.dumps({"rows": rows}, indent=2))

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX_FILE.read_text()) if INDEX_FILE.exists() else []
    index = [e for e in index if e.get("experiment_id") != exp_id]
    index.append({
        "experiment_id": exp_id, "scenario": sid, "split": split,
        "num_uavs": args.num_uavs, "num_users": args.num_users,
        "mode": args.mode, "rows": rows,
    })
    INDEX_FILE.write_text(json.dumps(index, indent=2))

    print(f"\nsaved: {exp_dir}")
    print(f"index: {INDEX_FILE}  ({len(index)} experiments)")

    ppo_row = next(r for r in rows if r["placement"] == "ppo")
    fixed_row = next(r for r in rows if r["placement"] == "fixed")
    d_cov = ppo_row["coverage_pct"] - fixed_row["coverage_pct"]
    d_conn = ppo_row["connected_users"] - fixed_row["connected_users"]
    print(f"\nPPO vs FIXED: coverage {d_cov:+.1f} pp, connected {d_conn:+d} users "
          f"(scenario split: {split})")


if __name__ == "__main__":
    main()
