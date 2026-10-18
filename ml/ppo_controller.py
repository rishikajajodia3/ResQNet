"""PPO policy for the closed-loop controller (closed_loop_control.py,
config "policy": "ppo"). The rule-based controller stays the default.

One control interval in PPO mode:
  1. the controller's last observation holds Sionna path gains at the current
     positions (ns-3 frequency) and the ns-3 MEASURED per-user PDR there;
  2. those are turned into the exact observation vector used in training
     (MultiUAV3DEnv.load_state; measured PDR replaces the training model's
     predicted PDR);
  3. the MaskablePPO checkpoint picks one of 7 actions per UAV, with unsafe
     moves masked using the controller's real ray-cast scene geometry;
  4. the same safety filter as in training (single-move rule + joint
     separation) is applied, every intervention is recorded;
  5. the controller moves the UAVs, re-runs Sionna, hands the obstruction
     matrix to ns-3 and measures again (closed_loop_control.observe), and the
     new measurement is the next interval's input.
PPO moves are executed as chosen (no rule-based accept / revert).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import multiuav3d_env as E


class PPOPolicy:
    def __init__(self, ppo_cfg: dict, ctl_cfg: dict):
        ck = ppo_cfg.get("checkpoint")
        if not ck:
            raise ValueError('policy "ppo" needs ppo.checkpoint (path to model.zip)')
        self.path = Path(ck).expanduser()
        tc_path = self.path.parent / "train_config.json"
        self.train_cfg = json.loads(tc_path.read_text()) if tc_path.exists() else {}
        if self.train_cfg.get("obs_version") != E.OBS_VERSION:
            raise ValueError(f"checkpoint observation version {self.train_cfg.get('obs_version')!r} "
                             f"!= {E.OBS_VERSION!r}; not a 3D policy for this controller")
        for k in ("step_h_m", "step_v_m"):
            if float(self.train_cfg[k]) != float(ctl_cfg[k]):
                raise ValueError(f"controller {k}={ctl_cfg[k]} but the policy was trained with {self.train_cfg[k]}")
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(self.path, device="cpu")
        if list(self.model.action_space.nvec) != [E.N_ACTIONS] * E.MAX_UAVS:
            raise ValueError("checkpoint action space is not MultiDiscrete([7]*5)")
        self.deterministic = bool(ppo_cfg.get("deterministic", True))
        self.max_steps = int(self.train_cfg.get("max_steps", 12))
        self.info = {"policy": "ppo", "checkpoint": str(self.path),
                     "checkpoint_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                     "run_id": self.train_cfg.get("run_id"), "obs_version": E.OBS_VERSION,
                     "trained_with_rf": self.train_cfg.get("rf_label"),
                     "trained_pdr_model": self.train_cfg.get("pdr_model"),
                     "latency_in_reward": self.train_cfg.get("latency_in_reward", False),
                     "deterministic": self.deterministic, "action_names": E.ACTION_NAMES}

    def _env(self, ctl) -> E.MultiUAV3DEnv:
        w = {"closed_loop": {"role": "evaluation", "uav_region": ctl.region, "user_region": ctl.user_region}}
        return E.MultiUAV3DEnv(w, ctl.geometry, rf=None, pdr_model=None, step_h_m=ctl.cfg["step_h_m"],
                               step_v_m=ctl.cfg["step_v_m"], safety=ctl.cfg["safety"],
                               coverage_threshold=float(self.train_cfg.get("coverage_threshold", 0.5)),
                               max_steps=self.max_steps, allow_held_out=True)

    def select(self, ctl, obs: dict) -> tuple[dict, list[dict]]:
        env = self._env(ctl)
        o = env.load_state(ctl.region, ctl.positions, obs["users"], obs["pg"], obs["meas"]["per_user_pdr"],
                           t=ctl.step_idx - 1)
        mask = env.action_masks()
        a, _ = self.model.predict(o, deterministic=self.deterministic, action_masks=mask)
        a = np.asarray(a).reshape(-1)
        T, acts, unsafe, cancelled = env._apply(a)
        m2 = mask.reshape(E.MAX_UAVS, E.N_ACTIONS)
        moves, records = {}, []
        for k in range(ctl.K):
            raw, ex = E.ACTION_NAMES[int(a[k])], E.ACTION_NAMES[acts[k]]
            blocked = [E.ACTION_NAMES[j] for j in range(1, E.N_ACTIONS) if not m2[k, j]]
            if raw != ex:
                safety = f"{raw} cancelled by the safety filter (conflict with another UAV's move)"
            else:
                safety = None
            r = {"uav": k, "action": ex, "policy_action": raw, "kind": "ppo", "from": list(ctl.positions[k]),
                 "to": [round(float(c), 2) for c in T[k]], "masked_actions": blocked, "safety_intervention": safety,
                 "reason": f"PPO chose {raw}" + (f"; {safety}" if safety else "")
                           + (f"; masked as unsafe: {', '.join(blocked)}" if blocked else "")}
            if ex == "stay":
                r["outcome"] = "held"
            else:
                r["outcome"] = "pending"
                moves[k] = ex
            records.append(r)
        return moves, records
