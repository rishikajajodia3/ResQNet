# ResQNet — Multi-UAV Architecture (honest current state)

This document describes the multi-UAV / multi-scenario upgrade of ResQNet.
It is written to be accurate about **what works** and **what the current
result actually is**, not to oversell it.

The original single-UAV / 10-user pipeline is **unchanged** and still runs:

```bash
python scripts/run_full_pipeline.py --mode sionna --steps 5
```

Multi-UAV is opt-in: pass `--num-uavs / --num-users / --scenario` and the
same scripts run the multi-UAV path instead.

---

## 1. Complete pipeline

```
scenario (configs/scenarios.json)
  → deterministic user layout + separated initial UAV positions (ml/scenarios.py)
  → RF evaluation
       • Sionna RT  : sionna/sionna_feedback_engine.py :: evaluate_multi_uav()
                      K transmitters, ONE PathSolver call, per-(user,UAV) metrics
       • surrogate  : ml/multiuav_env.py :: surrogate_multi_uav()
                      free-space path loss, LOS only, NO buildings
  → best-RSS serving-UAV association per user
  → co-channel SINR from the ray-traced (or free-space) received powers
  → multi-UAV PPO   : ml/multiuav_env.py :: MultiUAVEnv
                      action = MultiDiscrete([5] * MAX_UAVS)  (per UAV: stay/±X/±Y)
                      observation = fixed 71-dim vector (independent of K and N)
  → UAV repositioning
  → data/uav_positions.json   { num_uavs, uav_positions[K], num_users,
                                user_positions[N], associations[N] }
  → data/ppo_trajectory.json  per step: uav_positions[K], actions[K],
                                per-user { serving_uav, per_uav[K], rss, snr, sinr, ... }
  → NS-3 (ns3/uav-demo.cc)    K UAV nodes, N user nodes, ONE UdpServer per UAV,
                              each user is a UdpClient of its associated UAV,
                              FlowMonitor → one flow per user
  → data/full_pipeline_results.json   ns3_summary + flows[] + per_uav[] breakdown
  → dashboard (dashboard/)    3D scene with K UAVs, per-user link to serving UAV,
                              Sionna-connected vs NS-3-delivery shown separately
```

Backward compatibility: for `num_uavs == 1` the handoff file, the trajectory
step schema, and the NS-3 stdout / topology are **byte-identical** to the
legacy pipeline (verified by tests and by `diff` against the committed
`data/full_pipeline_results.json`).

---

## 2. Training vs validation

| | RF model | Purpose | Where |
|---|---|---|---|
| **Training** | **surrogate / free-space** (fast, deterministic, LOS only) | let PPO see many spatial configs cheaply | `ml/train_multiuav.py`, `configs/training.json` |
| **Validation** | **real Sionna RT ray tracing** | check the policy / placement on the actual 3D mesh | `ml/evaluate_multiuav.py --mode sionna` |

Surrogate results are **never** labelled as Sionna. In code:
`surrogate_multi_uav()` stamps `scene_name = "surrogate:free_space"`;
`train_config.json` records `rf_model = "surrogate/free_space"`; the
dashboard `/api/training/summary` says *"surrogate / free-space (training).
Sionna used only for validation."*

The trained model is `models/ppo_multiuav/model.zip` (git-ignored,
reproducible with `python ml/train_multiuav.py`). Current model: 120 000
timesteps, 10 005 episodes, seed 42, ~5 min wall time, `explained_variance
≈ 0.965`.

---

## 3. Scenario split (train / held-out)

`configs/training.json`, enforced in code by
`scenarios.assert_train_eval_disjoint()`:

| Scenario | Scene | Distribution | Split |
|---|---|---|---|
| `open_area` | city_damaged | uniform, wide | **train** |
| `dense_urban` | city_baseline | uniform | **train** |
| `clustered_survivors` | city_damaged | 3 clusters | **train** |
| `blocked_urban` | city_damaged | uniform | **held out** |
| `wide_area` | city_damaged | random, large footprint | **held out** |

The training env only ever samples the *train* scenarios; the held-out
scenarios are used only in `ml/evaluate_multiuav.py`. This is the setup for
a generalization test — see §6 for whether it currently generalizes (it
does not, and the reason is geometric, not a training bug).

---

## 4. Co-channel SINR assumptions

For `K` UAVs and a user served by UAV `s` (chosen as `argmax` received
power):

```
P_k    = ray-traced received power from UAV k  (linear, mW)
         = ( sum_j |a_{k,j}|^2 ) * tx_power_watts * 1000
N      = thermal noise power (from ConnectivityConfig.noise_power_dbm, default -94 dBm)

SNR_s  = 10 * log10( P_s / N )                         # interference-free
SINR_s = 10 * log10( P_s / ( N + sum_{k != s} P_k ) )  # co_channel  (default)
SINR_s = SNR_s                                          # orthogonal
```

**Assumptions for `co_channel`:** all UAVs transmit on the **same channel**,
**full buffer** (always transmitting), **no scheduling / no coordination**.
This is a deliberately pessimistic upper bound on self-interference. It is
computed from the *real* ray-traced per-UAV powers — nothing is fabricated.

**For `K == 1`**: `sum_{k != s} P_k = 0`, so `SINR = SNR` exactly. Verified
by `test_1uav_sinr_equals_snr`.

Both `SNR` and `SINR` are stored per user (`snr_db` / `sinr_db`,
`best_snr_db` / `best_sinr_db`) and aggregated (`mean_best_snr_db` /
`mean_best_sinr_db`), and the dashboard shows both side by side.

Observed magnitude: for 3 co-located-ish 30 dBm UAVs with near-LOS to
ground users, mean SNR ≈ 50–53 dB but mean co-channel SINR ≈ 0–3 dB. That
large gap is the modelled self-interference; it is why the PPO reward
includes a UAV-overlap penalty.

---

## 5. Sionna vs NS-3

They are **different models of different things** and are kept visually and
textually separate everywhere:

| | Sionna RT | NS-3 |
|---|---|---|
| What | electromagnetic propagation (ray tracing) | network / MAC / traffic |
| Channel | Mitsuba shoot-and-bounce over the 3D mesh | `YansWifiChannelHelper::Default()` (log-distance / Friis) |
| Output | per-user × per-UAV RSS / SNR / SINR / path gain / delay / paths, connectivity | per-flow packet delivery, throughput, delay, flow-failure rate |

**Whether NS-3 consumes Sionna's channel loss depends on the path.**
In the one-shot pipeline (`scripts/run_full_pipeline.py`,
`ml/evaluate_multiuav.py`) it does not: only the *geometry* (UAV positions,
user positions, serving-UAV association) is passed and NS-3 uses its own
distance-based channel. In the obstruction-aware paths
(`ml/closed_loop_control.py` with `channel_mode: "obstruction"`,
`ml/evaluate_obstruction.py`, `ml/gen_ns3_3d_data.py`) the handoff also
carries `obstruction_loss_db`: Sionna's per-link loss beyond free space at the
NS-3 operating frequency (2.412 GHz, verified with `--printChannel`), which
`ns3/uav-demo.cc` adds on top of its log-distance loss for every UAV–user link.
These callers check that NS-3 printed "Obstruction loss:" (that is, applied the
matrix) and fail otherwise. In every path a user can be Sionna-connected but get
no NS-3 delivery (MAC contention on the shared channel) or vice versa; the
dashboard shows the two states separately. See `docs/ppo3d.md` for the 3D PPO
closed loop.

**Association**: NS-3 uses the `associations` array from
`data/uav_positions.json` (best-RSS server chosen by Sionna). Nearest-UAV
by 3-D Euclidean distance is used **only as a fallback** when that array is
absent or the wrong length (`ns3/uav-demo.cc`:
`assocFromFile = (assoc.size() == nUser)`;
`ml/multiuav_io.py`: `if assoc is None: assoc = nearest_uav_association(...)`).
The end-to-end runs use "from handoff (best-RSS server)".

---

## 6. Current PPO result — honest

**The multi-UAV pipeline works end to end. The trained PPO policy does not
beat fixed placement, and no generalization is claimed.**

`ml/evaluate_multiuav.py` compares three placements (same scenario, users,
seed):

| Scenario | Split | RF | fixed | random | PPO | PPO − fixed |
|---|---|---|---|---|---|---|
| dense_urban | train | surrogate | 100 % cov, 30/30 | 100 %, 30/30 | 100 %, 30/30, move 0 m | **+0.0** |
| blocked_urban | held out | surrogate | 100 %, 30/30 | 100 %, 30/30 | 100 %, 30/30, move 0 m | **+0.0** |
| wide_area | held out | surrogate | 100 %, 30/30 | 100 %, 30/30 | 100 %, 30/30, move 0 m | **+0.0** |
| blocked_urban | held out | **Sionna** | 100 %, 20/20 | 100 %, 20/20 | 100 %, 20/20, move ~7 m (SINR slightly worse) | **+0.0** |

**Why PPO learns to stay (this is correct behaviour, not a bug):**

1. The UAV operating region (`uav_region`, roughly ±50–80 m around the mesh
   origin) sits in the **relatively open part of the `city_damaged` mesh**.
   Within ±200 m of the origin the mesh has only a few hundred triangles;
   the dense modelled buildings are 400–800 m away (see
   `dashboard/assets/scene_meta.json::geometry_honesty_note`).
2. From the deterministic **spatially-separated** `initial_uav_positions()`,
   3 UAVs at ~30 m altitude already have near-line-of-sight to every ground
   user — in the free-space surrogate **and** in real Sionna ray tracing.
   Fixed placement already achieves 100 % coverage and 100 % connectivity.
3. Coverage and connectivity therefore cannot improve, so the only active
   reward term left is the **movement penalty** (`- w_move * moved_frac`).
   Staying put is optimal, and the policy learned exactly that.

We did **not** modify the reward, the scenarios, the initial positions, or
the policy to force movement or manufacture a favourable comparison.

### What would be needed to demonstrate a real repositioning benefit

A configuration where **fixed placement leaves genuine coverage gaps**, e.g.:

* Put users out in the dense-building region 400–800 m from the origin.
  This requires widening `uav_region` in `configs/scenarios.json` **and**
  the 3D asset crop in `dashboard/build_assets.py`, and running the
  evaluation in Sionna over that area (slower).
* Many more users, spread far enough that 3 UAVs from separated fixed
  points cannot all reach them.
* Reduced per-UAV Tx power / higher connectivity thresholds so coverage
  holes appear.
* Moving obstructions / a scenario where the good positions change over the
  episode (the earthquake-style temporal datasets already in `sionna/`
  could seed this).

All of these are **scenario / geometry design**, not pipeline changes — the
infrastructure (Sionna multi-UAV eval, PPO env, reward weights, train/eval
split, baselines, NS-3, dashboard) is in place and parameterised for them.

---

## 7. Reward (documented, unchanged)

`ml/multiuav_env.py`, `DEFAULT_REWARD_WEIGHTS`:

```
reward =  0.35 * covered_fraction
        + 0.25 * connected_fraction
        + 0.10 * normalised mean serving RSS
        + 0.10 * Δ connected_fraction this step
        + 0.10 * load_balance          (1 - std(users_per_uav)/mean)
        - 0.15 * overlap_penalty       (grows as UAVs come within min_separation_m)
        - 0.05 * fraction of UAVs that moved
```

The movement term is a **penalty**. Nothing in the reward rewards movement.

---

## 8. Files

New: `configs/scenarios.json`, `configs/training.json`, `ml/scenarios.py`,
`ml/multiuav_io.py`, `ml/multiuav_env.py`, `ml/train_multiuav.py`,
`ml/evaluate_multiuav.py`, `ml/test_multiuav.py`, `docs/multiuav.md`,
`dashboard/frontend/src/multiuav.js` + 4 new dashboard components.

Modified (backward-compatible): `ml/run_ppo.py`, `ns3/uav-demo.cc`,
`scripts/run_full_pipeline.py`, `sionna/sionna_feedback_engine.py`
(+240 / −0, additive), `dashboard/backend/*`, several dashboard components,
`.gitignore`.

Not touched: `ml/dynamic_model.py`, `ml/train_dynamic*.py`,
`models/ppo_uav_dynamic.zip`, `models/dynamic_coverage_model.pkl`,
`sionna/scenes/*`, `sionna/sionna_pipeline.py`, `ResQNet_3D_Demo.ipynb`.

Git-ignored (regenerable): `models/ppo_multiuav/`, `data/experiments/`,
`data/runs/`, `dashboard/assets/*.glb`, `dashboard/frontend/node_modules/`,
`dashboard/frontend/dist/`.

---

## 9. Reproduce

```bash
source .venv/bin/activate
pip install -r dashboard/requirements.txt        # fastapi, uvicorn

# scenario check
python ml/scenarios.py

# train the multi-UAV policy (surrogate, ~5 min)
python ml/train_multiuav.py

# baselines: fixed vs random vs PPO  (held-out scenario)
python ml/evaluate_multiuav.py --scenario blocked_urban --num-uavs 3 --num-users 30 --mode surrogate --steps 12
NS3_PATH=~/ns-allinone-3.47/ns-3.47 python ml/evaluate_multiuav.py --scenario blocked_urban --num-uavs 3 --num-users 20 --mode sionna --steps 8 --ns3

# full end-to-end multi-UAV run
NS3_PATH=~/ns-allinone-3.47/ns-3.47 python scripts/run_full_pipeline.py \
    --num-uavs 3 --num-users 30 --scenario blocked_urban --mode sionna --steps 6

# tests (add RESQNET_SLOW_TESTS=1 for the Sionna + NS-3 tests)
RESQNET_SLOW_TESTS=1 python ml/test_multiuav.py

# 1-UAV pipeline still works, unchanged
python scripts/run_full_pipeline.py --mode sionna --steps 5
```
