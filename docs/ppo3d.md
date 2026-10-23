# 3D PPO positioning (per-UAV X, Y and Z)

Status: **implementation + pilot only.** No claim is made that PPO has learned
useful 3D positioning until the policy has been compared with the baselines in
real Sionna + ns-3 closed-loop runs on the held-out `urban_core` site (not run
yet; separate approval gate). Results of the pilot stages are in the
run directories listed at the end.

## 1. Pieces

| File | Role |
|---|---|
| `ml/multiuav3d_env.py` | Gymnasium env `MultiUAV3DEnv`: 7 actions per UAV, safety, reward, observation |
| `ml/train_multiuav3d.py` | MaskablePPO training; new run directory per run, never overwrites |
| `ml/eval_3d_validation.py` | offline comparison on the validation window (predicted metrics, model selection only) |
| `ml/radiomap_bench.py` | stage-0 cost benchmark of Sionna `RadioMapSolver` |
| `ml/scan_training_areas.py` | stage-1 height-map scan: dense training / validation windows away from `urban_core` |
| `ml/radiomap_cache.py` | stage-2 precompute (resumable) + `RadioMapRF` lookup + stage-3 accuracy check vs `PathSolver` |
| `ml/gen_ns3_3d_data.py` | stage-4 obstruction-aware ns-3 runs on the training / validation windows |
| `ml/latency_model.py`, `ml/fit_latency_model.py` | stage-5 latency predictor + pre-registered validation; PDR-model check under obstruction |
| `ml/ppo_controller.py` | PPO policy inside the closed loop (`closed_loop_control.py`, `"policy": "ppo"`) |
| `configs/training_areas_3d.json` | training windows `train_a/b/c`, validation window `val_a` |
| `configs/eval_val_a_3d.json`, `configs/closed_loop_ppo3d.json` | closed loop on the validation window |
| `ml/test_multiuav3d.py` | tests (env, safety, reward, radio-map lookup, closed-loop PPO, dashboard schema) |

The old 2D environment (`ml/multiuav_env.py`, `MultiDiscrete([5]*5)`), its
checkpoints and `ml/run_ppo.py` are unchanged. The 3D checkpoints are not
compatible with them and live in `models/ppo_multiuav_3d/<run_id>/`.

## 2. Action space and coordinates

`MultiDiscrete([7] * 5)`: one action per UAV slot (`MAX_UAVS = 5`; unused slots
are ignored and may only "stay"):

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| action | stay | −X | +X | −Y | +Y | down | up |
| move | – | −`step_h` | +`step_h` | −`step_h` | +`step_h` | −`step_v` | +`step_v` |

Coordinates are the Sionna scene's own (metres, z up), the same numbers ns-3
receives. Defaults `step_h_m = 15`, `step_v_m = 10` (as the rule-based
controller); both are constructor arguments. Every UAV chooses its own action,
so UAVs can be at different altitudes. Nothing forces them to be.

## 3. Safety

`closed_loop_control.safety_violations` (the rule-based controller's own rule,
imported unchanged) at the destination of every move:

* inside the scenario boundary (`uav_region` x / y);
* altitude in [max(15, region z min), min(60, region z max)] m;
* ≥ 5 m above the highest roof within 10 m, at the destination **and** at the
  midpoint of the move;
* ≥ 15 m from every other UAV.

Unsafe single moves are masked (`env.action_masks()`, a `[5 × 7]` boolean
vector; "stay" is always valid). If an unmasked policy chooses one anyway (plain
PPO fallback), it becomes "stay" and is counted (`unsafe_attempts`). Simultaneous
moves are then checked jointly: if two UAVs would end up, or pass at their move
midpoints, closer than 15 m, the higher-index UAV's move is cancelled
(deterministic; counted as `separation_cancels`).

During training, roof heights come from a 5 m height map of the scene
(`HeightmapGeometry`). In the closed loop every move is re-checked with ray
casts on the real scene (`SceneGeometry`).

## 4. Reward

All terms are in [0, 1]; weights are in `DEFAULT_REWARD_WEIGHTS` and can be
overridden with `--reward-weights`:

```
score = 0.35·mean PDR + 0.20·coverage(PDR ≥ 0.5) + 0.25·worst-10 % PDR
r     = score + 0.10·(score − previous score)
            − 0.03·metres moved / (active UAVs · max step)
            − 0.05·(unsafe attempts + separation cancels) / active UAVs
```

* No term depends on altitude (test `test_reward_has_no_altitude_preference`).
  Altitude matters only through its effect on the network.
* PDR is **predicted** in training: Sionna radio-map path gain → obstruction
  loss beyond free space → ns-3 link budget → per-user PDR from the
  ns-3-calibrated model. It is not ns-3-measured PDR. In the closed loop the
  same observation slots are filled with ns-3 **measurements**.
* Throughput under the fixed offered load is offered load × PDR, so it would
  duplicate PDR. It is logged, not rewarded.
* Latency is **not** in the reward or the observation unless the latency
  predictor passes its pre-registered validation (section 7).

## 5. Observation (`OBS_VERSION = "3d-v1"`, 163 floats)

Per UAV slot (17): active flag; x, y, z normalised to the region; clearance
above the highest roof within 10 m; 6 move-blocked flags; share of users
served; mean and worst PDR of its users; mean obstruction excess of its links
and share of links above 10 dB; distance to the nearest other UAV.
Global (6): mean PDR, coverage, worst-10 % PDR, users / 50, UAVs / 5,
step / max steps. Grids (2 × 6 × 6): user density and density of users below
the coverage threshold.

## 6. Data separation

* `urban_core` (configs/eval_urban_core.json) is the held-out final-evaluation
  site. The environment refuses it unless `allow_held_out=True` (evaluation
  only); the radio-map, ns-3 data and latency scripts refuse it outright. Its
  cached runs (`~/.cache/resqnet_obstruction/main`) and the rule-based
  seed-7000 runs are never read by any 3D script.
* Training windows `train_a`, `train_b`, `train_c` and validation window
  `val_a` are the densest non-overlapping 200 m windows whose edge is ≥ 100 m
  from `urban_core` (actual gaps ≥ 289 m). Their built fraction (0.70–0.74)
  matches the figure recorded for `urban_core` (0.72).
* Model selection, reward variants and the latency / PDR model choice use
  `val_a` only.

## 7. Latency predictor (pre-registered criteria)

Fitted on the stage-4 ns-3 runs of the training windows, tested on `val_a`.
It may enter the PPO observation / reward only if, on `val_a`:

* Spearman ρ ≥ 0.6 between predicted and measured per-UAV mean latency, and
* ≥ 70 % correct sign of the latency change for one-move placement pairs.

The calibrated PDR model (`models/network_surrogate_v1`, fitted on open,
distance-only scenes) is checked on the same data: placement-mean PDR
MAE ≤ 0.05 and Spearman ρ ≥ 0.6 on `val_a`. If v1 fails, a v2 with the same
features and monotone constraints is fitted on the training windows and used
only if it passes the same test. Results: `models/latency_surrogate_v1/<run>/report.json`.

## 8. Closed loop (`"policy": "ppo"`)

```
measure (ns-3, obstruction applied) ─► observation (measured per-user PDR,
    Sionna path gain / obstruction at the current positions, xyz, masks)
  ─► MaskablePPO action per UAV ─► safety filter (masks + joint separation,
    real ray-cast geometry) ─► move ─► Sionna RT at the ns-3 frequency
  ─► obstruction matrix in the ns-3 handoff (applied, verified) ─► ns-3
  ─► measured per-user PDR / latency ─► next observation
```

The default is still `"policy": "rule_based"` (configs/closed_loop_control.json
is unchanged). PPO moves are executed as chosen; the rule-based accept / revert
step is not applied. `state.json` records the policy and checkpoint (path and
sha256) under `provenance.policy`, each decision's policy action, executed
action, from / to xyz, masked actions and safety interventions, and per-interval
`history[].actions`. The dashboard's control tab shows them ("Policy, positions
and actions").

Run on the validation window:

```
python ml/closed_loop_control.py --config configs/closed_loop_ppo3d.json \
  --override '{"ppo": {"checkpoint": "models/ppo_multiuav_3d/<run>/best_val_model.zip"}}' \
  --out ~/.cache/resqnet_control/ppo3d_val_<id>
```

## 9. Commands

```
python ml/radiomap_bench.py --samples 10000000 --num-tx 1 --out ~/.cache/resqnet_3d/bench
python ml/scan_training_areas.py --out ~/.cache/resqnet_3d/scan
python ml/radiomap_cache.py precompute --window train_a --samples 10000000
python ml/radiomap_cache.py check --window train_a --links 300
python ml/gen_ns3_3d_data.py --out ~/.cache/resqnet_3d/ns3_data --layouts 40
python ml/fit_latency_model.py --out models/latency_surrogate_v1/<run>
python ml/train_multiuav3d.py --smoke --out-root ~/.cache/resqnet_3d/smoke
python ml/train_multiuav3d.py --train-windows train_a --timesteps 100000 --run-id pilot_<id>
python ml/eval_3d_validation.py --checkpoint models/ppo_multiuav_3d/<run>/model.zip \
    --out models/ppo_multiuav_3d/<run>/validation_offline.json
python -m unittest ml/test_multiuav3d.py
```

## 10. Measured results so far (2026-10-17)

| Stage | Result |
|---|---|
| 0 benchmark (200 m window, 2 m cells, 1e7 samples) | 1.7 s / map, ~0.4 GB; median 0.75 dB vs PathSolver, 0/30 usable flips |
| 1 scan | 8 s, 0.4 GB; windows in `configs/training_areas_3d.json` |
| 2 precompute | train_a 979 positions / 98 shards, 2851 s, 0.31 GB; val_a 95 shards, 3084 s, 0.31 GB |
| 3 accuracy (300 links each, PathSolver at exact user points) | train_a median 0.09 dB, p90 1.36 dB, flips 1.7 % → **pass**; val_a median 0.14 dB, p90 2.70 dB, flips 0.3 % → **pass** |
| 4 ns-3 data | 320 / 320 runs (80 per window: 40 base + 40 one-move neighbours); the first 16 from the interrupted run were reused. Resume: 304 runs, 4113 s with 1 worker, peak 1.03 GB. `obstruction_applied: true` in all 320 |
| 5 latency (val_a) | per-UAV Spearman ρ 0.87 (user-count baseline 0.29), pair-sign 72.5 % (29 / 40) → **passes** the pre-registered gate; median abs log error 0.30 |
| 5 PDR v1 (val_a) | MAE 0.105 (bias −0.105: under-predicts PDR under obstruction), ρ 0.96 → **fails** MAE ≤ 0.05 |
| 5 PDR v2 (fitted on train_a/b/c, monotone) | val_a MAE 0.012, ρ 0.99, pair-sign 78 % (23 pairs) → **passes**; used for the pilot |

Stage 5 output: `models/latency_surrogate_v1/fit_20261017/` (`report.json`, `model.joblib`, `pdr_v2.joblib`); 4.7 s, 0.18 GB.

Latency decision: it passed, but the pair-sign margin is small (72.5 % on 40 pairs, binomial 95 % CI ≈ 56–85 %) and adding it changes the observation version. The pilot keeps the documented reward (no latency term, `latency_in_reward: false`); a latency term is a candidate for the next approved stage, not part of this pilot.
| G pilot a | `pilot_20261017_a`: stopped at ~5 min by host memory pressure, only `train_config.json`; kept, not reused |
| G pilot b | `pilot_20261017_b`: train_a, PDR v2, default reward; 100 352 steps, 620 s, 162 steps/s, 1.04 GB; checkpoints every 25k; selected on val_a |
| G variant c | `pilot_20261017_c`: as b but movement weight 0.03 → 0.10; 649 s; worse on val_a, not selected |

### Offline comparison (predicted metrics, training model, 30 episodes, identical starts / users)

val_a (validation window; model selection):

| method | score | mean PDR | coverage | worst-10 % | horiz. m | vert. m | distinct alt. |
|---|---|---|---|---|---|---|---|
| fixed | 0.475 | 0.718 | 0.736 | 0.307 | 0 | 0 | 1.43 |
| random actions | 0.473 | 0.714 | 0.732 | 0.306 | 301 | 83 | 2.23 |
| random placement | 0.463 | 0.689 | 0.701 | 0.325 | 0 | 0 | 2.37 |
| greedy one-step (model) | **0.720** | 0.961 | 0.974 | 0.754 | 27 | 4 | 1.70 |
| PPO b | 0.445 | 0.624 | 0.637 | 0.398 | 299 | 29 | 2.33 |
| PPO b, up/down masked | 0.344 | 0.490 | 0.499 | 0.288 | 368 | 0 | 1.43 |
| PPO c | 0.391 | 0.537 | 0.545 | 0.375 | 251 | 24 | 2.07 |

train_a (diagnostic only, never used for selection; `diagnostic_train_a.json`):

| method | score | horiz. m |
|---|---|---|
| fixed | 0.646 | 0 |
| random actions | 0.667 | 302 |
| greedy one-step | 0.730 | 9 |
| PPO b | **0.778** | 137 |
| PPO b, up/down masked | 0.777 | 146 |

Conclusion: PPO learns the training window (beats greedy there) but **does not
generalise** to val_a, where it is below fixed placement and far below the
greedy baseline. With one training window it learns window-specific moves
(action histogram dominated by +x / +y) and wanders ~300 m on val_a. The
pilot therefore does **not** show useful 3D positioning. Altitude actions
matter on val_a (masking them drops the score 0.445 → 0.344) but not on train_a.
Next step: train on several windows (train_b / train_c radio maps needed).
No safety violations in any method.

### Real closed-loop episode on val_a (Sionna RT → obstruction matrix → ns-3, measured)

Same config (`configs/closed_loop_ppo3d.json`: 3 UAVs, 30 users, user seed 8100,
start at 40 m), one episode each, one ns-3 RNG run per interval — indicative only.

| | PPO b (`best_val_model.zip`, sha256 8897232…) | rule-based |
|---|---|---|
| output | `~/.cache/resqnet_control/ppo3d_val_20261017_b` | `~/.cache/resqnet_control/rule_val_20261017` |
| intervals / moves | 8 / 13 UAV moves (x, y and z; final z 30, 50, 60 m) | 3 / 1 move (15 m) |
| measured PDR before → after | 0.965 → 0.985 | 0.965 → 0.995 |
| worst-10 % PDR | 0.656 → 0.886 | 0.656 → 0.966 |
| mean latency | 25.6 → 56.2 ms | 25.6 → 21.7 ms |
| controller objective | 0.675 → 0.666 | 0.675 → **0.750** |
| wall time / peak memory | 147 s / 1.32 GB | 63 s / 1.48 GB |

Integration checks: 7 Sionna and 7 ns-3 calls (start + 6 intervals with moves;
unchanged positions are not re-measured); every ns-3 run reported the
obstruction matrix applied (the controller raises otherwise); ns-3's measured
per-user PDR is the PDR input of the next PPO observation
(`ppo_controller.select` → `env.load_state`, test `TestClosedLoopPPO`);
`state.json` records policy, checkpoint sha256, per-UAV policy action, masked
actions and xyz before / after. Safety masks were active (e.g. "masked as
unsafe: up" at the 60 m ceiling).

Accuracy files: `~/.cache/resqnet_3d/radiomaps/<window>/accuracy_v1_s10000000_d3_c2_seed0.json`.

### Tests (2026-10-17, after all heavy jobs)

| suite | result |
|---|---|
| `test_multiuav3d` | 31 / 31 OK |
| `test_closed_loop_control` | 26 / 26 OK |
| `test_closed_loop` | 27 / 27 OK |
| `test_network_reward` | 17 / 17 OK |
| `test_multiuav` | 52 run, OK, 8 skipped (slow tests, need `RESQNET_SLOW_TESTS=1`) |
| `test_reward_verification` | 4 / 4 OK |
| `test_obstruction` | 22 run, 1 failure → fixed → isolation class rerun OK (349 s, 0.63 GB) |
| frontend `npm test` | 8 pass, 1 skipped (backend-fixture test), 0 fail; `vite build` OK |

The `test_obstruction` failure was real: `test_urban_core_not_in_existing_configs`
forbids the string "urban_core" in any `configs/training*.json`, and
`configs/training_areas_3d.json` named it in its comment, its `held_out` field
and the `gap_to_urban_core_m` keys. The config was reworded (backup in
`~/.cache/resqnet_3d/backups/pre_edit/`); window coordinates are unchanged, so
the radio-map caches stay valid. The test was not changed.

### Next approval gate (measured costs)

1. Radio maps for `train_b` and `train_c` (needed to train on more than one
   window, the fix the pilot points to): ~50 min each, 0.31 GB, run one at a time.
2. Multi-window training (`--train-windows train_a train_b train_c`), measured
   ~160 steps/s and 1.04 GB: 300k steps ≈ 35 min, 1M ≈ 1 h 50 min (+ validation).
   Selection on val_a (offline, then one real closed-loop episode, ~2.5 min).
3. Optional: a latency term (the predictor passed, narrowly), as a new
   observation version — a design change to approve separately.
4. Final held-out `urban_core` evaluation (~5–6 h) only after a policy beats the
   baselines on val_a.

## 11. Limitations

* The radio maps are a 2 m cell average at 1.5 m height. A user is looked up
  in the cell that contains it (no interpolation). Accuracy against
  `PathSolver` at the exact user point is measured in stage 3.
* Training PDR is a model prediction (PDR v2, val_a MAE 0.012 against ns-3 on
  80 placements). The closed loop replaces it with ns-3 measurements. The
  train / deploy gap is real and is reported, not hidden.
* Training on a window without a radio-map cache fails with a missing
  `meta.json` error (train_b / train_c are not built yet).
* `progress.csv` / `val_progress.csv` are written during training (added after
  the interrupted run); runs before that only have the end-of-run logs.
* UAV positions in training are lattice points (15 m / 10 m, z ∈ {20…60} m),
  because the radio maps are precomputed there. The closed loop can start
  off-lattice (start altitudes are raised to the safety floor).
* Training safety uses a 5 m height map. The closed loop uses exact ray casts,
  so a move the policy expects to be allowed can be masked there.

## 12. Multi-area run (2026-10-18)

Approved scope: radio maps for `train_b` and `train_c`, one 300k-step
multi-area PPO run, evaluation on `val_a`, regression tests. Not in scope (not
run): 1M-step training, any other radio map, latency reward term, `urban_core`.

### 12.1 Preflight (read-only)

* Branch `feat/multi-uav-pipeline`, HEAD `ae3a508`, stash `stash@{0}` ("my local
  PPO and notebook changes") unchanged, 37 status entries (10 tracked
  modifications, the rest untracked), matching the 2026-10-17 state.
* No leftover Python / ns-3 processes. WSL 11.3 GB available; Windows 2.4 GB free.
* train_a / val_a caches complete (98 / 95 shards, no tmp files) with passing
  accuracy files; train_b / train_c absent. ns-3 data 4 × 80 runs. Surrogates in
  `models/latency_surrogate_v1/fit_20261017/` (latency pass, PDR v1 fail, v2 pass).
* `configs/training_areas_3d.json`: train_a/b/c role `train`, val_a role
  `validation`; `urban_core` not present. Scene `sionna/scenes/city_damaged.xml`
  sha256 `55264330…`.

### 12.2 Radio maps train_b, train_c (same pipeline and settings as train_a)

Command (one area at a time, then the stage-3 check):

```
python ml/radiomap_cache.py precompute --window train_b --samples 10000000
python ml/radiomap_cache.py check --window train_b --links 300
# then the same for train_c
```

| | train_b | train_c |
|---|---|---|
| safe lattice positions / shards | 966 (14 unsafe skipped) / 97 | 977 (3 skipped) / 98 |
| wall time (measured) | 1642 s | 1355 s |
| peak RSS (measured, `/usr/bin/time`) | 0.32 GB | 0.32 GB |
| Windows free during run (sampled) | 2.0–3.2 GB | 3.4–3.9 GB |
| accuracy: median / p90 abs error | 0.08 / 2.39 dB | 0.11 / 1.70 dB |
| usable flips (limit 5 %) | 1.0 % | 1.0 % |
| accuracy check | **pass** (46 s, 0.50 GB) | **pass** (63 s, 0.50 GB) |
| integrity | 966 unique positions, z {20…60}, no NaN, path gain −139…−63 dB, all inside region, meta region / scene sha / 2.412 GHz match config, loads as `RadioMapRF`, no tmp files | same (977 positions, −133…−64 dB) |

Output: `~/.cache/resqnet_3d/radiomaps/{train_b,train_c}/v1_s10000000_d3_c2/`,
accuracy files next to them, logs `~/.cache/resqnet_3d/logs/{precompute,check,mem_precompute}_train_{b,c}.log`.

### 12.3 Multi-area training — interrupted by host memory pressure

Pre-training checks (all passed, `scratchpad/envcheck.sh`): train env windows
{train_a, train_b, train_c} each with its own radio-map cache, val env {val_a}
only, `urban_core` absent; PDR model `calibrated:…/fit_20261017/pdr_v2.joblib`;
action space `MultiDiscrete([7]*5)`, obs 163, 12 steps/episode, steps 15 m / 10 m;
default reward weights, no latency term; 300 resets drew train_a/b/c 96/97/107
times; ±x, ±y, up / down move exactly the intended axis in every area.

Change before the run (metadata only, no behaviour change):
`train_config.json` now also records `area_sampling` and `provenance` (git HEAD,
sha256 of the env / training / cache / controller sources and the area config,
the PDR-model and height-map sha256, radio-map cache paths and `meta.json` sha256).

Command (same settings as `pilot_20261017_b` except windows and steps):

```
python ml/train_multiuav3d.py --train-windows train_a train_b train_c --val-windows val_a \
  --timesteps 300000 --seed 0 --pdr-model-path models/latency_surrogate_v1/fit_20261017/pdr_v2.joblib \
  --run-id multi_20261018_a
```

Result: **stopped by Claude Code's low-memory reaper after ~2 min.** Saved:
`models/ppo_multiuav_3d/multi_20261018_a/{train_config.json, progress.csv}`
only. Last logged step 6 144 / 300 000; **no checkpoint** (first due at 25 000);
no validation yet. Measured: job RSS 1.33 GB (pilot: 1.04 GB — four training envs
plus the validation env each hold the radio maps of their windows), 59 steps/s
over the first 6k steps (pilot: ~160 steps/s; the cause is not established),
Windows free memory 2.95 GB before start, 1.68 GB at the stop. The WSL VM itself
was 1.29 GB; the rest of the 15.6 GB was held by other Windows applications
(browser processes ~2.5 GB in the top 8). The run was not restarted (stop rule).
The run directory is kept and must not be reused.

Not done because of the stop: evaluation (stage 4) and regression tests (stage 5).

### 12.4 State and next decision (2026-10-18)

Git: branch `feat/multi-uav-pipeline`, HEAD `ae3a508`, stash unchanged, 37
status entries (same 10 tracked modifications; new outputs are untracked or
under `~/.cache`). No commit / push / reset / clean.

Completed: train_b and train_c radio maps (built, accuracy pass, integrity
pass). Remaining: the 300k multi-area run, its val_a evaluation (offline
30-episode comparison with fixed / random / greedy, checkpoints 100k / 200k / 300k,
and seeded real closed-loop episodes vs rule-based), regression tests.

Options for the restart (needs the user's decision):

1. Free host memory (close browser windows) and rerun the same command with a
   new run id (e.g. `multi_20261018_b`). Measured: 1.33 GB job RSS; at the
   observed 59 steps/s ≈ 85 min for 300k steps (+ validation every 20k).
2. Memory reduction in code (approval needed, no change to the experiment):
   share one `RadioMapRF` per window across the training / validation envs
   instead of one per env — expected to cut the job's RSS by several hundred MB
   (estimate, not measured).
3. `--n-envs 2` instead of 4 also lowers memory but changes the PPO rollout
   size (512 vs 1024 steps per update) — a hyperparameter deviation from the
   pilot, so not applied without approval.
4. Start Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` (the
   reaper then no longer stops jobs; risk of a real out-of-memory instead).

### 12.5 Option 2: shared radio maps (2026-10-18)

Measured map size: 39 MB per window in memory (float32, [~970 positions, 100, 100]).
Before: each of the 4 train envs loaded train_a/b/c and the val env val_a, 13 copies, ~506 MB; this matches
the multi-area vs pilot RSS gap (1.33 vs 1.04 GB). `RadioMapRF` is a pure lookup (no writes after `__init__`,
`path_gain` returns a copy, no RNG; each env keeps its own `_rng`), so sharing cannot change observations,
rewards, randomisation or episodes.

Change (`ml/train_multiuav3d.py` only): `make_env` keeps one `MultiWindowRF` per (window set, samples) in
`_RF_SHARED`, so the 4 train envs share train_a/b/c and the val env keeps val_a; map / position arrays are set
read-only; `train_config.json` records `radiomap_sharing`. Algorithm, windows, reward, seed, n_envs 4,
n_steps 256, batch 128, checkpoints every 25k and 300k target unchanged. Only differences: the source sha256 of
`train_multiuav3d.py` in provenance, and the extra config key.
Expected job RSS ~0.98 GB (1.33 - 0.35; estimate, not measured).

Tests: `python -m unittest ml/test_multiuav3d.py` 31/31 OK, 6.0 s, peak RSS 0.83 GB.

Blocked before launch: Windows free memory 1.4-1.7 GB at the start and 0.87-0.99 GB after the tests (WSL
holds ~0.9 GB page cache; browser and other apps hold the rest), below the ~1.7 GB level where the reaper
stopped `multi_20261018_a`. `multi_20261018_b` not started; equivalence check (shared vs separate maps, same
seeds, identical obs / rewards) not yet run. Needed before launch: roughly 3 GB or more Windows free memory.

### 12.6 `multi_20261018_b`: completed run and results (2026-10-18)

Supersedes the open decision in 12.4 / 12.5: the run was restarted on a different host (Windows 11, RTX GPU for
Sionna, data on `D:`), so the memory limit that stopped `multi_20261018_a` no longer applied.

**Setup on this host.**
- Windows Python 3.12 runs the radio maps, training and offline evaluation. `ml/utils.py:peak_rss_mb()` replaces
  `resource.getrusage` (Linux-only) in `radiomap_cache.py`, `radiomap_bench.py` and `train_multiuav3d.py`: psutil's
  peak working set on Windows, unchanged on Linux. `sb3-contrib` and `scikit-learn==1.9.0` are needed (`pdr_v2.joblib`
  was pickled with 1.9.0).
- The closed loop (Sionna + ns-3) runs in WSL Ubuntu: ns-3.47 at `~/ns-allinone-3.47/ns-3.47` (optimized build),
  venv `~/resqnet-venv` with the Windows package versions. OptiX cannot start in WSL (its `libnvoptix.so.1` has no
  `optixQueryFunctionTable`), so Sionna runs on Dr.Jit's CPU backend: `DRJIT_LIBLLVM_PATH` points at a user-space
  libLLVM 18 (micromamba, conda-forge `libllvm18`), `CUDA_VISIBLE_DEVICES=` is empty.
- All four radio maps were rebuilt here (GPU, ~2 min each). Accuracy matches 10.: train_a 0.09 / 1.36 dB / 1.7 %,
  val_a 0.14 / 2.75 dB / 0.3 %, train_b 0.085 / 2.39 dB / 1.0 %, train_c 0.11 / 1.70 dB / 1.0 % (median / p90 /
  usable flips), all **pass**. Tests: `ml/test_multiuav3d.py` 31/31 OK on Windows.
- Reproduction check: the rule-based closed loop on val_a gives exactly the numbers in 10. (0.675 → 0.750).

**Training.** Same command as 12.3 with `--run-id multi_20261018_b` (train_a/b/c, val_a, 300k steps, seed 0,
PDR v2, shared radio maps from 12.5). 300 032 steps in 5570 s (54 steps/s), ~0.93 GB. Validation score
(every 20k steps) rises from 0.276 to its best **0.693 at 160k** (`best_val_model.zip`, sha256 `66368d0eceee8f45…`)
and then stays at 0.61–0.69. Action histogram: stay 47 %, +x 17 %, up 12 %, −y 10 %, +y 10 %, −x 2 %, down 2 %.

**Offline comparison on val_a** (same protocol as 10., 30 episodes; val_a is also the selection window, so the
checkpoint row is optimistic — `urban_core` is the unbiased test and was not run):

| method | score | mean PDR | coverage | worst-10 % | horiz. m | vert. m |
|---|---|---|---|---|---|---|
| fixed | 0.475 | 0.718 | 0.736 | 0.307 | 0 | 0 |
| random actions | 0.473 | 0.714 | 0.732 | 0.306 | 301 | 83 |
| random placement | 0.463 | 0.689 | 0.701 | 0.325 | 0 | 0 |
| greedy one-step (model) | 0.720 | **0.961** | **0.974** | 0.754 | 27 | 4 |
| pilot PPO b (one window) | 0.445 | 0.624 | 0.637 | 0.398 | 299 | 29 |
| **multi PPO b, best (160k)** | **0.742** | 0.951 | 0.962 | **0.867** | 211 | 22 |
| multi PPO b, best, up/down masked | 0.738 | 0.950 | 0.961 | 0.853 | 224 | 0 |

Checkpoints (`validation_offline_ckpt*.json`): 100k 0.714 (masked z 0.686), 200k 0.741 (0.753), 300k 0.746 (0.753).
Training on three windows fixes the pilot's failure to generalise: PPO now beats greedy on score, mostly through
the worst-served users (worst-10 % 0.867 vs 0.754), at slightly lower mean PDR / coverage. Altitude control adds
little: from 200k on, masking up/down is as good or better. No safety violations.

**Real closed-loop episode on val_a** (`configs/closed_loop_ppo3d.json`, one episode each, one ns-3 run per
interval — indicative only):

| | multi PPO b (`best_val_model.zip`) | rule-based |
|---|---|---|
| output (WSL) | `~/.cache/resqnet_control/ppo3d_val_20261018_b` | `~/.cache/resqnet_control/rule_val_20261018` |
| intervals / UAV moves | 7 with moves / 17 (final z 40, 60, 60 m) | 3 / 1 (15 m) |
| coverage | 0.967 → **1.000** | 0.967 → 1.000 |
| mean PDR | 0.965 → 0.955 | 0.965 → **0.995** |
| worst-10 % PDR | 0.656 → 0.743 | 0.656 → **0.966** |
| mean latency | 25.6 → 77.1 ms | 25.6 → **21.7** ms |
| controller objective | 0.675 → 0.594 (best 0.718 after step 1) | 0.675 → **0.750** |
| Sionna / ns-3 calls, time | 8 / 8, 45 s / 80 s | 3 / 2, 42 s / 24 s |

Final PPO positions: `data/ppo3d_val_a_positions.json`.

Conclusion: offline (training model, no latency term) PPO is now the best method on val_a, but in the measured loop
it is **worse than the rule-based controller**. Two measured causes: (1) the controller objective weights latency
0.2 and PPO's reward has no latency term (`latency_in_reward: false`, 7.), so PPO climbs to 60 m and spreads the
UAVs, and latency rises 25.6 → 77 ms (the pilot did the same, 25.6 → 56 ms); (2) PPO keeps moving after its best
measured point (0.718 after the first move) until the safety masks stop it at the window edge / ceiling. Next steps
that do not change this run: add the latency predictor (passed its gate, 7.) to the reward, or let the closed loop
keep the best measured placement; then run `urban_core` once.

**Caveat (ns-3 channel, unchanged here).** `uav-demo.cc` chains `MatrixPropagationLossModel` after
`LogDistancePropagationLossModel` (exponent 3, 46.6777 dB at 1 m). The matrix holds Sionna's loss *in excess of
free space*, but the LogDistance base is itself above free space by 6.6 + 10·log10(d) dB at 2.412 GHz
(≈ 24–27 dB at 50–100 m), so obstructed links are penalised by both. Kept as is for this run because the
user-to-user hidden-node behaviour and every number in 10. use the same channel; fixing it (free-space base, or
Sionna's total loss instead of the excess) needs new ns-3 data and a refit of the PDR model.

**Sionna paths for the PPO placement** (this run; the notebook now shows `multi_lat_20261018_c`, 12.7). `ResQNet_Sionna_Rays.ipynb` (helpers in `sionna/ray_viz.py`) traces
UAV → user paths in val_a with the pipeline's settings (max depth 3, line of sight, specular reflection,
refraction; diffraction and diffuse scattering off: one material, scattering coefficient 0) and draws the real
path vertices: an interactive 3D view with reflection points, Sionna's own render (`data/sionna_rays_val_a.png`)
and a per-link table. With the PPO placement and 10 users: 79 paths (15 line of sight, 18 one reflection, 9 two or
three reflections, 37 through a wall), and every user has line of sight to its best UAV. `POSITIONS_FILE = None`
shows the hand-placed start formation instead (99 paths, 7 line of sight).

### 12.7 `multi_lat_20261018_c`: latency in the reward (2026-10-18)

Why: in 12.6 PPO lost the measured closed loop because the controller objective weights latency 0.2 and PPO's
reward had no latency term. The latency predictor passed its pre-registered gate (7.), so it now enters the reward.

**Change.**
- `ml/multiuav3d_env.py`:
  - `score = 0.35·PDR + 0.20·coverage + 0.25·worst-10 % − w_lat·latency score`, where latency score is the mean
    predicted per-user delay / 100 ms, capped at 3. That is the closed-loop objective's term
    (`closed_loop_control.network_metrics`), averaged over delivering users.
  - `w_lat` defaults to 0, so earlier runs and their evaluation are unchanged. Re-evaluating `multi_20261018_b`
    with the new code gives identical numbers (greedy 0.7196, PPO 0.742).
  - `CalibratedPDR(latency_path=...)` computes PDR and latency from the same features: the test checks it
    against `latency_model.LatencyModel` and the previous PDR path.
  - The observation is unchanged (`3d-v1`), so old and new checkpoints run in the same closed loop.
- `ml/train_multiuav3d.py`:
  - new option `--latency-model-path`;
  - `train_config.json` records `latency_in_reward` and the latency-model sha256;
  - the validation log gains `mean_latency_ms`.
- `ml/eval_3d_validation.py`:
  - new options `--latency-model-path` and `--reward-weights`;
  - greedy now uses `env.score_of`, so with the latency term on it also optimises latency.
- Tests: `ml/test_multiuav3d.py` 35/35 on Windows (4 new: latency off by default, weight needs a predictor,
  term and cap, predictor consistency); `ml/test_closed_loop.py` passes in WSL (the 4 checkpointing tests need the
  Linux ns-3 binary; the 3 dashboard tests need pydantic, absent from the WSL venv).

**Predictor check on the measured loop.** At the exact positions and users of the 12.6 episodes (Sionna path
gains, Windows GPU), predicted vs measured mean latency: the PPO episode 32 → 63 ms vs 25.6 → 77 ms (the
predictor gets the direction right at every altitude step but understates the rise); the rule-based start → end
32.0 → 31.2 vs 25.6 → 21.7 ms.

**Training.** As 12.6 plus `--latency-model-path models/latency_surrogate_v1/fit_20261017/model.joblib
--reward-weights '{"latency": 0.2}'`. 300 032 steps, 8151 s (37 steps/s). Validation (latency-aware score) rises
to its best, 0.658, at the **last** checkpoint (300k), so the run is probably not converged. Action histogram:
stay 37 %, +x 23 %, −y 12 %, +y 11 %, up 8 %, down 6 %, −x 3 %.

**Offline, val_a, latency-aware score** (`validation_offline_latency.json` in both run directories):

| method | score | mean PDR | coverage | worst-10 % | pred. latency ms | horiz. m |
|---|---|---|---|---|---|---|
| fixed | 0.426 | 0.718 | 0.736 | 0.307 | 24.6 | 0 |
| random actions | 0.421 | 0.714 | 0.732 | 0.306 | 26.1 | 301 |
| greedy one-step (latency-aware) | **0.690** | **0.966** | **0.975** | 0.816 | **23.5** | 45 |
| PPO `multi_20261018_b` (no latency term) | 0.680 | 0.951 | 0.962 | **0.867** | 30.8 | 211 |
| PPO `multi_lat_20261018_c` | 0.674 | 0.950 | 0.963 | 0.846 | 31.4 | 257 |

Offline, the latency term does not change much: both PPO runs are slightly below the latency-aware greedy.

**Real closed-loop episode on val_a** (same config / users / start as 12.6; one episode, indicative only):

| | PPO `multi_lat_20261018_c` | PPO `multi_20261018_b` | rule-based |
|---|---|---|---|
| intervals with moves / UAV moves | 2 / 4 (final z 30, 40, 40 m) | 7 / 17 (z 40, 60, 60) | 1 / 1 |
| coverage | 0.967 → 1.000 | 0.967 → 1.000 | 0.967 → 1.000 |
| mean PDR | 0.965 → **0.995** | 0.965 → 0.955 | 0.965 → **0.995** |
| worst-10 % PDR | 0.656 → 0.957 | 0.656 → 0.743 | 0.656 → **0.966** |
| mean latency | 25.6 → 31.3 ms | 25.6 → 77.1 ms | 25.6 → **21.7** ms |
| controller objective | 0.675 → **0.729** | 0.675 → 0.594 | 0.675 → **0.750** |
| Sionna / ns-3 calls, time | 3 / 3, 12 s / 17 s | 8 / 8, 45 s / 80 s | 3 / 2, 42 s / 24 s |

Output: `~/.cache/resqnet_control/ppo3d_val_20261018_lat_c` (WSL). Final positions: `data/ppo3d_val_a_positions.json`.

Conclusion: with latency in the reward, PPO stops after two intervals instead of climbing to the 60 m ceiling,
and the measured objective goes from a loss (0.594) to a gain (0.729). PDR, coverage and worst-10 % now match the
rule-based controller; the remaining gap (0.729 vs 0.750) is latency, 31 vs 22 ms. One episode on one window, one
ns-3 run per interval: indicative, not a significance test. Not yet run: `urban_core` (the unbiased test),
more seeds / user layouts, and continued training (validation was still rising at 300k; `--resume-from
models/ppo_multiuav_3d/multi_lat_20261018_c/model.zip`).

`ResQNet_Sionna_Rays.ipynb` showed this placement until 12.8: 113 traced paths (7 line of sight, 21 one reflection,
31 two or three reflections, 54 through a wall). 5 of 10 users have no line of sight to their best UAV and are
reached through reflections or walls.

### 12.8 Continued training (`multi_lat_20261019_d`) and a 5-layout closed-loop comparison (2026-10-19)

**Training.** `--resume-from models/ppo_multiuav_3d/multi_lat_20261018_c/model.zip --timesteps 300000` with
the 12.7 settings: 300k → 600k steps, 7354 s (41 steps/s). Fix in `ml/train_multiuav3d.py` first:
- On resume, the validation, progress and steps/s counters now count from the resumed step. Before the fix, the
  validation schedule started at 20k, so resuming at 300k would have validated 15 times in a row.
- The resumed model is validated first, so `best_val_model.zip` is only replaced by a better checkpoint.
- Checked with a short resumed smoke run; `ml/test_multiuav3d.py` 35/35.

Validation (10 episodes, latency-aware): 0.656 at 300k, best **0.733 at 520k** (`best_val_model.zip`), 0.715 at
600k.

**Offline, val_a, latency-aware score, 30 episodes:**

| method | score | mean PDR | coverage | worst-10 % | pred. latency ms |
|---|---|---|---|---|---|
| greedy one-step (latency-aware) | 0.690 | 0.966 | 0.975 | 0.816 | **23.5** |
| PPO 300k (`multi_lat_20261018_c`) | 0.674 | 0.950 | 0.963 | 0.846 | 31.4 |
| **PPO 520k** (`multi_lat_20261019_d/best_val_model.zip`) | **0.712** | **0.973** | **0.985** | **0.928** | 29.1 |
| PPO 600k (`multi_lat_20261019_d/model.zip`) | 0.694 | 0.967 | 0.981 | 0.891 | 31.6 |

**Real closed loop, 5 user layouts.** Same config as 12.6 / 12.7 with `user_seed` 8100–8104, one episode per
method and layout, run in WSL as `python ml/closed_loop_control.py --config configs/closed_loop_ppo3d.json --override '{"user_seed": <seed>, "ppo": {"checkpoint": "<model>"}}'` (rule-based: `'{"policy": "rule_based", "user_seed": <seed>}'`). Output: `~/.cache/resqnet_control/multiseed_20261019/<method>_<seed>`.

Objective before → after (UAV moves):

| user seed | rule-based | PPO 300k | PPO 520k |
|---|---|---|---|
| 8100 | 0.675 → **0.750** (1) | 0.675 → 0.729 (4) | 0.675 → 0.682 (19) |
| 8101 | 0.678 → **0.723** (1) | 0.678 → 0.304 (34) | 0.678 → 0.609 (28) |
| 8102 | 0.563 → 0.651 (2) | 0.563 → 0.733 (10) | 0.563 → **0.743** (19) |
| 8103 | 0.633 → 0.633 (0) | 0.633 → 0.374 (33) | 0.633 → **0.727** (27) |
| 8104 | 0.473 → **0.756** (3) | 0.473 → 0.377 (29) | 0.473 → 0.683 (28) |

Mean of the final state over the 5 layouts (start: objective 0.605):

| | objective | coverage | PDR | worst-10 % | latency ms | wins vs rule | best objective during episode |
|---|---|---|---|---|---|---|---|
| rule-based | **0.703** | 0.987 | 0.979 | 0.836 | 31.0 | – | 0.703 |
| PPO 300k | 0.503 | 0.753 | 0.750 | 0.387 | 21.6 | 1 / 5 | 0.695 |
| **PPO 520k** | 0.689 | **0.993** | **0.982** | **0.864** | 41.5 | 2 / 5 | **0.734** |

Conclusion:
- **Model choice:** PPO 520k is the model to use. The 300k model's single good episode (12.7) was luck: on 3 of
  5 layouts it lost most of the coverage (0.30–0.38).
- **PPO 520k against rule-based:** essentially even, 0.689 vs 0.703 with 5 episodes, a difference well inside
  the per-layout spread.
  - PPO 520k beats the rule-based controller on 2 layouts and has better coverage, PDR and worst-10 %.
  - It pays in latency, 41.5 vs 31.0 ms.
  - It passes through better placements than it ends at (mean best 0.734 vs final 0.689): it keeps moving
    (about 24 moves per episode against 1–3).
  - The two remaining improvements are the 12.6 "keep the best measured placement" option and a stronger
    latency / movement penalty in training.
- Still one window (val_a), one ns-3 run per interval. `urban_core` not run.

`ResQNet_Sionna_Rays.ipynb` now shows PPO 520k's final placement for user seed 8100 (`data/ppo3d_val_a_positions.json`):
- 112 traced paths: 12 line of sight, 22 one reflection, 34 two or three reflections, 44 through a wall.
- Every user has line of sight to its best UAV.
