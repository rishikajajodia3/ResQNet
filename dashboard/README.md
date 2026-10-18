# ResQNet Dashboard

A read-only engineering/research dashboard over the **existing** ResQNet
pipeline:

```
3D disaster environment
  -> Sionna RT (propagation / radio analysis)
  -> wireless metrics (RSS / SINR / path gain / coverage)
  -> PPO agent
  -> UAV repositioning
  -> data/uav_positions.json
  -> NS-3 (network / MAC / traffic validation)
  -> throughput / delay / flow status
```

The dashboard **does not reimplement** PPO, Sionna, or NS-3. It reads the
real output files those tools already write, and the "Run ResQNet" button
shells out to the existing `scripts/run_full_pipeline.py`.

---

## Layout

```
dashboard/
  backend/
    app.py          FastAPI app + routes; run with `python dashboard/backend/app.py`
    config.py       paths, NS3_PATH resolution
    runners.py      background run manager (calls scripts/run_full_pipeline.py)
    schemas.py      pydantic models for shaped responses
  frontend/         Vite + React + @react-three/fiber + Recharts
  build_assets.py   OBJ -> cropped .glb + scene_meta.json (originals untouched)
  assets/           generated: city_damaged.glb, city_baseline.glb, scene_meta.json
  requirements.txt  fastapi + uvicorn (installed into the project .venv)
```

---

## One-time setup

```bash
cd ~/ResQNet
source .venv/bin/activate

# backend deps (into the existing venv)
pip install -r dashboard/requirements.txt

# web-ready 3D assets (writes dashboard/assets/, does not touch sionna/scenes/)
python dashboard/build_assets.py

# frontend deps
cd dashboard/frontend
npm install
```

---

## Running (two terminals)

**Terminal 1 - backend**

```bash
cd ~/ResQNet
source .venv/bin/activate
python dashboard/backend/app.py
# -> http://127.0.0.1:8000   (API docs at /docs)
```

**Terminal 2 - frontend**

```bash
cd ~/ResQNet/dashboard/frontend
npm run dev
# -> http://localhost:5173   (Vite proxies /api to the backend)
```

Open **http://localhost:5173**.

### Optional: single-process (built) mode

```bash
cd ~/ResQNet/dashboard/frontend && npm run build   # -> dashboard/frontend/dist
cd ~/ResQNet && source .venv/bin/activate && python dashboard/backend/app.py
# the backend now also serves the built UI at http://127.0.0.1:8000
```

### NS-3 path

The "Run ResQNet" button needs an NS-3 install. Resolution order:

1. `NS3_PATH` environment variable, else
2. `~/ns-allinone-3.47/ns-3.47` (default).

Before each run the backend copies the current `ns3/uav-demo.cc` into
`<NS3_PATH>/scratch/uav-demo.cc` (a step `run_full_pipeline.py` does not do
itself) and then invokes `scripts/run_full_pipeline.py --mode <mode>
--steps <n> --ns3-path <NS3_PATH>`.

---

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | file/asset presence, NS-3 path |
| GET | `/api/run/latest` | newest complete result (snapshot if any, else live `data/*.json`) |
| GET | `/api/ppo/trajectory` | **raw** `data/ppo_trajectory.json` |
| GET | `/api/ns3/results` | **raw** `data/full_pipeline_results.json` |
| GET | `/api/experiment` | `experiment_summary.json` + `experiment_metadata.json` + per-step CSV |
| GET | `/api/datasets/grid-sweep` | numeric points from `sionna/grid_sweep_dataset.json` + provenance note |
| GET | `/api/datasets/earthquake` | per-state timesteps from `sionna/earthquake_trajectory_dataset.json` + provenance note |
| GET | `/api/scene/meta` | `scene_meta.json` + live UAV/user coords + trajectory |
| GET | `/api/scene/geometry?variant=damaged\|baseline` | the `.glb` file |
| POST | `/api/run` `{mode, steps}` | `{run_id, status}` (202) |
| GET | `/api/run/{run_id}` | run status + log tail |
| GET | `/api/runs` | run history (this process) |

---

## Closed-loop control view (simulated digital twin)

The "Closed-loop control (simulated)" tab drives `ml/closed_loop_control.py`
(settings: `configs/closed_loop_control.json`) through
`dashboard/backend/control.py`. Each control interval: observe (Sionna RT at
the current UAV positions + ns-3 per-user delivery/latency) -> evaluate the
objective and per-UAV diagnostics -> each UAV that needs to move picks its own
up/down/+-x/+-y move (safety-checked: scenario boundary, 15-60 m, 5 m above
roofs within 10 m, 15 m separation) -> move -> re-simulate -> keep the move if
the measured objective holds, otherwise fly back and ban that move for a few
intervals. The ns-3 channel is obstruction-aware (Sionna loss per UAV-user
link) or distance-only (original), selectable.

| Method | Path | |
|---|---|---|
| GET | `/api/control/config` | default config, priority presets |
| GET | `/api/control/state` | full live state (UAVs, survivors, metrics, history, decisions, before/after, provenance) |
| POST | `/api/control/start` | `{channel_mode, preset \| weights, max_steps, initial_altitude_m, traffic_seconds, events, rf_mode, network}` |
| POST | `/api/control/pause`, `/resume`, `/reset` | pause/resume act at the next phase boundary; reset stops the process (kept on disk) |
| GET | `/api/control/runs` | past sessions |

Sessions are written to `~/.cache/resqnet_control/<run_id>/` (override with
`RESQNET_CONTROL_DIR`): `state.json`, `run.log`, `config.json`. One interval
takes about 30-60 s with Sionna + ns-3. Everything is simulated; no physical
UAV or telemetry is connected. `rf_mode=surrogate` / `network=mock` are test
modes and are labelled as such in the UI.

CLI equivalent: `python ml/closed_loop_control.py --out ~/.cache/resqnet_control/my_run`.
Tests: `python -m unittest ml/test_closed_loop_control.py`, `cd dashboard/frontend && npm test`.

---

## Run history

Each dashboard-triggered run copies the produced
`data/{uav_positions,ppo_trajectory,full_pipeline_results}.json` into
`data/runs/<run_id>/` plus a `meta.json`. The live `data/*.json` files stay
exactly where they are, so `scripts/run_full_pipeline.py`, `ml/run_ppo.py`
and the notebooks are unaffected. `data/runs/` is git-ignored.

`GET /api/run/latest` prefers the newest successful snapshot; with none it
falls back to the live files. The run registry itself is in-memory (per
backend process); the snapshots on disk persist.

---

## How the 3D web assets were generated

`build_assets.py` loads `sionna/scenes/city_damaged.obj` and
`city_baseline.obj` with `trimesh`, keeps only faces fully inside a
+/- 700 m box around the origin, and exports `.glb` (about 18.5k triangles /
~445 KB each). **Coordinates are not changed** - the cropped meshes stay in
the original Sionna frame, so real UAV/user coordinates overlay directly.
Regenerate with `python dashboard/build_assets.py` (`--half-extent`,
`--decimate` available). The originals in `sionna/scenes/` are never
modified.

The viewer wraps mesh + markers in a group rotated -90 deg about X, because
the OBJ/GLB uses Z as the height axis.

---

## Known limitations (shown honestly in the UI, not faked)

- **Geometry near the operating region.** The dense modelled city sits
  ~400-800 m from the origin. Within +/- 200 m of the origin the mesh has
  only a few hundred triangles - mostly open ground with sparse low
  structures. The UAV (~26 m) therefore has near line-of-sight to the
  ground users; this is why Sionna reports high coverage here. The wide
  crop makes this spatial relationship visible.
- **Damage labels.** `sionna/scenes/damage_labels.geojson` is in a
  different, unaligned coordinate frame. It is **not** projected onto the
  scene - the UI states "not shown".
- **Sionna vs NS-3.** They use different physical-layer models. NS-3 uses
  its own Wi-Fi channel; in the pipeline-results view it does **not**
  consume Sionna ray-traced channel loss. (The closed-loop control view can:
  in obstruction-aware mode each UAV-user link gets Sionna's loss in excess
  of free space on top of ns-3's distance model.) The UI keeps the two stages
  visually distinct.
- **SINR = SNR.** One UAV, no co-channel interference, so SINR reduces to
  SNR. The code's `mean_sinr_db` naming is kept; no multi-UAV interference
  is modelled or implied.
- **Single run.** Every value is one Monte-Carlo ray-tracing run; sub-dB
  variation near thresholds is expected (see `experiment_metadata.json`).
- **Coverage heatmap over the live scene** is not available as data - the
  pipeline only saves it as a PNG - so it is not shown.
- **Earthquake / grid-sweep datasets** were generated on a placeholder
  street-canyon scene, not `city_damaged.xml`; the API labels this.
