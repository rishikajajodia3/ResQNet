// All requests go to same-origin "/api" (Vite proxies this to FastAPI in dev).
const BASE = "/api";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (e) {
      /* ignore */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (e) {
      /* ignore */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => get("/health"),
  ppoTrajectory: () => get("/ppo/trajectory"),
  ns3Results: () => get("/ns3/results"),
  experiment: () => get("/experiment"),
  latest: () => get("/run/latest"),
  sceneMeta: () => get("/scene/meta"),
  gridSweep: () => get("/datasets/grid-sweep"),
  earthquake: () => get("/datasets/earthquake"),
  scenarios: () => get("/scenarios"),
  trainingSummary: () => get("/training/summary"),
  experiments: () => get("/experiments"),
  geometryUrl: (variant) => `${BASE}/scene/geometry?variant=${variant}`,
  startRun: (opts) => post("/run", opts),
  runStatus: (runId) => get(`/run/${runId}`),
  // closed-loop control (ml/closed_loop_control.py via dashboard/backend/control.py)
  controlConfig: () => get("/control/config"),
  controlState: () => get("/control/state"),
  controlStart: (opts) => post("/control/start", opts),
  controlPause: () => post("/control/pause", {}),
  controlResume: () => post("/control/resume", {}),
  controlReset: () => post("/control/reset", {}),
  controlRuns: () => get("/control/runs"),
  controlPpoModels: () => get("/control/ppo_models"),
};
