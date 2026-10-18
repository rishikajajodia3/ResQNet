// Pure helpers that turn the backend control state (GET /api/control/state,
// written by ml/closed_loop_control.py) into what the closed-loop view draws.
// Nothing here simulates or interpolates: every value comes from the state.

export const UAV_HUES = ["#1d4ed8", "#0f766e", "#7c3aed", "#b45309", "#be185d"];
export const PDR_GOOD = "#157347";
export const PDR_MID = "#b7791f";
export const PDR_BAD = "#b42318";

export const uavColor = (k) => UAV_HUES[k % UAV_HUES.length];

export function pdrColor(pdr, target = 0.9, threshold = 0.5) {
  if (pdr === null || pdr === undefined) return PDR_BAD;
  if (pdr >= target) return PDR_GOOD;
  if (pdr >= threshold) return PDR_MID;
  return PDR_BAD;
}

export const ACTIVE = new Set(["starting", "running", "pausing", "paused"]);

export function isActive(state) {
  return !!state && ACTIVE.has(state.status) && state.session?.process_alive !== false;
}

// UAV and user geometry for the 3D scene, straight from the state.
export function sceneData(state) {
  if (!state || !state.uavs) return { uavs: [], users: [], trails: [] };
  const target = state.config?.objective?.pdr_target ?? 0.9;
  const thr = state.config?.objective?.coverage_pdr_threshold ?? 0.5;
  const uavs = state.uavs.map((u, k) => ({
    ...u,
    index: k,
    color: uavColor(k),
  }));
  const users = (state.users || []).map((u) => ({
    ...u,
    color: pdrColor(u.pdr, target, thr),
    servingPos: uavs[u.serving_uav - 1]?.measured_at || uavs[u.serving_uav - 1]?.position,
  }));
  // accepted positions over time, per UAV (reverted trials are not part of the path)
  const trails = uavs.map((_, k) =>
    (state.history || []).map((h) => h.positions[k]).filter(Boolean)
  );
  return { uavs, users, trails };
}

const HEAD = ["objective", "coverage", "pdr", "worst_decile_pdr", "mean_latency_ms"];

// One row per control interval for the time-series charts.
export function historySeries(state) {
  return (state?.history || []).map((h) => {
    const row = { step: h.step, kind: h.kind, users: h.num_users };
    for (const k of HEAD) row[k] = h.metrics?.[k] ?? null;
    if (h.trial_metrics && h.kind === "reverted") row.trial_objective = h.trial_metrics.objective;
    (h.positions || []).forEach((p, k) => { row[`z${k + 1}`] = p[2]; });
    return row;
  });
}

export const METRIC_LABELS = {
  objective: ["Objective J", true, 4],
  coverage: ["Coverage (users with PDR ≥ threshold)", true, 3],
  pdr: ["Packet delivery ratio", true, 3],
  worst_decile_pdr: ["Worst-decile user PDR", true, 3],
  mean_latency_ms: ["Mean per-user latency (ms)", false, 1],
  users_pdr_ge_target: ["Users at PDR target", true, 0],
  users_no_delivery: ["Users with no delivery", false, 0],
};

// Before (first measured state) vs after (current accepted state).
export function beforeAfter(state) {
  const b = state?.before?.metrics;
  const a = state?.after?.metrics;
  if (!b || !a) return [];
  return Object.entries(METRIC_LABELS).map(([key, [label, higherBetter, digits]]) => {
    const vb = b[key];
    const va = a[key];
    const d = vb === null || va === null || vb === undefined || va === undefined ? null : va - vb;
    let verdict = "same";
    if (d !== null && Math.abs(d) > 1e-9) verdict = (d > 0) === higherBetter ? "better" : "worse";
    return { key, label, before: vb, after: va, delta: d, verdict, digits };
  });
}

export function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

export function fmtSigned(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const s = Number(v).toFixed(digits);
  return v > 0 ? `+${s}` : s;
}

export function fmtPos(p) {
  return p ? `(${p[0].toFixed(0)}, ${p[1].toFixed(0)}, ${p[2].toFixed(0)})` : "—";
}

// Which policy drives the controller, straight from state.provenance.policy
// (written by the backend; absent in older runs = rule-based).
export function policyInfo(state) {
  const p = state?.provenance?.policy;
  if (!p || p.policy !== "ppo") {
    return { kind: "rule_based", label: "Rule-based 3D controller", details: [] };
  }
  const details = [
    ["checkpoint", p.checkpoint],
    ["sha256", p.checkpoint_sha256 ? p.checkpoint_sha256.slice(0, 12) : null],
    ["training run", p.run_id],
    ["trained with", p.trained_with_rf],
    ["training PDR model", p.trained_pdr_model],
    ["latency in reward", p.latency_in_reward ? "yes" : "no"],
    ["inference", p.deterministic ? "deterministic" : "stochastic"],
  ].filter(([, v]) => v !== null && v !== undefined);
  return { kind: "ppo", label: "PPO policy (MaskablePPO, 7 actions per UAV)", details };
}

// One row per control interval and UAV: the action recorded by the backend
// for that interval and the position after it (history[].actions / positions).
export function stepActionRows(state) {
  const rows = [];
  const hist = state?.history || [];
  hist.forEach((h, t) => {
    const prev = t > 0 ? hist[t - 1].positions : null;
    (h.positions || []).forEach((p, k) => {
      rows.push({
        step: h.step, kind: h.kind, uav: `UAV-${k + 1}`, k,
        action: h.actions ? h.actions[k] ?? null : null,
        from: prev ? prev[k] ?? null : null,
        position: p,
      });
    });
  });
  return rows;
}

// Per-UAV obstruction of the links it serves, from state.users[].obstruction_db
// (the Sionna excess loss handed to ns-3; 200 dB = no path).
export function uavObstruction(state) {
  return (state?.uavs || []).map((u, k) => {
    const mine = (state?.users || []).filter((s) => s.serving_uav === k + 1 && s.obstruction_db != null);
    const vals = mine.map((s) => s.obstruction_db);
    const mean = vals.length ? vals.reduce((a, b) => a + Math.min(b, 60), 0) / vals.length : null;
    return {
      uav: u.id, links: vals.length, mean_db: mean,
      share_over_10db: vals.length ? vals.filter((v) => v > 10).length / vals.length : null,
      no_path: vals.filter((v) => v >= 199).length,
    };
  });
}

// Decisions newest first, optionally for one UAV only (moves and reverts
// always shown; holds collapsed unless includeHolds).
export function decisionRows(state, { uavId = null, includeHolds = true } = {}) {
  return (state?.decisions || [])
    .filter((d) => (uavId ? d.uav_id === uavId : true))
    .filter((d) => includeHolds || d.outcome !== "held")
    .slice()
    .reverse();
}
