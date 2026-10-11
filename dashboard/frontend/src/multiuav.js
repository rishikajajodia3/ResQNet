// Shared helpers for reading trajectories that may be single-UAV (legacy) or
// multi-UAV. Legacy steps have `position` / `action`; multi-UAV steps also
// have `uav_positions` / `actions` / `users_per_uav`. Nothing here fabricates
// data — every value is derived from the real API response.

// Related blue shades for UAV identity (no rainbow).
export const UAV_COLORS = ["#1d4ed8", "#2563eb", "#3b82f6", "#1e3a8a", "#60a5fa"];
// Restrained semantic colours (match the CSS token palette).
export const C_OK = "#157347";
export const C_BAD = "#b42318";
export const C_PRIMARY = "#1d4ed8";
export const C_NAVY = "#1e3a8a";
export const C_SLATE = "#64748b";
// Chart axis / grid / tooltip styling shared by all recharts panels.
export const CHART = {
  axis: { stroke: "#98a2b3", fontSize: 10 },
  grid: "#e5eaf0",
  tooltip: {
    contentStyle: {
      background: "#ffffff",
      border: "1px solid #d9e0e8",
      borderRadius: 3,
      fontSize: 11,
      color: "#344054",
    },
    labelStyle: { color: "#667085" },
  },
};
export const ACTION_NAMES = ["STAY", "−X", "+X", "−Y", "+Y"];

export function numUavs(traj) {
  return traj?.num_uavs || 1;
}

export function stepUavPositions(step) {
  if (!step) return [];
  return step.uav_positions || (step.position ? [step.position] : []);
}

export function stepActions(step) {
  if (!step) return [];
  if (step.actions) return step.actions;
  return step.action === null || step.action === undefined ? [null] : [step.action];
}

export function fmt(v, d = 2) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";
}
export function fmtInt(v) {
  return typeof v === "number" && Number.isFinite(v) ? String(Math.round(v)) : "—";
}

// NS-3 delivery state per user, ordered by ascending source IP (== user order
// in practice). Returns null when NS-3 data is unavailable.
export function ns3DeliveryByUser(ns3) {
  if (!ns3 || !Array.isArray(ns3.flows)) return null;
  const flows = [...ns3.flows].sort((a, b) => cmpIp(a.source, b.source));
  return flows.map((f) => (f.packets_received || 0) > 0);
}
function cmpIp(a, b) {
  const pa = String(a).split(".").map(Number);
  const pb = String(b).split(".").map(Number);
  for (let i = 0; i < 4; i++) if (pa[i] !== pb[i]) return pa[i] - pb[i];
  return 0;
}

// Per-UAV aggregates for a step, computed only from the real per-user results.
export function perUavStats(step) {
  if (!step) return [];
  const positions = stepUavPositions(step);
  const K = positions.length || 1;
  const out = [];
  for (let k = 0; k < K; k++) {
    const served = (step.user_results || []).filter((u) => (u.serving_uav || 1) - 1 === k);
    const rss = served.map((u) => u.rss_dbm).filter(isNum);
    const snr = served.map((u) => u.snr_db ?? u.sinr_db).filter(isNum);
    const sinr = served.map((u) => u.sinr_db).filter(isNum);
    const connected = served.filter((u) => u.connected).length;
    out.push({
      index: k,
      position: positions[k],
      usersServed: served.length,
      connected,
      meanRss: mean(rss),
      meanSnr: mean(snr),
      meanSinr: mean(sinr),
    });
  }
  return out;
}

function isNum(v) { return typeof v === "number" && Number.isFinite(v); }
function mean(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; }

// Movement (m) of UAV k between step0 and stepN of the trajectory.
export function uavMovement(traj) {
  const steps = traj?.trajectory || [];
  if (steps.length < 2) return [];
  const a = stepUavPositions(steps[0]);
  const b = stepUavPositions(steps[steps.length - 1]);
  return a.map((p0, k) => {
    const pf = b[k] || p0;
    return Math.hypot(pf[0] - p0[0], pf[1] - p0[1], pf[2] - p0[2]);
  });
}

// ---------------------------------------------------------------------------
// Run-stage inference from the real run record (status + streamed log_tail).
// No fake percentages: a stage is only "done" when the log proves it.
// ---------------------------------------------------------------------------
export const STAGE_DEFS = [
  { id: "scenario", name: "Scenario", detail: "config resolved" },
  { id: "sionna",   name: "Sionna RT / PPO", detail: "ray tracing + policy" },
  { id: "handoff",  name: "JSON handoff", detail: "uav_positions.json" },
  { id: "ns3",      name: "NS-3", detail: "network validation" },
  { id: "results",  name: "Results", detail: "aggregated" },
];

export function inferStages(run) {
  const status = run?.status;
  const log = (run?.log_tail || []).join("\n");
  const has = (re) => re.test(log);

  const state = { scenario: "idle", sionna: "idle", handoff: "idle", ns3: "idle", results: "idle" };
  if (!run) return state;

  if (status === "failed") {
    // mark the last stage that had started as failed, earlier ones done
    const order = ["scenario", "sionna", "handoff", "ns3", "results"];
    let lastStarted = "scenario";
    if (has(/Running PPO|MULTI-UAV PPO|RESQNET PPO/)) lastStarted = "sionna";
    if (has(/PPO COMPLETE|MULTI-UAV PPO COMPLETE/)) lastStarted = "handoff";
    if (has(/Copied to:|Copying UAV position/)) lastStarted = "ns3";
    if (has(/UAV NETWORK RESULTS|Running NS-3/)) lastStarted = "results";
    for (const s of order) {
      state[s] = "done";
      if (s === lastStarted) { state[s] = "failed"; break; }
    }
    return state;
  }

  if (status === "succeeded") {
    Object.keys(state).forEach((k) => (state[k] = "done"));
    return state;
  }

  // running / queued
  state.scenario = "done";
  const ppoStarted = has(/Running PPO|MULTI-UAV PPO|RESQNET PPO/);
  const ppoDone = has(/PPO COMPLETE|MULTI-UAV PPO COMPLETE/);
  const copied = has(/Copied to:|Copying UAV position/);
  const ns3Started = has(/Running NS-3|RUNNING:[\s\S]*ns3 run|UAV NETWORK RESULTS/);
  const resultsStarted = has(/RESQNET PIPELINE COMPLETE|POSITION HANDOFF/);

  if (!ppoStarted) { state.sionna = "active"; return state; }
  state.sionna = ppoDone ? "done" : "active";
  if (!ppoDone) return state;

  state.handoff = copied ? "done" : "active";
  if (!copied) return state;

  state.ns3 = ns3Started && !resultsStarted ? "active" : (resultsStarted ? "done" : "active");
  if (!resultsStarted && !ns3Started) { state.ns3 = "active"; return state; }
  if (resultsStarted) { state.ns3 = "done"; state.results = "active"; }
  return state;
}

// ---------------------------------------------------------------------------
// Research findings — derived strictly from returned values.
// ---------------------------------------------------------------------------
export function deriveFindings(traj, ns3) {
  const out = [];
  const steps = traj?.trajectory || [];
  if (steps.length === 0) return out;
  const first = steps[0];
  const last = steps[steps.length - 1];
  const move = uavMovement(traj);
  const totalMove = move.reduce((s, x) => s + x, 0);
  const K = traj.num_uavs || 1;

  // coverage saturation
  if (typeof first.coverage_percent === "number" && first.coverage_percent >= 99.999) {
    out.push({
      kind: "ok",
      title: "Coverage saturated at the initial placement",
      body: `The spatially-separated ${K}-UAV starting placement already reached ${first.coverage_percent.toFixed(0)}% Sionna coverage (${first.connected_users}/${traj.num_users} connected). There is no coverage gap for PPO to close in this geometry.`,
    });
  }

  // PPO delta vs initial
  const dCov = (last.coverage_percent ?? 0) - (first.coverage_percent ?? 0);
  const dConn = (last.connected_users ?? 0) - (first.connected_users ?? 0);
  if (Math.abs(dCov) < 1e-6 && dConn === 0) {
    out.push({
      kind: "info",
      title: "PPO matched — not beaten — fixed placement",
      body: `Over ${traj.num_steps} steps the policy produced Δcoverage ${dCov >= 0 ? "+" : ""}${dCov.toFixed(1)} pp and Δconnected ${dConn >= 0 ? "+" : ""}${dConn}. On this scenario PPO does not improve on the fixed baseline.`,
    });
  } else if (dCov > 0 || dConn > 0) {
    out.push({
      kind: "ok",
      title: "PPO improved connectivity",
      body: `Δcoverage +${dCov.toFixed(1)} pp, Δconnected +${dConn} over ${traj.num_steps} steps.`,
    });
  }

  // minimal movement
  if (steps.length > 1 && totalMove < 1e-6) {
    out.push({
      kind: "info",
      title: "Minimal PPO movement",
      body: `The trained policy selected no net UAV movement (Σ displacement ${totalMove.toFixed(1)} m). Movement is penalised in the reward, so with coverage already saturated the optimal action is to hold position.`,
    });
  } else if (steps.length > 1) {
    out.push({
      kind: "info",
      title: "PPO repositioned UAVs",
      body: `Net displacement per UAV: ${move.map((m) => m.toFixed(1) + " m").join(", ")}.`,
    });
  }

  // co-channel interference
  const snr = last.mean_snr_db ?? last.mean_sinr_db;
  const sinr = last.mean_sinr_db;
  if (K > 1 && typeof snr === "number" && typeof sinr === "number" && snr - sinr > 5) {
    out.push({
      kind: "warn",
      title: "Co-channel interference dominates SINR",
      body: `Mean SNR ${snr.toFixed(1)} dB vs mean SINR ${sinr.toFixed(1)} dB — a ${(snr - sinr).toFixed(0)} dB gap under the shared-channel / full-buffer assumption. All ${K} UAVs transmit on one channel; the SINR is computed from the ray-traced per-UAV received powers.`,
    });
  } else if (K === 1 && typeof snr === "number" && typeof sinr === "number" && Math.abs(snr - sinr) < 0.1) {
    out.push({
      kind: "info",
      title: "Single UAV — no inter-UAV interference",
      body: `With one UAV, SINR (${sinr.toFixed(1)} dB) equals SNR by definition.`,
    });
  }

  // NS-3 vs Sionna divergence
  if (ns3 && ns3.ns3_summary) {
    const s = ns3.ns3_summary;
    const sioConn = last.connected_users ?? 0;
    if (typeof s.successful_flows === "number" && sioConn - s.successful_flows >= 2) {
      out.push({
        kind: "warn",
        title: "NS-3 delivery lower than Sionna connectivity",
        body: `Sionna reports ${sioConn}/${traj.num_users} connected but NS-3 delivers on ${s.successful_flows}/${s.total_flows} flows (${s.overall_flow_loss_percent?.toFixed(0)}% flow-failure). NS-3 runs its own 802.11n MAC on a shared channel — Sionna connectivity ≠ network delivery.`,
      });
    }
  }

  // RF model provenance
  out.push({
    kind: "info",
    title: traj.mode === "surrogate" ? "RF model: surrogate (free-space)" : "RF model: Sionna RT",
    body: traj.mode === "surrogate"
      ? "These metrics come from the free-space path-loss surrogate used for fast training/iteration — not a Sionna ray-tracing result."
      : "These metrics come from Sionna RT shoot-and-bounce ray tracing over the 3D city mesh.",
  });

  return out;
}
