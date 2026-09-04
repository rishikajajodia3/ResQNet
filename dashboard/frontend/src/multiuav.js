// Shared helpers for reading trajectories that may be single-UAV (legacy) or
// multi-UAV. Legacy steps have `position` / `action`; multi-UAV steps also
// have `uav_positions` / `actions`.

export const UAV_COLORS = ["#ffcf6b", "#7cc4ff", "#c78be0", "#7fd6a6", "#e08a8a"];

export const ACTION_NAMES = ["stay", "-X", "+X", "-Y", "+Y"];

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
  return typeof v === "number" ? v.toFixed(d) : "—";
}

// Join a trajectory step's per-user Sionna results with the NS-3 flow list so
// each user carries both a Sionna "connected" state and an NS-3 "delivery"
// state. NS-3 flows are user i -> its UAV; flow order follows FlowMonitor,
// which is not guaranteed to match user index, so we match by count only and
// fall back to null when unknown.
export function ns3DeliveryByUser(ns3) {
  if (!ns3 || !Array.isArray(ns3.flows)) return null;
  // flows sorted by source address ascending == user order in practice
  const flows = [...ns3.flows].sort((a, b) => cmpIp(a.source, b.source));
  return flows.map((f) => (f.packets_received || 0) > 0);
}

function cmpIp(a, b) {
  const pa = String(a).split(".").map(Number);
  const pb = String(b).split(".").map(Number);
  for (let i = 0; i < 4; i++) if (pa[i] !== pb[i]) return pa[i] - pb[i];
  return 0;
}
