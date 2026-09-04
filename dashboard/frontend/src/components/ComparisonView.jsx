import { Panel } from "./Panel.jsx";
import { UAV_COLORS, stepUavPositions, fmt } from "../multiuav.js";

function side(step) {
  if (!step) return null;
  return {
    positions: stepUavPositions(step),
    coverage: step.coverage_percent,
    connected: step.connected_users,
    rss: step.mean_rss_dbm,
    snr: step.mean_snr_db ?? step.mean_sinr_db,
    sinr: step.mean_sinr_db,
  };
}

function Row({ label, a, b, unit, better }) {
  const delta = typeof a === "number" && typeof b === "number" ? b - a : null;
  const good = better && delta !== null ? better(delta) : null;
  return (
    <tr>
      <td>{label}</td>
      <td>{fmt(a)}{unit ? " " + unit : ""}</td>
      <td>{fmt(b)}{unit ? " " + unit : ""}</td>
      <td style={{ color: good === null ? "var(--text-dim)" : good ? "var(--ok)" : "var(--bad)" }}>
        {delta === null ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(2)}
      </td>
    </tr>
  );
}

export function ComparisonView({ traj }) {
  const steps = traj?.trajectory || [];
  if (steps.length < 2) return null;
  const a = side(steps[0]);
  const b = side(steps[steps.length - 1]);
  const nUav = traj.num_uavs || 1;

  return (
    <Panel
      title="Initial vs PPO-optimised"
      hint={`${nUav} UAV${nUav > 1 ? "s" : ""} · step 0 → step ${traj.num_steps}`}
    >
      <div style={{ overflowX: "auto" }}>
        <table className="data">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Initial</th>
              <th>PPO final</th>
              <th>Δ</th>
            </tr>
          </thead>
          <tbody>
            <Row label="Coverage %" a={a.coverage} b={b.coverage} better={(d) => d >= 0} />
            <Row label="Connected users" a={a.connected} b={b.connected} better={(d) => d >= 0} />
            <Row label="Mean serving RSS" a={a.rss} b={b.rss} unit="dBm" better={(d) => d >= 0} />
            <Row label="Mean SNR" a={a.snr} b={b.snr} unit="dB" better={(d) => d >= 0} />
            <Row label="Mean SINR" a={a.sinr} b={b.sinr} unit="dB" better={(d) => d >= 0} />
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}>
        {a.positions.map((p0, k) => {
          const pf = b.positions[k] || p0;
          const moved = Math.hypot(pf[0] - p0[0], pf[1] - p0[1], pf[2] - p0[2]);
          return (
            <div className="kpi" key={k} style={{ borderLeft: `3px solid ${UAV_COLORS[k % UAV_COLORS.length]}`, minWidth: 190 }}>
              <div className="k-label">UAV {k + 1} — moved {moved.toFixed(1)} m</div>
              <div className="k-sub mono">
                {p0.map((v) => v.toFixed(0)).join(", ")} → {pf.map((v) => v.toFixed(0)).join(", ")}
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
