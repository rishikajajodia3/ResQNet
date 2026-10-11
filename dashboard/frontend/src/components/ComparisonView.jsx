import { Panel } from "./Panel.jsx";
import { UAV_COLORS, stepUavPositions, uavMovement, fmt } from "../multiuav.js";

function snapshot(step) {
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

const METRICS = [
  { key: "coverage", label: "Coverage", unit: "%", d: 1 },
  { key: "connected", label: "Connected users", unit: "", d: 0 },
  { key: "rss", label: "Mean serving RSS", unit: "dBm", d: 2 },
  { key: "snr", label: "Mean SNR", unit: "dB", d: 2 },
  { key: "sinr", label: "Mean SINR", unit: "dB", d: 2 },
];

export function ComparisonView({ traj }) {
  const steps = traj?.trajectory || [];
  if (steps.length < 2) return null;
  const a = snapshot(steps[0]);
  const b = snapshot(steps[steps.length - 1]);
  const move = uavMovement(traj);
  const totalMove = move.reduce((s, x) => s + x, 0);
  const nUav = traj.num_uavs || 1;

  const dCov = (b.coverage ?? 0) - (a.coverage ?? 0);
  const dConn = (b.connected ?? 0) - (a.connected ?? 0);
  let conclusion, cls;
  if (Math.abs(dCov) < 1e-6 && dConn === 0) {
    cls = "";
    conclusion = `PPO matched — not beaten — fixed placement (Δcoverage ${dCov >= 0 ? "+" : ""}${dCov.toFixed(1)} pp, Δconnected ${dConn >= 0 ? "+" : ""}${dConn}).`;
  } else if (dCov > 0 || dConn > 0) {
    cls = "pos";
    conclusion = `PPO improved connectivity: Δcoverage +${dCov.toFixed(1)} pp, Δconnected +${dConn}.`;
  } else {
    cls = "neg";
    conclusion = `PPO regressed vs fixed: Δcoverage ${dCov.toFixed(1)} pp, Δconnected ${dConn}.`;
  }

  return (
    <Panel
      title="Fixed / initial vs PPO-optimised"
      right={`step 0 → step ${traj.num_steps} · ${nUav} UAV`}
    >
      <table className="cmp-table">
        <thead>
          <tr><th>Metric</th><th>Fixed / initial</th><th>PPO-optimised</th><th>Δ</th></tr>
        </thead>
        <tbody>
          {METRICS.map((m) => {
            const av = a[m.key], bv = b[m.key];
            const dd = typeof av === "number" && typeof bv === "number" ? bv - av : null;
            const ddCls = dd == null || Math.abs(dd) < 1e-9 ? "" : dd > 0 ? "up" : "down";
            return (
              <tr key={m.key}>
                <td>{m.label}</td>
                <td className="num">{fmt(av, m.d)}{m.unit && " " + m.unit}</td>
                <td className="num">{fmt(bv, m.d)}{m.unit && " " + m.unit}</td>
                <td className="num"><span className={`d ${ddCls}`}>{dd == null ? "—" : (dd >= 0 ? "+" : "") + dd.toFixed(m.d)}{m.unit && " " + m.unit}</span></td>
              </tr>
            );
          })}
          <tr>
            <td>UAV movement</td>
            <td className="num">0.0 m</td>
            <td className="num">{totalMove.toFixed(1)} m</td>
            <td className="num"><span className="d">{totalMove.toFixed(1)} m</span></td>
          </tr>
        </tbody>
      </table>

      <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
        {a.positions.map((p0, k) => {
          const pf = b.positions[k] || p0;
          return (
            <div className="action-cell" key={k} style={{ borderLeftColor: UAV_COLORS[k % UAV_COLORS.length], flex: "1 1 220px" }}>
              <div className="ac-label">UAV {String(k + 1).padStart(2, "0")} · {move[k]?.toFixed(1) ?? "0.0"} m</div>
              <div className="ac-value mono" style={{ fontSize: 11, color: "var(--text-3)" }}>
                {p0.map((v) => v.toFixed(0)).join(", ")} → {pf.map((v) => v.toFixed(0)).join(", ")}
              </div>
            </div>
          );
        })}
      </div>

      <div className={`cmp-conclusion ${cls}`}>{conclusion}</div>
    </Panel>
  );
}
