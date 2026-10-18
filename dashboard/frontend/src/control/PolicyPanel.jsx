import { Fragment, useState } from "react";
import { policyInfo, stepActionRows, uavObstruction, uavColor, fmtNum, fmtPos } from "./model.js";

// Policy provenance + per-interval xyz / action table + per-UAV link obstruction.
// Everything shown here is read from the backend state; nothing is derived
// beyond formatting and simple averages.
export function PolicyBadge({ state }) {
  const p = policyInfo(state);
  return (
    <span className={`pill pol-${p.kind}`} title={p.details.map(([k, v]) => `${k}: ${v}`).join("\n")}>
      {p.kind === "ppo" ? "PPO" : "rule-based"}
    </span>
  );
}

export function PolicyPanel({ state }) {
  const p = policyInfo(state);
  const rows = stepActionRows(state);
  const obs = uavObstruction(state);
  const [all, setAll] = useState(false);
  const lastStep = rows.length ? rows[rows.length - 1].step : null;
  const shown = all ? rows.slice().reverse() : rows.filter((r) => r.step === lastStep);
  return (
    <div className="ctl-policy">
      <div className="ctl-insp">
        <span>policy</span><b>{p.label}</b>
        {p.details.map(([k, v]) => (
          <Fragment key={k}><span>{k}</span><span className="mono">{String(v)}</span></Fragment>
        ))}
      </div>
      <h4 className="ctl-sub">
        Positions and actions {all ? "(all intervals)" : `(interval ${lastStep ?? "—"})`}{" "}
        <label className="small"><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> all</label>
      </h4>
      <div className="ctl-scroll">
        <table className="ctl-table">
          <thead><tr><th>int.</th><th>UAV</th><th>action</th><th>from (x, y, z)</th><th>to (x, y, z)</th></tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr key={`${r.step}-${r.k}`}>
                <td>{r.step}</td>
                <td><span className="dot" style={{ background: uavColor(r.k) }} />{r.uav}</td>
                <td><b>{r.action ?? (r.kind === "initial" ? "start" : "—")}</b></td>
                <td>{fmtPos(r.from)}</td>
                <td>{fmtPos(r.position)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h4 className="ctl-sub">Link obstruction at the measured positions (Sionna excess loss passed to ns-3)</h4>
      <table className="ctl-table">
        <thead><tr><th>UAV</th><th>served links</th><th>mean excess (dB, capped 60)</th><th>&gt; 10 dB</th><th>no path</th></tr></thead>
        <tbody>
          {obs.map((o, k) => (
            <tr key={o.uav}>
              <td><span className="dot" style={{ background: uavColor(k) }} />{o.uav}</td>
              <td>{o.links}</td>
              <td>{fmtNum(o.mean_db, 1)}</td>
              <td>{o.share_over_10db == null ? "—" : `${Math.round(o.share_over_10db * 100)} %`}</td>
              <td>{o.no_path}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
