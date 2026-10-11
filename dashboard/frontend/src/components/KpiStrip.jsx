import { fmt, fmtInt } from "../multiuav.js";

function Cell({ label, value, unit, sub, delta, primary, state }) {
  return (
    <div className={`kpi-cell${primary ? " primary" : ""}`}>
      <div className="kc-label">
        {state && state !== "ok" && <span className={`sdot ${state}`} />}
        {label}
      </div>
      <div className="kc-value">
        {value}
        {unit && <span className="u">{unit}</span>}
      </div>
      {delta ? (
        <span className={`kc-delta ${delta.dir}`}>{delta.text}</span>
      ) : (
        sub && <div className="kc-sub">{sub}</div>
      )}
    </div>
  );
}

function deltaOf(a, b, unit = "", digits = 1) {
  if (typeof a !== "number" || typeof b !== "number") return null;
  const d = b - a;
  const dir = Math.abs(d) < 1e-9 ? "flat" : d > 0 ? "up" : "down";
  const text = (d >= 0 ? "+" : "") + d.toFixed(digits) + (unit ? " " + unit : "") + " vs initial";
  return { dir, text };
}

export function KpiStrip({ traj, ns3, stepIndex }) {
  const steps = traj?.trajectory || [];
  const step = steps[stepIndex] || steps[steps.length - 1];
  if (!step) return null;
  const first = steps[0];
  const nUav = traj.num_uavs || 1;
  const s = ns3?.ns3_summary;

  const covState =
    step.coverage_percent >= 99.99 ? "ok" : step.coverage_percent >= 70 ? "warn" : "bad";
  const connState =
    step.connected_users >= traj.num_users ? "ok"
      : step.connected_users >= traj.num_users * 0.7 ? "warn" : "bad";
  const ns3Rate = s ? (s.successful_flows / s.total_flows) * 100 : null;
  const ns3State = ns3Rate == null ? "ok" : ns3Rate >= 80 ? "ok" : ns3Rate >= 40 ? "warn" : "bad";
  const sinrState =
    nUav > 1 && typeof step.mean_sinr_db === "number" && step.mean_sinr_db < 3 ? "warn" : "ok";

  return (
    <div className="kpi-strip">
      <Cell label="Coverage" value={fmt(step.coverage_percent, 1)} unit="%"
        state={covState} delta={deltaOf(first.coverage_percent, step.coverage_percent, "pp")} />
      <Cell label="Connected users" value={`${step.connected_users}`} unit={`/ ${traj.num_users}`}
        state={connState} delta={deltaOf(first.connected_users, step.connected_users, "", 0)} />
      <Cell label="UAV fleet" value={fmtInt(nUav)} primary
        sub={`${traj.mode === "surrogate" ? "surrogate RF" : "Sionna RT"} · ${traj.scenario || "legacy"}`} />
      <Cell label="Mean SNR" value={fmt(step.mean_snr_db ?? step.mean_sinr_db)} unit="dB"
        sub="interference-free" />
      <Cell label="Mean SINR" value={fmt(step.mean_sinr_db)} unit="dB"
        state={sinrState} sub={nUav > 1 ? "co-channel" : "= SNR (1 UAV)"} />
      <Cell label="NS-3 delivery" value={s ? `${s.successful_flows}` : "—"} unit={s ? `/ ${s.total_flows}` : ""}
        state={ns3State} sub={s ? `${fmt(s.overall_flow_loss_percent, 0)}% flow-failure` : "no NS-3 data"} />
    </div>
  );
}
