import { Panel } from "./Panel.jsx";
import { UAV_COLORS, perUavStats, uavMovement, fmt } from "../multiuav.js";

export function UavFleet({ traj, stepIndex, ns3, selected, onSelect }) {
  const step = traj?.trajectory?.[stepIndex];
  if (!step) return null;
  const stats = perUavStats(step);
  const move = uavMovement(traj);
  const perUavNs3 = ns3?.per_uav || [];

  return (
    <Panel title="UAV fleet" right={`${stats.length} active · step ${step.step}`} flush>
      <div className="fleet">
        <div className="fleet-head">
          <span>UAV</span>
          <span>Status</span>
          <span>Users</span>
          <span className="fc-hide">RSS</span>
          <span className="fc-hide">SNR</span>
          <span className="fc-hide">SINR</span>
          <span className="fc-hide">Move</span>
        </div>
        {stats.map((st) => {
          const color = UAV_COLORS[st.index % UAV_COLORS.length];
          const isSel = selected?.type === "uav" && selected.index === st.index;
          const overloaded = st.usersServed > 0 && st.connected / st.usersServed < 0.7;
          const ns = perUavNs3.find((p) => p.uav_index === st.index);
          const poorDelivery = ns && ns.flows > 0 && ns.delivering_flows / ns.flows < 0.4;
          const state = poorDelivery || overloaded ? "warn" : "ok";
          return (
            <div
              key={st.index}
              className={`fleet-card${isSel ? " selected" : ""}`}
              style={{ borderLeftColor: isSel ? "var(--primary)" : color }}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(isSel ? null : { type: "uav", index: st.index })}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(isSel ? null : { type: "uav", index: st.index });
                }
              }}
            >
              <span className="fc-id">UAV {String(st.index + 1).padStart(2, "0")}</span>
              <span className="fc-status">
                <span className={`sdot ${state}`} />
                {poorDelivery ? "Low delivery" : overloaded ? "Users unlinked" : "Nominal"}
              </span>
              <span className="fc-num">{st.usersServed}</span>
              <span className="fc-num fc-hide">{fmt(st.meanRss, 0)} dBm</span>
              <span className="fc-num fc-hide">{fmt(st.meanSnr, 0)} dB</span>
              <span className="fc-num fc-hide">{fmt(st.meanSinr, 1)} dB</span>
              <span className="fc-num fc-hide">{typeof move[st.index] === "number" ? move[st.index].toFixed(1) + " m" : "—"}</span>
            </div>
          );
        })}
      </div>
      <p className="inline-note" style={{ padding: "8px 12px 12px" }}>
        Per-UAV values aggregate the real per-user Sionna results for this step,
        grouped by best-RSS serving UAV.
      </p>
    </Panel>
  );
}
