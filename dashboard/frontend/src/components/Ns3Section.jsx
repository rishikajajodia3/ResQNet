import { Panel } from "./Panel.jsx";
import { UAV_COLORS, fmt } from "../multiuav.js";

function Kpi({ label, value, unit, sub }) {
  return (
    <div className="kpi">
      <div className="k-label">{label}</div>
      <div className="k-value">
        {value}
        {unit && <span className="k-unit">{unit}</span>}
      </div>
      {sub && <div className="k-sub">{sub}</div>}
    </div>
  );
}

export function Ns3Section({ ns3 }) {
  if (!ns3) return null;
  const s = ns3.ns3_summary || {};
  const flows = ns3.flows || [];
  const multi = (ns3.num_uavs || 1) > 1;
  const perUav = ns3.per_uav || [];

  return (
    <Panel
      title="NS-3 network validation"
      tag={<span className="tag ns3">NS-3</span>}
      hint={
        ns3.position_handoff_to_ns3
          ? `UAV position handoff verified${multi ? " (all " + ns3.num_uavs + " UAVs)" : ""}`
          : "handoff not verified"
      }
    >
      <div className="kpi-grid">
        <Kpi label="Total flows" value={s.total_flows ?? "—"} sub="one UDP flow per user" />
        <Kpi label="Delivering flows" value={s.successful_flows ?? "—"} />
        <Kpi label="Flows with no delivery" value={s.failed_flows ?? "—"} />
        <Kpi label="Avg throughput" value={fmt(s.average_throughput_mbps, 4)} unit="Mbps" sub="delivering flows only" />
        <Kpi label="Avg delay" value={fmt(s.average_delay_ms, 3)} unit="ms" sub="delivering flows only" />
        <Kpi
          label="Flow-failure rate"
          value={fmt(s.overall_flow_loss_percent, 1)}
          unit="%"
          sub="flows with zero delivery / total"
        />
      </div>

      <p className="inline-note">
        &ldquo;Flow-failure rate&rdquo; = share of the {s.total_flows ?? "?"} UDP flows that
        received no packets — not per-packet loss. Delivering flows individually report 0 % loss.
        NS-3 uses its own Wi-Fi channel model; it does not consume Sionna&rsquo;s ray-traced loss.
      </p>

      {multi && perUav.length > 0 && (
        <>
          <h4 style={{ fontSize: 12, color: "var(--text-dim)", margin: "12px 0 6px" }}>
            Per-UAV (flows are user → associated UAV)
          </h4>
          <div className="kpi-grid">
            {perUav.map((pu) => (
              <div
                className="kpi"
                key={pu.uav_index}
                style={{ borderLeft: `3px solid ${UAV_COLORS[pu.uav_index % UAV_COLORS.length]}` }}
              >
                <div className="k-label">UAV {pu.uav_index + 1}</div>
                <div className="k-value">
                  {pu.delivering_flows}/{pu.flows}
                  <span className="k-unit">flows delivering</span>
                </div>
                <div className="k-sub">
                  {fmt(pu.avg_throughput_mbps, 4)} Mbps · {fmt(pu.avg_delay_ms, 3)} ms
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div style={{ overflowX: "auto", marginTop: 12 }}>
        <table className="data">
          <thead>
            <tr>
              <th>Source</th>
              <th>Destination</th>
              {multi && <th>UAV</th>}
              <th>Tx packets</th>
              <th>Rx packets</th>
              <th>Throughput Mbps</th>
              <th>Delay ms</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {flows.map((fl, i) => {
              const ok = fl.status === "OK";
              const k = fl.uav_index;
              return (
                <tr key={i}>
                  <td>{fl.source}</td>
                  <td>{fl.destination}</td>
                  {multi && (
                    <td>
                      {k === undefined ? (
                        "—"
                      ) : (
                        <span
                          className="badge"
                          style={{ color: UAV_COLORS[k % UAV_COLORS.length], borderColor: UAV_COLORS[k % UAV_COLORS.length] }}
                        >
                          {k + 1}
                        </span>
                      )}
                    </td>
                  )}
                  <td>{fl.packets_sent ?? "—"}</td>
                  <td>{fl.packets_received ?? 0}</td>
                  <td>{typeof fl.throughput_mbps === "number" ? fl.throughput_mbps.toFixed(4) : "—"}</td>
                  <td>{typeof fl.average_delay_ms === "number" ? fl.average_delay_ms.toFixed(3) : "—"}</td>
                  <td>
                    <span className={`badge ${ok ? "ok" : "bad"}`}>
                      {ok ? "delivering" : "no delivery"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
