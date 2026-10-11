import { Panel, EmptyState } from "./Panel.jsx";
import { UAV_COLORS, fmt } from "../multiuav.js";

function Metric({ label, value, unit, sub, accent }) {
  return (
    <div className={`metric${accent ? " " + accent : ""}`}>
      <div className="m-label">{label}</div>
      <div className="m-val">{value}{unit && <span className="u">{unit}</span>}</div>
      {sub && <div className="m-sub">{sub}</div>}
    </div>
  );
}

export function Ns3Section({ ns3, selected, onSelectUser }) {
  if (!ns3) {
    return (
      <Panel title="NS-3 network validation" tag={<span className="tag ns3">NS-3</span>}>
        <EmptyState label="No NS-3 results yet — run an experiment." />
      </Panel>
    );
  }
  const s = ns3.ns3_summary || {};
  const flows = ns3.flows || [];
  const multi = (ns3.num_uavs || 1) > 1;
  const perUav = ns3.per_uav || [];
  const rate = s.total_flows ? (s.successful_flows / s.total_flows) * 100 : 0;
  const rateAccent = rate >= 80 ? "ok" : rate >= 40 ? "warn" : "bad";

  // map flows to user index by ascending source IP for row selection sync
  const ordered = [...flows]
    .map((f, i) => ({ ...f, _orig: i }))
    .sort((a, b) => cmpIp(a.source, b.source));

  return (
    <Panel
      title="NS-3 network validation"
      tag={<span className="tag ns3">NS-3</span>}
      right={
        <span className="hint">
          <span className={`sdot ${ns3.position_handoff_to_ns3 ? "ok" : "bad"}`} />
          handoff {ns3.position_handoff_to_ns3 ? "verified" : "unverified"}{multi ? ` · ${ns3.num_uavs} UAV` : ""}
        </span>
      }
    >
      <div className="metric-grid">
        <Metric label="Total flows" value={s.total_flows ?? "—"} sub="one UDP flow per user" />
        <Metric label="Delivering" value={s.successful_flows ?? "—"} accent={rateAccent} />
        <Metric label="No delivery" value={s.failed_flows ?? "—"} />
        <Metric label="Avg throughput" value={fmt(s.average_throughput_mbps, 4)} unit="Mbps" sub="delivering flows" />
        <Metric label="Avg delay" value={fmt(s.average_delay_ms, 3)} unit="ms" sub="delivering flows" />
        <Metric label="Flow-failure rate" value={fmt(s.overall_flow_loss_percent, 1)} unit="%" accent={rateAccent} sub="zero-delivery / total" />
      </div>

      {multi && perUav.length > 0 && (
        <>
          <div className="section-title" style={{ marginTop: 14 }}>Per-UAV — flows are user → associated UAV</div>
          <div className="metric-grid">
            {perUav.map((pu) => (
              <div className="metric" key={pu.uav_index} style={{ borderLeftColor: UAV_COLORS[pu.uav_index % UAV_COLORS.length] }}>
                <div className="m-label">UAV {String(pu.uav_index + 1).padStart(2, "0")}</div>
                <div className="m-val">{pu.delivering_flows}/{pu.flows}<span className="u">flows</span></div>
                <div className="m-sub">{fmt(pu.avg_throughput_mbps, 4)} Mbps · {fmt(pu.avg_delay_ms, 3)} ms</div>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="table-scroll" style={{ marginTop: 12 }}>
        <table className="data">
          <thead>
            <tr>
              <th>User</th><th>Source</th><th>Dest</th>{multi && <th>UAV</th>}
              <th>Tx</th><th>Rx</th><th>Mbps</th><th>Delay ms</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((fl, i) => {
              const ok = fl.status === "OK";
              const k = fl.uav_index;
              const isSel = selected?.type === "user" && selected.index === i;
              return (
                <tr
                  key={i}
                  className={`clickable${isSel ? " selected" : ""}`}
                  onClick={() => onSelectUser?.(isSel ? null : { type: "user", index: i })}
                >
                  <td>u{i + 1}</td>
                  <td>{fl.source}</td>
                  <td>{fl.destination}</td>
                  {multi && (
                    <td>{k === undefined ? "—" : (
                      <span className="badge" style={{ color: UAV_COLORS[k % UAV_COLORS.length], borderColor: UAV_COLORS[k % UAV_COLORS.length] }}>{k + 1}</span>
                    )}</td>
                  )}
                  <td>{fl.packets_sent ?? "—"}</td>
                  <td>{fl.packets_received ?? 0}</td>
                  <td>{typeof fl.throughput_mbps === "number" ? fl.throughput_mbps.toFixed(4) : "—"}</td>
                  <td>{typeof fl.average_delay_ms === "number" ? fl.average_delay_ms.toFixed(3) : "—"}</td>
                  <td><span className={`badge ${ok ? "ok" : "bad"}`}>{ok ? "delivering" : "no delivery"}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="inline-note">
        &ldquo;Flow-failure rate&rdquo; = share of the {s.total_flows ?? "?"} UDP flows with zero packets —
        not per-packet loss. NS-3 uses its own <span className="mono">YansWifiChannel</span>; it does not
        consume Sionna&rsquo;s ray-traced channel loss.
      </p>
    </Panel>
  );
}

function cmpIp(a, b) {
  const pa = String(a).split(".").map(Number);
  const pb = String(b).split(".").map(Number);
  for (let i = 0; i < 4; i++) if (pa[i] !== pb[i]) return pa[i] - pb[i];
  return 0;
}
