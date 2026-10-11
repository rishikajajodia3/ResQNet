import { Panel } from "./Panel.jsx";
import { UAV_COLORS, fmt, ns3DeliveryByUser } from "../multiuav.js";

const sci = (v) => (typeof v === "number" ? v.toExponential(2) : "—");

export function PerUserTable({ traj, stepIndex, ns3, selected, onSelectUser }) {
  const step = traj?.trajectory?.[stepIndex];
  const users = step?.user_results || [];
  const multi = (traj?.num_uavs || 1) > 1;
  const delivery = ns3DeliveryByUser(ns3);

  return (
    <Panel
      title="Per-user link table"
      tag={<span className="tag sionna">{traj?.mode === "surrogate" ? "Surrogate" : "Sionna RT"}</span>}
      right={<span className="hint">step {step?.step ?? 0} · Sionna connectivity and NS-3 delivery shown separately</span>}
      flush
    >
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>User</th><th>Position m</th>{multi && <th>Serving</th>}
              <th>RSS dBm</th><th>SNR dB</th><th>SINR dB</th><th>Path gain dB</th>
              <th>Paths</th><th>Delay s</th><th>Sionna</th><th>NS-3</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u, i) => {
              const k = (u.serving_uav || 1) - 1;
              const del = delivery ? delivery[i] : null;
              const isSel = selected?.type === "user" && selected.index === i;
              return (
                <tr
                  key={u.user_id || i}
                  className={`clickable${isSel ? " selected" : ""}`}
                  onClick={() => onSelectUser?.(isSel ? null : { type: "user", index: i })}
                >
                  <td>{(u.user_id || `user_${i + 1}`).replace("user_", "u")}</td>
                  <td>{fmt(u.position?.[0], 0)}, {fmt(u.position?.[1], 0)}, {fmt(u.position?.[2], 1)}</td>
                  {multi && (
                    <td>
                      <span className="badge" style={{ color: UAV_COLORS[k % UAV_COLORS.length], borderColor: UAV_COLORS[k % UAV_COLORS.length] }}>
                        UAV {k + 1}
                      </span>
                    </td>
                  )}
                  <td>{fmt(u.rss_dbm)}</td>
                  <td>{fmt(u.snr_db ?? u.sinr_db)}</td>
                  <td>{fmt(u.sinr_db)}</td>
                  <td>{fmt(u.path_gain_db)}</td>
                  <td>{u.num_paths ?? "—"}</td>
                  <td>{sci(u.shortest_delay_sec)}</td>
                  <td><span className={`badge ${u.connected ? "ok" : "bad"}`}>{u.connected ? "connected" : "no"}</span></td>
                  <td>
                    {del === null
                      ? <span className="badge">—</span>
                      : <span className={`badge ${del ? "ok" : "bad"}`}>{del ? "delivery" : "no delivery"}</span>}
                  </td>
                </tr>
              );
            })}
            {users.length === 0 && (
              <tr><td colSpan={multi ? 11 : 10} style={{ textAlign: "center", color: "var(--text-4)" }}>
                No per-user data for this step.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="inline-note" style={{ padding: "8px 14px" }}>
        <span className="tag sionna">Sionna</span> = ray-traced propagation / connectivity ·{" "}
        <span className="tag ns3">NS-3</span> = 802.11n MAC + UDP delivery on its own channel.
        A user can be Sionna-connected with no NS-3 delivery (MAC contention) or vice-versa.
        NS-3 flows matched to users by ascending source IP.
      </p>
    </Panel>
  );
}
