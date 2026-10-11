import { useEffect, useState } from "react";
import { Panel, Loading, EmptyState } from "./Panel.jsx";
import { api } from "../api.js";

const f = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "—");

export function ExperimentTable() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [openId, setOpenId] = useState(null);

  useEffect(() => { api.experiments().then(setData).catch(setErr); }, []);

  if (err) return null;
  if (!data) return <Panel title="Experiment history"><Loading /></Panel>;

  const rows = data.experiments || [];

  return (
    <Panel
      title="Experiment history"
      right={<span className="hint">fixed / random / PPO · all values from real runs</span>}
    >
      {rows.length === 0 ? (
        <EmptyState
          label="No experiments recorded yet."
          action="python ml/evaluate_multiuav.py --scenario blocked_urban --num-uavs 3 --num-users 30"
        />
      ) : (
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Experiment</th><th>Split</th><th>UAV</th><th>Users</th><th>RF</th><th>Placement</th>
                <th>Coverage %</th><th>Connected</th><th>SINR dB</th><th>Move m</th>
                <th>NS-3</th><th>Mbps</th><th>ms</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((exp) => {
                const open = openId === exp.experiment_id;
                const shown = open ? (exp.rows || []) : (exp.rows || []).filter((r) => r.placement === "ppo");
                return shown.map((r, i) => (
                  <tr
                    key={exp.experiment_id + "-" + r.placement}
                    className="clickable"
                    onClick={() => setOpenId(open ? null : exp.experiment_id)}
                  >
                    <td>{i === 0 ? exp.experiment_id : ""}</td>
                    <td>{r.split ?? exp.split ?? "—"}</td>
                    <td>{r.num_uavs ?? exp.num_uavs}</td>
                    <td>{r.num_users ?? exp.num_users}</td>
                    <td>{r.rf_model ?? exp.mode ?? "—"}</td>
                    <td>
                      <span className={`badge ${r.placement === "ppo" ? "" : "muted"}`}>{r.placement}</span>
                    </td>
                    <td>{f(r.coverage_pct, 1)}</td>
                    <td>{r.connected_users != null ? `${r.connected_users}/${r.num_users ?? exp.num_users}` : "—"}</td>
                    <td>{f(r.mean_best_sinr_db)}</td>
                    <td>{f(r.movement_m, 1)}</td>
                    <td>{r.ns3_delivering_flows != null ? `${r.ns3_delivering_flows}/${r.ns3_total_flows}` : "—"}</td>
                    <td>{f(r.ns3_avg_throughput_mbps, 4)}</td>
                    <td>{f(r.ns3_avg_delay_ms, 3)}</td>
                  </tr>
                ));
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="inline-note">
        {rows.length > 0 && "Row shows the PPO placement; click to expand fixed / random baselines. "}
        Populate more with <span className="mono">ml/evaluate_multiuav.py</span> (add <span className="mono">--ns3</span> for network columns).
      </p>
    </Panel>
  );
}
