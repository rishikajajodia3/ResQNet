import { useEffect, useState } from "react";
import { Panel, Loading } from "./Panel.jsx";
import { api } from "../api.js";

const f = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "—");

export function ExperimentTable() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.experiments().then(setData).catch(setErr);
  }, []);

  if (err) return null;
  if (!data) return <Panel title="Experiment comparison"><Loading /></Panel>;

  const rows = data.experiments || [];
  return (
    <Panel
      title="Experiment comparison"
      hint="baselines vs PPO · all values from real runs"
    >
      {rows.length === 0 ? (
        <p className="inline-note" style={{ marginTop: 0 }}>
          No experiments recorded yet. Run{" "}
          <span className="mono">python ml/evaluate_multiuav.py --scenario … --num-uavs …</span>{" "}
          to populate <span className="mono">data/experiments/</span>.
        </p>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Experiment</th><th>Scenario</th><th>Split</th><th>UAVs</th><th>Users</th>
                <th>RF</th><th>Placement</th>
                <th>Coverage %</th><th>Connected</th><th>Mean SINR dB</th>
                <th>Move m</th><th>NS-3 delivering</th><th>Thpt Mbps</th><th>Delay ms</th>
              </tr>
            </thead>
            <tbody>
              {rows.flatMap((exp) =>
                (exp.rows || []).map((r, i) => (
                  <tr key={exp.experiment_id + "-" + i}>
                    <td>{i === 0 ? exp.experiment_id : ""}</td>
                    <td>{r.scenario ?? exp.scenario}</td>
                    <td>{r.split ?? exp.split ?? "—"}</td>
                    <td>{r.num_uavs ?? exp.num_uavs}</td>
                    <td>{r.num_users ?? exp.num_users}</td>
                    <td>{r.rf_model ?? exp.mode ?? "—"}</td>
                    <td>{r.placement}</td>
                    <td>{f(r.coverage_pct, 1)}</td>
                    <td>{r.connected_users != null ? `${r.connected_users}/${r.num_users ?? exp.num_users}` : "—"}</td>
                    <td>{f(r.mean_best_sinr_db)}</td>
                    <td>{f(r.movement_m, 1)}</td>
                    <td>{r.ns3_delivering_flows != null ? `${r.ns3_delivering_flows}/${r.ns3_total_flows}` : "—"}</td>
                    <td>{f(r.ns3_avg_throughput_mbps, 4)}</td>
                    <td>{f(r.ns3_avg_delay_ms, 3)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
