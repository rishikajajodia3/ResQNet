import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Panel, Loading } from "./Panel.jsx";
import { api } from "../api.js";

const AXIS = { stroke: "#6b7480", fontSize: 11 };

export function TrainingResults() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.trainingSummary().then(setData).catch(setErr);
  }, []);

  if (err) {
    return (
      <Panel title="Multi-UAV training" tag={<span className="tag">surrogate</span>}>
        <p className="inline-note">
          No trained multi-UAV model yet. Run{" "}
          <span className="mono">python ml/train_multiuav.py</span> (surrogate; held-out
          scenarios excluded from training).
        </p>
      </Panel>
    );
  }
  if (!data) return <Panel title="Multi-UAV training"><Loading /></Panel>;

  const cfg = data.config || {};
  const eps = data.episodes || [];
  const chart = eps.map((e) => ({
    episode: e.episode,
    reward: e.ep_reward,
    coverage: e.coverage_pct,
    sinr: e.mean_best_sinr_db,
  }));

  return (
    <Panel
      title="Multi-UAV training"
      tag={<span className="tag">surrogate / free-space</span>}
      hint={`${data.num_episodes} episodes · seed ${cfg.base_seed ?? "?"}`}
    >
      <p className="inline-note" style={{ marginTop: 0 }}>
        Train scenarios: <span className="mono">{(cfg.train?.scenarios || []).join(", ")}</span>.
        Held out (never trained on):{" "}
        <span className="mono">{(cfg.eval_scenarios_held_out || []).join(", ")}</span>.
        RF model during training: {data.rf_model}
      </p>
      <div className="charts-grid">
        {[
          ["Episode reward", "reward", "#4c8dff"],
          ["Coverage % (surrogate)", "coverage", "#5ab0a6"],
          ["Mean SINR dB (surrogate)", "sinr", "#a98bd6"],
        ].map(([title, key, color]) => (
          <div className="chart-box" key={key}>
            <h4>{title}</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={chart} margin={{ top: 4, right: 10, bottom: 0, left: -18 }}>
                <CartesianGrid stroke="#2a323d" strokeDasharray="2 3" />
                <XAxis dataKey="episode" tick={AXIS} stroke="#2a323d" />
                <YAxis tick={AXIS} stroke="#2a323d" width={48} />
                <Tooltip
                  contentStyle={{ background: "#161b22", border: "1px solid #2a323d", borderRadius: 6, fontSize: 12 }}
                  labelStyle={{ color: "#9aa4b2" }}
                />
                <Line type="monotone" dataKey={key} stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
      <div style={{ overflowX: "auto", marginTop: 10 }}>
        <table className="data">
          <thead>
            <tr>
              <th>Episode</th><th>Scenario</th><th>UAVs</th><th>Users</th>
              <th>Reward</th><th>Coverage %</th><th>Connected %</th>
            </tr>
          </thead>
          <tbody>
            {eps.slice(-12).map((e) => (
              <tr key={e.episode}>
                <td>{e.episode}</td>
                <td>{e.scenario}</td>
                <td>{e.num_uavs}</td>
                <td>{e.num_users}</td>
                <td>{typeof e.ep_reward === "number" ? e.ep_reward.toFixed(3) : e.ep_reward}</td>
                <td>{e.coverage_pct}</td>
                <td>{e.connected_pct}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
