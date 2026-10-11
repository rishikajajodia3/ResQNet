import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Panel, Loading, EmptyState } from "./Panel.jsx";
import { api } from "../api.js";
import { CHART } from "../multiuav.js";

const AXIS = CHART.axis;
const GRID = CHART.grid;
const TT = CHART.tooltip;

export function TrainingResults() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.trainingSummary().then(setData).catch(setErr); }, []);

  if (err) {
    return (
      <Panel title="Multi-UAV training" tag={<span className="tag muted">surrogate</span>}>
        <EmptyState
          label="No trained multi-UAV model yet."
          action="python ml/train_multiuav.py"
        />
      </Panel>
    );
  }
  if (!data) return <Panel title="Multi-UAV training"><Loading /></Panel>;

  const cfg = data.config || {};
  const eps = data.episodes || [];
  // downsample for the line charts (keep the table full-fidelity)
  const stride = Math.max(1, Math.ceil(eps.length / 400));
  const chart = eps.filter((_, i) => i % stride === 0).map((e) => ({
    episode: e.episode, reward: e.ep_reward, coverage: e.coverage_pct, sinr: e.mean_best_sinr_db,
  }));

  return (
    <Panel
      title="Multi-UAV training"
      tag={<span className="tag muted">surrogate / free-space</span>}
      right={<span className="hint">{data.num_episodes} episodes · seed {cfg.base_seed ?? "?"}</span>}
    >
      <div className="config-summary" style={{ marginTop: 0 }}>
        Train: <b>{(cfg.train?.scenarios || []).join(", ")}</b> ·{" "}
        held out (never trained on): <b>{(cfg.eval_scenarios_held_out || []).join(", ")}</b> ·{" "}
        RF model: <b>{data.rf_model}</b>
      </div>

      <div className="charts-grid" style={{ marginTop: 12 }}>
        {[
          ["Episode reward", "reward", "#1e3a8a"],
          ["Coverage % (surrogate)", "coverage", "#1d4ed8"],
          ["Mean SINR dB (surrogate)", "sinr", "#2563eb"],
        ].map(([title, key, color]) => (
          <div className="chart-box" key={key}>
            <h4>{title}</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={chart} margin={{ top: 4, right: 10, bottom: 0, left: -16 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 3" />
                <XAxis dataKey="episode" tick={AXIS} stroke={GRID} />
                <YAxis tick={AXIS} stroke={GRID} width={44} />
                <Tooltip {...TT} />
                <Line type="monotone" dataKey={key} stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      <div className="table-scroll" style={{ marginTop: 10, maxHeight: 260 }}>
        <table className="data">
          <thead>
            <tr><th>Episode</th><th>Scenario</th><th>UAVs</th><th>Users</th><th>Reward</th><th>Coverage %</th><th>Connected %</th></tr>
          </thead>
          <tbody>
            {eps.slice(-16).reverse().map((e) => (
              <tr key={e.episode}>
                <td>{e.episode}</td><td>{e.scenario}</td><td>{e.num_uavs}</td><td>{e.num_users}</td>
                <td>{typeof e.ep_reward === "number" ? e.ep_reward.toFixed(3) : e.ep_reward}</td>
                <td>{e.coverage_pct}</td><td>{e.connected_pct}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="inline-note">
        Training uses the free-space surrogate for speed. Coverage/SINR here are surrogate
        values — <b>not</b> Sionna. Held-out scenarios are excluded from every training episode.
      </p>
    </Panel>
  );
}
