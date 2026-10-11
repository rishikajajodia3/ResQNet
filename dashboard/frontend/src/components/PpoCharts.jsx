import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { Panel } from "./Panel.jsx";
import { uavMovement, UAV_COLORS, CHART, C_PRIMARY } from "../multiuav.js";

const AXIS = CHART.axis;
const GRID = CHART.grid;
const TT = CHART.tooltip;

function Mini({ title, unit, data, dataKey, color, onStep, stepIndex, allZero }) {
  return (
    <div className="chart-box">
      <h4>{title}{unit ? ` · ${unit}` : ""}</h4>
      {data.length < 2 ? (
        <div className="chart-empty">single-step run — no trajectory</div>
      ) : (
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={data} margin={{ top: 4, right: 10, bottom: 0, left: -16 }}
            onClick={(e) => { if (e && typeof e.activeTooltipIndex === "number") onStep(e.activeTooltipIndex); }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 3" />
            <XAxis dataKey="step" tick={AXIS} stroke={GRID} />
            <YAxis tick={AXIS} stroke={GRID} width={44} domain={allZero ? [0, 1] : ["auto", "auto"]} />
            <Tooltip {...TT} />
            {typeof stepIndex === "number" && data[stepIndex] && (
              <ReferenceLine x={data[stepIndex].step} stroke={C_PRIMARY} strokeDasharray="3 3" />
            )}
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2}
              dot={{ r: 2 }} activeDot={{ r: 4 }} isAnimationActive animationDuration={350} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export function PpoCharts({ traj, stepIndex, onStep }) {
  const steps = traj?.trajectory || [];
  const data = steps.map((s) => ({
    step: s.step,
    coverage: s.coverage_percent,
    connected: s.connected_users,
    reward: s.reward,
    rss: s.mean_rss_dbm,
    snr: s.mean_snr_db ?? s.mean_sinr_db,
    sinr: s.mean_sinr_db,
  }));
  const multi = (traj?.num_uavs || 1) > 1;
  const move = uavMovement(traj);
  const rewardFlat = data.length > 1 && data.every((d) => d.reward === data[0].reward);

  return (
    <Panel
      title="PPO trajectory metrics"
      tag={<span className="tag ppo">PPO</span>}
      right={<span className="hint">click a point to scrub the timeline</span>}
    >
      <div className="charts-grid">
        <Mini title="Coverage vs step" unit="%" data={data} dataKey="coverage" color="#1d4ed8" onStep={onStep} stepIndex={stepIndex} />
        <Mini title="Reward vs step" data={data} dataKey="reward" color="#1e3a8a" onStep={onStep} stepIndex={stepIndex} allZero={rewardFlat} />
        <Mini title="Connected users vs step" data={data} dataKey="connected" color="#2563eb" onStep={onStep} stepIndex={stepIndex} />
        <Mini title="Mean serving RSS vs step" unit="dBm" data={data} dataKey="rss" color="#64748b" onStep={onStep} stepIndex={stepIndex} />
        <Mini title="Mean SNR vs step" unit="dB" data={data} dataKey="snr" color="#3b82f6" onStep={onStep} stepIndex={stepIndex} />
        <Mini title="Mean SINR vs step" unit="dB" data={data} dataKey="sinr" color="#1e40af" onStep={onStep} stepIndex={stepIndex} />
      </div>

      {steps.length > 1 && (
        <div className="config-summary" style={{ marginTop: 12 }}>
          Net UAV displacement:{" "}
          {move.map((m, k) => (
            <b key={k} style={{ color: UAV_COLORS[k % UAV_COLORS.length] }}>
              UAV{k + 1} {m.toFixed(1)} m{k < move.length - 1 ? " · " : ""}
            </b>
          ))}
          {multi && (
            <> — SNR↔SINR gap is the modelled co-channel self-interference.</>
          )}
        </div>
      )}
    </Panel>
  );
}
