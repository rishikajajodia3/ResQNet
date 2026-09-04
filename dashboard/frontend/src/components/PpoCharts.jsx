import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Panel } from "./Panel.jsx";

const AXIS = { stroke: "#6b7480", fontSize: 11 };
const GRID = "#2a323d";

function MiniChart({ title, data, dataKey, unit, color, onStep, stepIndex }) {
  return (
    <div className="chart-box">
      <h4>
        {title}
        {unit ? ` (${unit})` : ""}
      </h4>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart
          data={data}
          margin={{ top: 4, right: 10, bottom: 0, left: -18 }}
          onClick={(e) => {
            if (e && typeof e.activeTooltipIndex === "number")
              onStep(e.activeTooltipIndex);
          }}
        >
          <CartesianGrid stroke={GRID} strokeDasharray="2 3" />
          <XAxis dataKey="step" tick={AXIS} stroke={GRID} />
          <YAxis tick={AXIS} stroke={GRID} width={48} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{
              background: "#161b22",
              border: "1px solid #2a323d",
              borderRadius: 6,
              fontSize: 12,
            }}
            labelStyle={{ color: "#9aa4b2" }}
          />
          {typeof stepIndex === "number" && data[stepIndex] && (
            <ReferenceLine x={data[stepIndex].step} stroke="#4c8dff" strokeDasharray="3 3" />
          )}
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={2}
            dot={{ r: 2 }}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PpoCharts({ traj, stepIndex, onStep }) {
  const data = (traj?.trajectory || []).map((s) => ({
    step: s.step,
    coverage: s.coverage_percent,
    connected: s.connected_users,
    reward: s.reward,
    rss: s.mean_rss_dbm,
    snr: s.mean_snr_db ?? s.mean_sinr_db,
    sinr: s.mean_sinr_db,
  }));
  const multi = (traj?.num_uavs || 1) > 1;
  return (
    <Panel
      title="PPO trajectory"
      tag={<span className="tag sionna">{traj?.mode === "surrogate" ? "Surrogate" : "Sionna RT"}</span>}
      hint="click a point to inspect that step"
    >
      <div className="charts-grid">
        <MiniChart title="Coverage vs step" unit="%" data={data} dataKey="coverage"
          color="#5ab0a6" onStep={onStep} stepIndex={stepIndex} />
        <MiniChart title="Reward vs step" data={data} dataKey="reward"
          color="#4c8dff" onStep={onStep} stepIndex={stepIndex} />
        <MiniChart title="Mean serving RSS vs step" unit="dBm" data={data} dataKey="rss"
          color="#c8944b" onStep={onStep} stepIndex={stepIndex} />
        <MiniChart title="Mean SNR vs step" unit="dB" data={data} dataKey="snr"
          color="#7fd6a6" onStep={onStep} stepIndex={stepIndex} />
        <MiniChart title="Mean SINR vs step" unit="dB" data={data} dataKey="sinr"
          color="#a98bd6" onStep={onStep} stepIndex={stepIndex} />
        <MiniChart title="Connected users vs step" data={data} dataKey="connected"
          color="#4c8dff" onStep={onStep} stepIndex={stepIndex} />
      </div>
      {multi && (
        <p className="inline-note">
          SNR is interference-free; SINR is co-channel (all {traj.num_uavs} UAVs on one
          channel). The SNR↔SINR gap is the modelled self-interference — the reason the
          policy is rewarded for spreading UAVs out.
        </p>
      )}
    </Panel>
  );
}
