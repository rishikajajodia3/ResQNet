import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Panel } from "./Panel.jsx";
import { UAV_COLORS, CHART, C_OK, C_BAD } from "../multiuav.js";

const AXIS = CHART.axis;
const GRID = CHART.grid;
const TT = CHART.tooltip;

function UserBar({ title, unit, data, dataKey, colorKey, onPick, picked }) {
  return (
    <div className="chart-box">
      <h4>{title}{unit ? ` · ${unit}` : ""}</h4>
      {data.length === 0 ? (
        <div className="chart-empty">no per-user data for this step</div>
      ) : (
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -14 }}
            onClick={(e) => { if (e && typeof e.activeTooltipIndex === "number") onPick?.(e.activeTooltipIndex); }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 3" vertical={false} />
            <XAxis dataKey="label" tick={AXIS} stroke={GRID} interval={data.length > 24 ? 2 : 0} />
            <YAxis tick={AXIS} stroke={GRID} width={44} />
            <Tooltip {...TT} />
            <Bar dataKey={dataKey} isAnimationActive animationDuration={350} radius={[2, 2, 0, 0]}>
              {data.map((d, i) => (
                <Cell
                  key={i}
                  fill={d[colorKey]}
                  fillOpacity={picked == null || picked === i ? 1 : 0.4}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export function UserCharts({ traj, stepIndex, onSelectUser, selected }) {
  const step = traj?.trajectory?.[stepIndex];
  const users = step?.user_results || [];
  const data = users.map((u, i) => {
    const k = (u.serving_uav || 1) - 1;
    return {
      label: (u.user_id || `u${i + 1}`).replace("user_", "u"),
      snr: num(u.snr_db ?? u.sinr_db),
      sinr: num(u.sinr_db),
      rss: num(u.rss_dbm),
      color: UAV_COLORS[k % UAV_COLORS.length],
      connColor: u.connected ? C_OK : C_BAD,
    };
  });
  const picked = selected?.type === "user" ? selected.index : null;

  return (
    <Panel
      title="Per-user radio metrics"
      tag={<span className="tag sionna">{traj?.mode === "surrogate" ? "Surrogate" : "Sionna RT"}</span>}
      right={<span className="hint">step {step?.step ?? 0} · bar colour = serving UAV · click to inspect</span>}
    >
      <div className="charts-grid two">
        <UserBar title="SNR by user" unit="dB" data={data} dataKey="snr" colorKey="color"
          picked={picked} onPick={(i) => onSelectUser({ type: "user", index: i })} />
        <UserBar title="SINR by user (co-channel)" unit="dB" data={data} dataKey="sinr" colorKey="color"
          picked={picked} onPick={(i) => onSelectUser({ type: "user", index: i })} />
      </div>
      <p className="inline-note">
        SNR is interference-free; SINR includes co-channel interference from the other UAVs&rsquo;
        ray-traced received powers. For a single UAV the two charts are identical.
      </p>
    </Panel>
  );
}

function num(v) { return typeof v === "number" && Number.isFinite(v) ? +v.toFixed(2) : null; }
