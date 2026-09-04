import { Panel } from "./Panel.jsx";
import { UAV_COLORS, stepUavPositions, fmt } from "../multiuav.js";

function Kpi({ label, value, unit, sub, accent }) {
  return (
    <div className="kpi" style={accent ? { borderLeft: `3px solid ${accent}` } : undefined}>
      <div className="k-label">{label}</div>
      <div className="k-value">
        {value}
        {unit && <span className="k-unit">{unit}</span>}
      </div>
      {sub && <div className="k-sub">{sub}</div>}
    </div>
  );
}

export function KpiCards({ traj, stepIndex }) {
  const step = traj?.trajectory?.[stepIndex];
  if (!step) return null;
  const nUav = traj.num_uavs || 1;
  const uavPos = stepUavPositions(step);
  const upu = step.users_per_uav || null;
  const multi = nUav > 1;

  return (
    <Panel
      title="PPO / Sionna metrics"
      tag={<span className="tag sionna">{traj.mode === "surrogate" ? "Surrogate" : "Sionna RT"}</span>}
      hint={`step ${step.step} of ${traj.num_steps}${traj.scenario ? " · " + traj.scenario : ""}`}
    >
      <div className="kpi-grid">
        <Kpi label="Coverage" value={fmt(step.coverage_percent, 1)} unit="%" />
        <Kpi label="Connected users" value={`${step.connected_users} / ${traj.num_users}`} />
        <Kpi label="Mean serving RSS" value={fmt(step.mean_rss_dbm)} unit="dBm" />
        <Kpi
          label="Mean SNR"
          value={fmt(step.mean_snr_db ?? step.mean_sinr_db)}
          unit="dB"
          sub="interference-free"
        />
        <Kpi
          label="Mean SINR"
          value={fmt(step.mean_sinr_db)}
          unit="dB"
          sub={multi ? "co-channel (all UAVs share 1 channel)" : "1 UAV → SINR = SNR"}
        />
        <Kpi label="Mean path gain" value={fmt(step.mean_path_gain_db)} unit="dB" />
        <Kpi label="Step reward" value={fmt(step.reward, 3)} />
      </div>

      {multi && (
        <>
          <h4 style={{ fontSize: 12, color: "var(--text-dim)", margin: "14px 0 6px" }}>
            Per-UAV
          </h4>
          <div className="kpi-grid">
            {uavPos.map((p, k) => (
              <Kpi
                key={k}
                accent={UAV_COLORS[k % UAV_COLORS.length]}
                label={`UAV ${k + 1}`}
                value={`${p[0].toFixed(0)}, ${p[1].toFixed(0)}, ${p[2].toFixed(0)}`}
                unit="m"
                sub={upu ? `serves ${upu["uav_" + (k + 1)] ?? 0} users` : undefined}
              />
            ))}
          </div>
        </>
      )}

      <p className="inline-note">
        Single run. {traj.mode === "surrogate"
          ? "Surrogate = free-space path loss (no buildings); not a Sionna result."
          : "Monte-Carlo ray tracing; sub-dB variation between runs near thresholds."}
      </p>
    </Panel>
  );
}
