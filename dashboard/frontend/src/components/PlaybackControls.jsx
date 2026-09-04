import { useEffect, useRef } from "react";
import { Panel } from "./Panel.jsx";
import { UAV_COLORS, ACTION_NAMES, stepActions } from "../multiuav.js";

export function PlaybackControls({ traj, stepIndex, setStepIndex, playing, setPlaying }) {
  const steps = traj?.trajectory || [];
  const n = steps.length;
  const timer = useRef(null);

  useEffect(() => {
    clearInterval(timer.current);
    if (playing && n > 1) {
      timer.current = setInterval(() => {
        setStepIndex((i) => {
          if (i >= n - 1) {
            setPlaying(false);
            return i;
          }
          return i + 1;
        });
      }, 900);
    }
    return () => clearInterval(timer.current);
  }, [playing, n, setStepIndex, setPlaying]);

  if (n === 0) return null;
  const step = steps[stepIndex];
  const actions = stepActions(step);

  return (
    <Panel title="PPO playback" hint={`step ${step.step} / ${traj.num_steps}`}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <button
          className="primary"
          style={{ padding: "6px 14px" }}
          onClick={() => {
            if (stepIndex >= n - 1) setStepIndex(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          onClick={() => {
            setPlaying(false);
            setStepIndex(0);
          }}
          style={{ background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 12px", cursor: "pointer" }}
        >
          Reset
        </button>
        <input
          type="range"
          min={0}
          max={n - 1}
          value={stepIndex}
          onChange={(e) => {
            setPlaying(false);
            setStepIndex(+e.target.value);
          }}
          style={{ flex: 1, minWidth: 180 }}
        />
        <span className="mono" style={{ fontSize: 12, color: "var(--text-dim)" }}>
          reward {typeof step.reward === "number" ? step.reward.toFixed(3) : "—"}
        </span>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
        {actions.map((a, k) => (
          <div
            key={k}
            className="kpi"
            style={{ borderLeft: `3px solid ${UAV_COLORS[k % UAV_COLORS.length]}`, minWidth: 120 }}
          >
            <div className="k-label">UAV {k + 1}</div>
            <div className="k-value" style={{ fontSize: 15 }}>
              {a === null || a === undefined ? "—" : ACTION_NAMES[a] ?? a}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
