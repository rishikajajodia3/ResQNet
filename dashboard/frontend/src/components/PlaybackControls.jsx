import { useEffect, useRef } from "react";
import { Panel } from "./Panel.jsx";
import { UAV_COLORS, ACTION_NAMES, stepActions, fmt } from "../multiuav.js";

export function PlaybackControls({ traj, stepIndex, setStepIndex, playing, setPlaying }) {
  const steps = traj?.trajectory || [];
  const n = steps.length;
  const timer = useRef(null);

  useEffect(() => {
    clearInterval(timer.current);
    if (playing && n > 1) {
      timer.current = setInterval(() => {
        setStepIndex((i) => {
          if (i >= n - 1) { setPlaying(false); return i; }
          return i + 1;
        });
      }, 850);
    }
    return () => clearInterval(timer.current);
  }, [playing, n, setStepIndex, setPlaying]);

  if (n === 0) return null;
  const step = steps[stepIndex];
  const actions = stepActions(step);
  const atEnd = stepIndex >= n - 1;
  const label = stepIndex === 0 ? "INITIAL" : atEnd ? "FINAL" : `STEP ${step.step}`;

  return (
    <Panel
      title="PPO timeline"
      right={
        <span className="hint">
          {label} · reward {fmt(step.reward, 3)}
        </span>
      }
    >
      <div className="timeline-bar">
        <div className="tl-btns">
          <button onClick={() => { setPlaying(false); setStepIndex(0); }} aria-label="restart">⏮</button>
          <button onClick={() => { setPlaying(false); setStepIndex(Math.max(0, stepIndex - 1)); }} aria-label="previous">◀</button>
          <button
            className="primary"
            onClick={() => { if (atEnd) setStepIndex(0); setPlaying((p) => !p); }}
            aria-label={playing ? "pause" : "play"}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIndex(Math.min(n - 1, stepIndex + 1)); }} aria-label="next">▶</button>
        </div>

        <div className="tl-track">
          <div className="tl-marks">
            <span>init</span>
            {n > 3 && <span>{steps[Math.floor(n / 2)].step}</span>}
            <span>final</span>
          </div>
          <input
            type="range" min={0} max={n - 1} value={stepIndex}
            onChange={(e) => { setPlaying(false); setStepIndex(+e.target.value); }}
            aria-label="timeline"
          />
        </div>

        <div className="tl-readout">{stepIndex} / {n - 1}</div>
      </div>

      <div className="action-row">
        {actions.map((a, k) => {
          const moved = a && a !== 0;
          return (
            <div className="action-cell" key={k} style={{ borderLeftColor: UAV_COLORS[k % UAV_COLORS.length] }}>
              <div className="ac-label">UAV {String(k + 1).padStart(2, "0")}</div>
              <div className={`ac-value ${moved ? "moved" : "stay"}`}>
                {a === null || a === undefined ? "—" : ACTION_NAMES[a] ?? a}
              </div>
            </div>
          );
        })}
      </div>

      <p className="inline-note">
        Positions, actions and step metrics are read directly from{" "}
        <span className="mono">ppo_trajectory.json</span>. No intermediate values are interpolated.
      </p>
    </Panel>
  );
}
