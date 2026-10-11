import { Panel } from "./Panel.jsx";

// Experiment console. Controls map 1:1 to POST /api/run
// (mode, steps, scenario, num_uavs, num_users, interference_model).
// Seed and user distribution are shown for transparency but are fixed by
// the backend (seed 42; distribution defined per scenario) — they are
// displayed, not editable, so the UI never implies a control that does
// nothing.

export function ControlBar({
  variant, setVariant,
  scenario, setScenario, scenarios,
  numUavs, setNumUavs,
  numUsers, setNumUsers,
  userCounts, uavCounts,
  steps, setSteps,
  mode, setMode,
  interference, setInterference,
  onRun, run, disabled,
}) {
  const scen = (scenarios || []).find((s) => s.id === scenario);
  const legacy = !scenario;

  return (
    <Panel
      className="console"
      title="Experiment console"
      tag={<span className="tag muted">{legacy ? "legacy 1-UAV" : "multi-UAV"}</span>}
      right={
        <span className="hint">
          {run ? <><span className={`sdot ${run.status}`} /> {run.status}</> : "ready"}
        </span>
      }
    >
      <div className="console-grid">
        <div className="field">
          <label>Scene</label>
          <select value={variant} onChange={(e) => setVariant(e.target.value)}>
            <option value="damaged">After disaster (city_damaged)</option>
            <option value="baseline">Before disaster (city_baseline)</option>
          </select>
        </div>

        <div className="field">
          <label>Scenario</label>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            <option value="">— legacy: 1 UAV / 10 users —</option>
            {(scenarios || []).map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} · {s.split}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>UAVs</label>
          <select value={numUavs} onChange={(e) => setNumUavs(+e.target.value)} disabled={legacy}>
            {(uavCounts || [1, 2, 3, 5]).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>Users</label>
          <select value={numUsers} onChange={(e) => setNumUsers(+e.target.value)} disabled={legacy}>
            {(userCounts || [10, 20, 30, 50]).map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>RF model</label>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="sionna">Sionna RT — ray tracing (validation)</option>
            <option value="surrogate">Surrogate — free-space (fast)</option>
          </select>
        </div>

        <div className="field">
          <label>Interference model</label>
          <select
            value={interference}
            onChange={(e) => setInterference(e.target.value)}
            disabled={legacy}
          >
            <option value="co_channel">Co-channel (shared, full buffer)</option>
            <option value="orthogonal">Orthogonal (SINR = SNR)</option>
          </select>
        </div>

        <div className="field">
          <label>PPO steps</label>
          <input
            type="number" min="1" max="50" value={steps}
            onChange={(e) => setSteps(Math.max(1, Math.min(50, +e.target.value || 1)))}
          />
        </div>

        <div className="field">
          <label>User distribution</label>
          <input value={legacy ? "fixed (10 default)" : scen?.user_distribution || "—"} disabled readOnly />
        </div>

        <div className="field">
          <label>Seed</label>
          <input value="42 (fixed)" disabled readOnly />
        </div>
      </div>

      <div className="config-summary">
        {legacy ? (
          <>Legacy pipeline · <b>1 UAV</b> · <b>10 users</b> (DEFAULT_10_USERS) · RF <b>{mode}</b> · <b>{steps}</b> PPO steps</>
        ) : (
          <>
            <b>{scen?.name}</b> ({scen?.split}) · scene <b>{scen?.scene}</b> ·{" "}
            <b>{numUavs} UAV</b> / <b>{numUsers} users</b> ({scen?.user_distribution}) ·{" "}
            RF <b>{mode}</b> · interference <b>{interference}</b> · <b>{steps}</b> PPO steps · seed 42
          </>
        )}
      </div>

      <button className="run-btn" onClick={onRun} disabled={disabled}>
        {run?.status === "running" || run?.status === "queued" ? "Experiment running…" : "Run experiment"}
      </button>

      {run?.error && (
        <p className="inline-note" style={{ color: "var(--bad)", marginTop: 4 }}>{run.error}</p>
      )}
    </Panel>
  );
}
