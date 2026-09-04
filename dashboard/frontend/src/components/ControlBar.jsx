export function ControlBar({
  variant,
  setVariant,
  scenario,
  setScenario,
  scenarios,
  numUavs,
  setNumUavs,
  numUsers,
  setNumUsers,
  userCounts,
  uavCounts,
  steps,
  setSteps,
  mode,
  setMode,
  onRun,
  run,
  disabled,
}) {
  const status = run?.status;
  return (
    <div className="controlbar">
      <div className="field">
        <label>Scene</label>
        <select value={variant} onChange={(e) => setVariant(e.target.value)}>
          <option value="damaged">After disaster + PPO</option>
          <option value="baseline">Before disaster</option>
        </select>
      </div>

      <div className="field">
        <label>Scenario</label>
        <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
          <option value="">— (legacy 1-UAV / 10 users) —</option>
          {(scenarios || []).map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} [{s.split}]
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>UAVs</label>
        <select
          value={numUavs}
          onChange={(e) => setNumUavs(+e.target.value)}
          disabled={!scenario}
        >
          {(uavCounts || [1, 2, 3, 5]).map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Users</label>
        <select
          value={numUsers}
          onChange={(e) => setNumUsers(+e.target.value)}
          disabled={!scenario}
        >
          {(userCounts || [10, 20, 30, 50]).map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Wireless model</label>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="sionna">Sionna RT (ray tracing)</option>
          <option value="surrogate">Surrogate (free-space)</option>
        </select>
      </div>

      <div className="field">
        <label>PPO steps</label>
        <input
          type="number"
          min="1"
          max="50"
          value={steps}
          onChange={(e) => setSteps(Math.max(1, Math.min(50, +e.target.value || 1)))}
        />
      </div>

      <button className="primary" onClick={onRun} disabled={disabled}>
        {status === "running" || status === "queued" ? "Running…" : "Run experiment"}
      </button>

      <div className="run-state">
        <span className={`dot ${status || ""}`} />
        {!run && "no dashboard run yet"}
        {run && (
          <>
            <span className="mono">{run.run_id}</span>
            <span>— {run.status}</span>
            {run.error && <span style={{ color: "var(--bad)" }}>{run.error}</span>}
          </>
        )}
      </div>
    </div>
  );
}
