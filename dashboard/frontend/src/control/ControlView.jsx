import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { api } from "../api.js";
import { Panel } from "../components/Panel.jsx";
import { ControlScene } from "./ControlScene.jsx";
import { PolicyBadge, PolicyPanel } from "./PolicyPanel.jsx";
import {
  uavColor, historySeries, beforeAfter, decisionRows, fmtNum, fmtSigned, fmtPos, isActive, METRIC_LABELS,
} from "./model.js";
import "./control.css";

const AXIS = { stroke: "#98a2b3", fontSize: 10 };
const TOOLTIP = { contentStyle: { background: "#fff", border: "1px solid #d9e0e8", fontSize: 11 } };

function ProvenanceBanner({ state }) {
  const p = state?.provenance;
  const mock = p?.mock;
  return (
    <div className={`ctl-banner${mock ? " mock" : ""}`}>
      <b>SIMULATED</b>
      {state?.provenance && <PolicyBadge state={state} />}
      <span>
        Digital-twin simulation only: radio links from{" "}
        <b>{p?.rf || "Sionna RT ray tracing"}</b>, network performance from <b>{p?.network || "ns-3"}</b>.
        No physical UAV, sensor or live telemetry is connected.
        {mock && <> <b>MOCK network data</b>: values are a deterministic stand-in, not ns-3 measurements.</>}
      </span>
    </div>
  );
}

function Settings({ cfg, presets, form, setForm, disabled, ppoModels }) {
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const w = form.weights;
  const ckpts = (ppoModels || []).flatMap((r) => r.checkpoints.map((c) => `${r.run_id}/${c}`));
  return (
    <div className="ctl-settings">
      <label>Policy
        <select value={form.policy} disabled={disabled} onChange={(e) => set("policy", e.target.value)}>
          <option value="rule_based">Rule-based 3D controller (default)</option>
          <option value="ppo" disabled={!ckpts.length}>PPO 3D policy{ckpts.length ? "" : " (no trained run)"}</option>
        </select>
      </label>
      {form.policy === "ppo" && (
        <label>PPO checkpoint
          <select value={form.ppo_checkpoint} disabled={disabled} onChange={(e) => set("ppo_checkpoint", e.target.value)}>
            <option value="">— choose —</option>
            {ckpts.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
      )}
      <label>Site
        <select value={form.site} disabled={disabled} onChange={(e) => set("site", e.target.value)}>
          <option value="urban_core">urban_core (held-out evaluation site)</option>
          <option value="val_a">val_a (validation window)</option>
        </select>
      </label>
      <label>Channel in ns-3
        <select value={form.channel_mode} disabled={disabled} onChange={(e) => set("channel_mode", e.target.value)}>
          <option value="obstruction">Obstruction-aware (Sionna loss per link)</option>
          <option value="distance">Distance-only (original ns-3 channel)</option>
        </select>
      </label>
      <label>Priority
        <select value={form.preset} disabled={disabled} onChange={(e) => set("preset", e.target.value)}>
          {Object.keys(presets || {}).map((p) => <option key={p} value={p}>{p.replace("_", " ")}</option>)}
          <option value="custom">custom weights</option>
        </select>
      </label>
      <div className="ctl-weights">
        {["coverage", "pdr", "fairness", "latency"].map((k) => (
          <label key={k}>{k}
            <input type="number" step="0.05" min="0" disabled={disabled || form.preset !== "custom"}
                   value={form.preset === "custom" ? w[k] : presets?.[form.preset]?.[k] ?? ""}
                   onChange={(e) => set("weights", { ...w, [k]: Math.max(0, +e.target.value) })} />
          </label>
        ))}
      </div>
      <label>Control intervals
        <input type="number" min="1" max="40" value={form.max_steps} disabled={disabled}
               onChange={(e) => set("max_steps", +e.target.value)} />
      </label>
      <label>Start altitude (m)
        <input type="number" min="15" max="60" value={form.initial_altitude_m} disabled={disabled}
               onChange={(e) => set("initial_altitude_m", +e.target.value)} />
      </label>
      <label>ns-3 traffic per measurement (s)
        <input type="number" min="1" max="8" step="0.5" value={form.traffic_seconds} disabled={disabled}
               onChange={(e) => set("traffic_seconds", +e.target.value)} />
      </label>
      <label className="chk">
        <input type="checkbox" checked={form.events} disabled={disabled} onChange={(e) => set("events", e.target.checked)} />
        scripted event: new survivors appear{cfg?.events?.[0] ? ` at interval ${cfg.events[0].step}` : ""}
      </label>
      <details className="ctl-adv">
        <summary>Test modes</summary>
        <label>Radio model
          <select value={form.rf_mode} disabled={disabled} onChange={(e) => set("rf_mode", e.target.value)}>
            <option value="sionna">Sionna RT (buildings)</option>
            <option value="surrogate">free-space surrogate (no buildings, fast)</option>
          </select>
        </label>
        <label>Network
          <select value={form.network} disabled={disabled} onChange={(e) => set("network", e.target.value)}>
            <option value="ns3">ns-3 simulation</option>
            <option value="mock">MOCK (UI test only, not ns-3)</option>
          </select>
        </label>
      </details>
    </div>
  );
}

function StatusLine({ state }) {
  if (!state || state.status === "idle") return <div className="ctl-status">No control session. Press Start.</div>;
  const t = state.timing || {};
  return (
    <div className="ctl-status">
      <span className={`pill st-${state.status}`}>{state.status}</span>
      <span>interval <b>{state.step ?? 0}</b> / {state.max_steps ?? "?"}</span>
      <span>phase <b>{state.phase}</b>{state.phase_detail ? ` — ${state.phase_detail}` : ""}</span>
      <span className="dim">{state.run_id}</span>
      {t.rf_calls > 0 && (
        <span className="dim">Sionna {t.rf_calls}× {fmtNum(t.rf_s, 0)} s · ns-3 {t.network_calls}× {fmtNum(t.network_s, 0)} s</span>
      )}
      {state.error && <span className="err">{state.error}</span>}
    </div>
  );
}

function Kpis({ state }) {
  const rows = beforeAfter(state);
  if (!rows.length) return null;
  return (
    <div className="ctl-kpis">
      {rows.map((r) => (
        <div key={r.key} className="ctl-kpi">
          <span className="lbl">{r.label}</span>
          <span className="val">{fmtNum(r.after, r.digits)}</span>
          <span className={`d ${r.verdict}`}>{fmtSigned(r.delta, r.digits)} vs start</span>
        </div>
      ))}
    </div>
  );
}

function Fleet({ state, selected, onSelect }) {
  return (
    <div className="ctl-xscroll"><table className="ctl-table fleet-t">
      <thead>
        <tr><th>UAV</th><th>x</th><th>y</th><th>altitude</th><th>status</th><th>users</th>
          <th>below target</th><th>mean PDR</th><th>latency</th></tr>
      </thead>
      <tbody>
        {(state?.uavs || []).map((u, k) => (
          <tr key={u.id} className={selected === k ? "sel" : ""} onClick={() => onSelect(selected === k ? null : k)}>
            <td><span className="dot" style={{ background: uavColor(k) }} />{u.id}</td>
            <td>{u.position[0].toFixed(0)}</td>
            <td>{u.position[1].toFixed(0)}</td>
            <td><b>{u.position[2].toFixed(0)} m</b></td>
            <td><span className={`pill st-${u.status}`}>{u.status}</span></td>
            <td>{u.assigned_users ?? "—"}</td>
            <td>{u.poor_users ?? "—"}{u.no_delivery_users ? ` (${u.no_delivery_users} none)` : ""}</td>
            <td>{fmtNum(u.mean_pdr, 3)}</td>
            <td>{u.mean_latency_ms == null ? "—" : `${u.mean_latency_ms.toFixed(0)} ms`}</td>
          </tr>
        ))}
      </tbody>
    </table></div>
  );
}

function Inspector({ state, k, onClose }) {
  const u = state?.uavs?.[k];
  if (!u) return null;
  const decs = decisionRows(state, { uavId: u.id }).slice(0, 12);
  return (
    <Panel title={`Inspect ${u.id}`} right={<button className="btn ghost" onClick={onClose}>close</button>}>
      <div className="ctl-insp">
        <span>position (x, y, z)</span><b>{fmtPos(u.position)}</b>
        <span>altitude</span><b>{u.position[2].toFixed(1)} m</b>
        <span>status</span><b>{u.status}{u.last_action ? ` (last: ${u.last_action})` : ""}</b>
        <span>diagnosis</span><span>{u.diagnosis || "—"}</span>
        <span>last decision</span><span>{u.reason}</span>
        <span>assigned survivors</span><b>{u.assigned_users} ({u.poor_users} below PDR target)</b>
        <span>mean PDR / latency</span><b>{fmtNum(u.mean_pdr, 3)} / {u.mean_latency_ms == null ? "—" : `${u.mean_latency_ms.toFixed(1)} ms`}</b>
        <span>service radius (measured)</span><b>{u.footprint_m ? `${u.footprint_m} m` : "—"}</b>
      </div>
      <h4 className="ctl-sub">Decisions for {u.id}</h4>
      <DecisionTable rows={decs} compact />
    </Panel>
  );
}

function DecisionTable({ rows, compact }) {
  if (!rows.length) return <div className="dim">No decisions yet.</div>;
  return (
    <div className="ctl-scroll">
      <table className="ctl-table dec">
        <thead>
          <tr><th>int.</th>{!compact && <th>UAV</th>}<th>action</th><th>outcome</th><th>reason</th>
            <th>pred. ΔJ</th><th>meas. ΔJ</th><th>Δ cov</th><th>Δ PDR</th><th>Δ worst10</th><th>Δ lat (ms)</th></tr>
        </thead>
        <tbody>
          {rows.map((d, i) => {
            const m = d.measured_delta || {};
            return (
              <tr key={`${d.step}-${d.uav_id}-${i}`} className={`o-${d.outcome}`}>
                <td>{d.step}</td>
                {!compact && <td>{d.uav_id}</td>}
                <td><b>{d.action}</b>{d.to ? <div className="dim">{fmtPos(d.from)}→{fmtPos(d.to)}</div> : null}</td>
                <td><span className={`pill o-${d.outcome}`}>{d.outcome}</span></td>
                <td className="reason">{d.reason}{d.result ? <div className="dim">{d.result}</div> : null}</td>
                <td>{fmtSigned(d.predicted_delta_objective, 4)}</td>
                <td>{fmtSigned(m.objective, 4)}</td>
                <td>{fmtSigned(m.coverage, 3)}</td>
                <td>{fmtSigned(m.pdr, 3)}</td>
                <td>{fmtSigned(m.worst_decile_pdr, 3)}</td>
                <td>{fmtSigned(m.mean_latency_ms, 1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Charts({ state }) {
  const data = historySeries(state);
  const K = state?.uavs?.length || 0;
  const events = state?.events || [];
  if (data.length < 1) return null;
  const marks = events.map((e) => <ReferenceLine key={`e${e.step}`} x={e.step} stroke="#b7791f" strokeDasharray="3 3" />);
  return (
    <div className="ctl-charts">
      <div>
        <h4 className="ctl-sub">Objective and delivery (measured, ns-3)</h4>
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={data}>
            <CartesianGrid stroke="#e5eaf0" />
            <XAxis dataKey="step" {...AXIS} />
            <YAxis {...AXIS} domain={["auto", "auto"]} />
            <Tooltip {...TOOLTIP} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {marks}
            <Line dataKey="objective" name="J" stroke="#111827" strokeWidth={2} dot />
            <Line dataKey="trial_objective" name="J of reverted trial" stroke="#b42318" strokeWidth={0}
                  dot={{ r: 4, fill: "#b42318", stroke: "#b42318", strokeWidth: 1 }} isAnimationActive={false} />
            <Line dataKey="coverage" stroke="#157347" dot={false} />
            <Line dataKey="pdr" name="PDR" stroke="#1d4ed8" dot={false} />
            <Line dataKey="worst_decile_pdr" name="worst-decile PDR" stroke="#7c3aed" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div>
        <h4 className="ctl-sub">Mean per-user latency (ms, measured)</h4>
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={data}>
            <CartesianGrid stroke="#e5eaf0" />
            <XAxis dataKey="step" {...AXIS} />
            <YAxis {...AXIS} />
            <Tooltip {...TOOLTIP} />
            {marks}
            <Line dataKey="mean_latency_ms" name="latency" stroke="#b45309" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div>
        <h4 className="ctl-sub">UAV altitude per interval (m)</h4>
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={data}>
            <CartesianGrid stroke="#e5eaf0" />
            <XAxis dataKey="step" {...AXIS} />
            <YAxis {...AXIS} domain={[0, 70]} />
            <Tooltip {...TOOLTIP} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {Array.from({ length: K }).map((_, k) => (
              <Line key={k} dataKey={`z${k + 1}`} name={`UAV-${k + 1}`} stroke={uavColor(k)}
                    strokeWidth={2 + 2 * (K - 1 - k)} strokeOpacity={0.8} type="stepAfter" dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function BeforeAfter({ state }) {
  const rows = beforeAfter(state);
  if (!rows.length) return <div className="dim">Available after the first measurement.</div>;
  const bp = state.before.positions || [];
  const ap = state.after.positions || [];
  return (
    <div className="ctl-ba">
      <table className="ctl-table">
        <thead><tr><th>metric (ns-3 measured)</th><th>before</th><th>after</th><th>change</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>{r.label}</td><td>{fmtNum(r.before, r.digits)}</td><td>{fmtNum(r.after, r.digits)}</td>
              <td className={r.verdict}>{fmtSigned(r.delta, r.digits)}</td>
            </tr>
          ))}
          <tr><td>survivors</td><td>{state.before.num_users}</td><td>{state.after.num_users}</td>
            <td>{state.after.num_users !== state.before.num_users ? "changed by event" : ""}</td></tr>
        </tbody>
      </table>
      <table className="ctl-table">
        <thead><tr><th>UAV</th><th>before (x, y, z)</th><th>after (x, y, z)</th></tr></thead>
        <tbody>
          {ap.map((p, k) => (
            <tr key={k}><td><span className="dot" style={{ background: uavColor(k) }} />UAV-{k + 1}</td>
              <td>{fmtPos(bp[k])}</td><td><b>{fmtPos(p)}</b></td></tr>
          ))}
        </tbody>
      </table>
      <p className="dim small">
        "Before" is the first measured state of this session; "after" is the current accepted state. Both are
        single ns-3 simulations (RngRun {String(state.config?.ns3?.rng_runs ?? "")}) of{" "}
        {state.config?.ns3?.traffic_seconds} s of traffic, so small differences are within simulation noise.
        {state.after.num_users !== state.before.num_users && " The survivor set changed during the session, so before and after are not the same population."}
      </p>
    </div>
  );
}

const DEFAULT_FORM = {
  channel_mode: "obstruction", preset: "balanced",
  weights: { coverage: 0.35, pdr: 0.3, fairness: 0.15, latency: 0.2 },
  max_steps: 12, initial_altitude_m: 40, traffic_seconds: 3, events: true, rf_mode: "sionna", network: "ns3",
  policy: "rule_based", ppo_checkpoint: "", site: "urban_core",
};

export function ControlView() {
  const [state, setState] = useState(null);
  const [cfgInfo, setCfgInfo] = useState(null);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [selected, setSelected] = useState(null);
  const [err, setErr] = useState(null);
  const [showHolds, setShowHolds] = useState(false);
  const [busy, setBusy] = useState(false);
  const [ppoModels, setPpoModels] = useState([]);
  const lastKey = useRef("");

  useEffect(() => {
    api.controlConfig().then((c) => {
      setCfgInfo(c);
      const cf = c.config || {};
      setForm((f) => ({
        ...f,
        channel_mode: cf.channel_mode ?? f.channel_mode,
        max_steps: cf.max_steps ?? f.max_steps,
        initial_altitude_m: cf.initial_altitude_m ?? f.initial_altitude_m,
        traffic_seconds: cf.ns3?.traffic_seconds ?? f.traffic_seconds,
        preset: cf.objective?.preset ?? f.preset,
      }));
    }).catch((e) => setErr(String(e.message || e)));
    api.controlPpoModels().then((m) => setPpoModels(m.runs || [])).catch(() => setPpoModels([]));
  }, []);

  const poll = useCallback(async () => {
    try {
      const s = await api.controlState();
      const key = `${s.run_id}|${s.version}|${s.status}|${s.session?.process_alive}`;
      if (key !== lastKey.current) {
        lastKey.current = key;
        setState(s);
      }
      setErr(null);
    } catch (e) {
      setErr(String(e.message || e));
    }
  }, []);

  const active = isActive(state);
  useEffect(() => {
    poll();
    const id = setInterval(poll, active ? 1000 : 4000);
    return () => clearInterval(id);
  }, [poll, active]);

  const act = async (fn) => {
    setBusy(true);
    try {
      await fn();
      await poll();
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const start = () => act(() => {
    const body = {
      channel_mode: form.channel_mode, max_steps: form.max_steps, initial_altitude_m: form.initial_altitude_m,
      traffic_seconds: form.traffic_seconds, events: form.events, rf_mode: form.rf_mode, network: form.network,
    };
    if (form.preset === "custom") body.weights = form.weights;
    else body.preset = form.preset;
    body.site = form.site;
    if (form.policy === "ppo") {
      body.policy = "ppo";
      body.ppo_checkpoint = form.ppo_checkpoint;
    }
    setSelected(null);
    return api.controlStart(body);
  });

  const paused = state?.status === "paused" || state?.status === "pausing";
  const decisions = useMemo(() => decisionRows(state, { includeHolds: showHolds }), [state, showHolds]);

  return (
    <div className="ctl">
      <ProvenanceBanner state={state} />
      {err && <div className="ctl-err">Backend: {err}</div>}

      <Panel title="Closed-loop control" right={
        <div className="ctl-btns">
          <button className="btn primary" disabled={busy || active} onClick={start}>Start</button>
          <button className="btn" disabled={busy || !active || paused} onClick={() => act(api.controlPause)}>Pause</button>
          <button className="btn" disabled={busy || !active || !paused} onClick={() => act(api.controlResume)}>Resume</button>
          <button className="btn danger" disabled={busy || !state || state.status === "idle"}
                  onClick={() => act(api.controlReset)}>Reset</button>
        </div>
      }>
        <StatusLine state={state} />
        <Settings cfg={cfgInfo?.config} presets={cfgInfo?.presets} form={form} setForm={setForm} disabled={active}
                  ppoModels={ppoModels} />
        {form.policy === "ppo" ? (
          <p className="dim small">
            PPO mode: each interval the trained policy reads the last ns-3 measurement (per-user PDR) and the Sionna
            obstruction at the current positions, picks stay/±x/±y/up/down for every UAV (unsafe moves masked with the
            same safety rules), the UAVs move, Sionna and ns-3 are re-run and the new measurement is the next input.
            PPO moves are not reverted. The objective J below is still computed from ns-3 for comparison; the policy
            was trained on predicted PDR / coverage / worst-10 % PDR, not on J.
          </p>
        ) : (
        <p className="dim small">
          Each interval: observe (Sionna + ns-3) → evaluate the objective and per-UAV diagnostics → each UAV that
          needs to move independently picks up/down/±x/±y (safety-checked) → move → re-simulate → keep if the
          measured objective holds, otherwise fly back. One interval takes roughly 30–60 s with Sionna + ns-3.
          Pause takes effect at the next phase boundary.
        </p>
        )}
      </Panel>

      {state?.uavs && (
        <>
          <Kpis state={state} />
          <div className="ctl-main">
            <Panel title="3D digital twin (simulated)" flush right={
              <span className="dim small">
                {state.channel_mode === "obstruction" ? "ns-3: obstruction-aware" : "ns-3: distance-only"} ·{" "}
                {state.users?.length} survivors · click a UAV to inspect
              </span>
            }>
              <ControlScene state={state} selected={selected} onSelect={setSelected} />
            </Panel>
            <div className="ctl-side">
              <Panel title="UAVs" flush><Fleet state={state} selected={selected} onSelect={setSelected} /></Panel>
              {selected !== null && <Inspector state={state} k={selected} onClose={() => setSelected(null)} />}
              {(state.events?.length > 0 || state.pending_events?.length > 0) && (
                <Panel title="Scenario events">
                  {state.events.map((e, i) => <div key={i}>interval {e.step}: {e.description}</div>)}
                  {state.pending_events.map((e, i) => <div key={`p${i}`} className="dim">interval {e.step}: {e.description} (pending)</div>)}
                </Panel>
              )}
            </div>
          </div>

          <Panel title="Network metrics over time"><Charts state={state} /></Panel>

          <Panel title="Policy, positions and actions"><PolicyPanel state={state} /></Panel>

          <div className="ctl-two">
            <Panel title="Before vs after"><BeforeAfter state={state} /></Panel>
            <Panel title="Objective">
              <div className="ctl-obj">
                J = {fmtNum(state.weights?.coverage, 2)}·coverage + {fmtNum(state.weights?.pdr, 2)}·PDR +{" "}
                {fmtNum(state.weights?.fairness, 2)}·worst-decile PDR − {fmtNum(state.weights?.latency, 2)}·latency/
                {state.config?.objective?.latency_ref_ms} ms
              </div>
              <ul className="small">
                <li>coverage = share of survivors with measured PDR ≥ {state.config?.objective?.coverage_pdr_threshold}</li>
                <li>all four terms are ns-3 measurements; candidates are ranked with Sionna link predictions
                  (latency is not predicted) plus a correction learned from earlier measured outcomes</li>
                <li>a move is kept only if measured J drops by less than {state.config?.acceptance_tolerance}</li>
                <li>safety: altitude {state.config?.safety?.min_altitude_m}–{state.config?.safety?.max_altitude_m} m,{" "}
                  ≥ {state.config?.safety?.building_clearance_m} m above roofs within {state.config?.safety?.clearance_radius_m} m,{" "}
                  ≥ {state.config?.safety?.min_separation_m} m between UAVs, inside the scenario boundary</li>
              </ul>
            </Panel>
          </div>

          <Panel title="Decision log" right={
            <label className="small"><input type="checkbox" checked={showHolds} onChange={(e) => setShowHolds(e.target.checked)} /> show holds</label>
          }>
            <DecisionTable rows={decisions} />
          </Panel>

          {state.session?.log_tail?.length > 0 && (
            <details className="ctl-log"><summary>Controller log ({state.session.out_dir})</summary>
              <pre>{state.session.log_tail.join("\n")}</pre>
            </details>
          )}
        </>
      )}
    </div>
  );
}

export { METRIC_LABELS };
