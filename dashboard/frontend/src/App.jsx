import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { Panel, Loading, ErrorState } from "./components/Panel.jsx";
import { PipelineDiagram } from "./components/PipelineDiagram.jsx";
import { ControlBar } from "./components/ControlBar.jsx";
import { Scene3D } from "./components/Scene3D.jsx";
import { KpiCards } from "./components/KpiCards.jsx";
import { PpoCharts } from "./components/PpoCharts.jsx";
import { PlaybackControls } from "./components/PlaybackControls.jsx";
import { ComparisonView } from "./components/ComparisonView.jsx";
import { PerUserTable } from "./components/PerUserTable.jsx";
import { Ns3Section } from "./components/Ns3Section.jsx";
import { TrainingResults } from "./components/TrainingResults.jsx";
import { ExperimentTable } from "./components/ExperimentTable.jsx";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [traj, setTraj] = useState(null);
  const [ns3, setNs3] = useState(null);
  const [latestSource, setLatestSource] = useState(null);
  const [loadErr, setLoadErr] = useState(null);
  const [loading, setLoading] = useState(true);

  const [scenarioList, setScenarioList] = useState([]);
  const [userCounts, setUserCounts] = useState([10, 20, 30, 50]);
  const [uavCounts, setUavCounts] = useState([1, 2, 3, 5]);

  const [variant, setVariant] = useState("damaged");
  const [scenario, setScenario] = useState("");
  const [numUavs, setNumUavs] = useState(3);
  const [numUsers, setNumUsers] = useState(30);
  const [steps, setSteps] = useState(5);
  const [mode, setMode] = useState("surrogate");
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  const [run, setRun] = useState(null);
  const pollRef = useRef(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setLoadErr(null);
    try {
      const [m, latest] = await Promise.all([api.sceneMeta(), api.latest()]);
      setMeta(m);
      setTraj(latest.ppo_trajectory);
      setNs3(latest.ns3_results);
      setLatestSource(latest.source);
      setStepIndex((latest.ppo_trajectory?.trajectory?.length || 1) - 1);
    } catch (e) {
      setLoadErr(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
    api
      .scenarios()
      .then((s) => {
        setScenarioList(s.scenarios || []);
        if (s.user_counts) setUserCounts(s.user_counts);
        if (s.uav_counts) setUavCounts(s.uav_counts);
      })
      .catch(() => {});
    return () => clearInterval(pollRef.current);
  }, [loadAll]);

  const startRun = useCallback(async () => {
    try {
      const opts = { mode, steps };
      if (scenario) {
        opts.scenario = scenario;
        opts.num_uavs = numUavs;
        opts.num_users = numUsers;
      }
      const { run_id } = await api.startRun(opts);
      setRun({ run_id, status: "queued" });
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const st = await api.runStatus(run_id);
          setRun(st);
          if (st.status === "succeeded" || st.status === "failed") {
            clearInterval(pollRef.current);
            if (st.status === "succeeded") await loadAll();
          }
        } catch (e) {
          setRun((r) => ({ ...(r || {}), status: "failed", error: String(e.message || e) }));
          clearInterval(pollRef.current);
        }
      }, 2000);
    } catch (e) {
      setRun({ run_id: "-", status: "failed", error: String(e.message || e) });
    }
  }, [mode, steps, scenario, numUavs, numUsers, loadAll]);

  const running = run?.status === "running" || run?.status === "queued";
  const nUav = traj?.num_uavs || 1;

  return (
    <div className="app">
      <header className="masthead">
        <span className="brand">ResQNet</span>
        <span className="tagline">
          AI-driven adaptive multi-UAV communication for disaster recovery
        </span>
        <span className="spacer" />
        <span className="source-note">
          {traj
            ? `${nUav} UAV · ${traj.num_users} users · ${traj.mode}${traj.scenario ? " · " + traj.scenario : ""} · source ${latestSource || "…"}`
            : "…"}
        </span>
      </header>

      <ControlBar
        variant={variant}
        setVariant={setVariant}
        scenario={scenario}
        setScenario={setScenario}
        scenarios={scenarioList}
        numUavs={numUavs}
        setNumUavs={setNumUavs}
        numUsers={numUsers}
        setNumUsers={setNumUsers}
        userCounts={userCounts}
        uavCounts={uavCounts}
        steps={steps}
        setSteps={setSteps}
        mode={mode}
        setMode={setMode}
        onRun={startRun}
        run={run}
        disabled={running}
      />

      {loadErr && (
        <Panel title="Startup">
          <ErrorState error={loadErr} />
          <p className="inline-note">
            Is the backend running? <span className="mono">python dashboard/backend/app.py</span>
          </p>
        </Panel>
      )}

      {loading && !meta && (
        <Panel title="Loading">
          <Loading label="Fetching pipeline outputs" />
        </Panel>
      )}

      {meta && traj && (
        <>
          <div className="grid-main">
            <div className="stack">
              <Scene3D variant={variant} meta={meta} traj={traj} stepIndex={stepIndex} ns3={ns3} />
              <PlaybackControls
                traj={traj}
                stepIndex={stepIndex}
                setStepIndex={setStepIndex}
                playing={playing}
                setPlaying={setPlaying}
              />
            </div>
            <div className="stack">
              <PipelineDiagram />
              <KpiCards traj={traj} stepIndex={stepIndex} />
            </div>
          </div>

          <div className="section">
            <ComparisonView traj={traj} />
          </div>

          <div className="section">
            <PpoCharts traj={traj} stepIndex={stepIndex} onStep={setStepIndex} />
          </div>

          <div className="section">
            <Ns3Section ns3={ns3} />
          </div>

          <div className="section">
            <PerUserTable traj={traj} stepIndex={stepIndex} ns3={ns3} />
          </div>

          <div className="section">
            <TrainingResults />
          </div>

          <div className="section">
            <ExperimentTable />
          </div>

          {run && (
            <div className="section">
              <Panel
                title="Last dashboard run"
                hint={run.snapshot_dir ? `snapshot: ${run.snapshot_dir}` : ""}
              >
                <div className="run-state" style={{ marginBottom: 8 }}>
                  <span className={`dot ${run.status}`} />
                  <span className="mono">{run.run_id}</span> — {run.status}
                  {run.mode ? ` · ${run.mode} · ${run.steps} steps` : ""}
                  {run.scenario ? ` · ${run.scenario} · ${run.num_uavs} UAV · ${run.num_users} users` : ""}
                </div>
                {run.error && (
                  <p style={{ color: "var(--bad)" }} className="mono">
                    {run.error}
                  </p>
                )}
                {run.log_tail?.length > 0 && (
                  <div className="log-box">{run.log_tail.join("\n")}</div>
                )}
              </Panel>
            </div>
          )}
        </>
      )}
    </div>
  );
}
