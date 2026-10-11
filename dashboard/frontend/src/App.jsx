import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { Panel, Loading, ErrorState } from "./components/Panel.jsx";
import { PipelineDiagram } from "./components/PipelineDiagram.jsx";
import { ControlBar } from "./components/ControlBar.jsx";
import { RunStages } from "./components/RunStages.jsx";
import { KpiStrip } from "./components/KpiStrip.jsx";
import { Scene3D } from "./components/Scene3D.jsx";
import { UavFleet } from "./components/UavFleet.jsx";
import { PpoCharts } from "./components/PpoCharts.jsx";
import { UserCharts } from "./components/UserCharts.jsx";
import { PlaybackControls } from "./components/PlaybackControls.jsx";
import { ComparisonView } from "./components/ComparisonView.jsx";
import { ResearchFindings } from "./components/ResearchFindings.jsx";
import { PerUserTable } from "./components/PerUserTable.jsx";
import { Ns3Section } from "./components/Ns3Section.jsx";
import { TrainingResults } from "./components/TrainingResults.jsx";
import { ExperimentTable } from "./components/ExperimentTable.jsx";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [traj, setTraj] = useState(null);
  const [ns3, setNs3] = useState(null);
  const [latestSource, setLatestSource] = useState(null);
  const [health, setHealth] = useState(null);
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
  const [interference, setInterference] = useState("co_channel");

  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState(null); // {type:'uav'|'user', index}

  const [run, setRun] = useState(null);
  const pollRef = useRef(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setLoadErr(null);
    try {
      const [m, latest, h] = await Promise.all([
        api.sceneMeta(),
        api.latest(),
        api.health().catch(() => null),
      ]);
      setMeta(m);
      setHealth(h);
      setTraj(latest.ppo_trajectory);
      setNs3(latest.ns3_results);
      setLatestSource(latest.source);
      setStepIndex((latest.ppo_trajectory?.trajectory?.length || 1) - 1);
      setSelected(null);
    } catch (e) {
      setLoadErr(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
    api.scenarios().then((s) => {
      setScenarioList(s.scenarios || []);
      if (s.user_counts) setUserCounts(s.user_counts);
      if (s.uav_counts) setUavCounts(s.uav_counts);
    }).catch(() => {});
    return () => clearInterval(pollRef.current);
  }, [loadAll]);

  const startRun = useCallback(async () => {
    try {
      const opts = { mode, steps };
      if (scenario) {
        opts.scenario = scenario;
        opts.num_uavs = numUavs;
        opts.num_users = numUsers;
        opts.interference_model = interference;
      }
      const { run_id } = await api.startRun(opts);
      setRun({ run_id, status: "queued", scenario: scenario || null, num_uavs: numUavs, num_users: numUsers, mode, steps, log_tail: [] });
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
      }, 1800);
    } catch (e) {
      setRun({ run_id: "-", status: "failed", error: String(e.message || e) });
    }
  }, [mode, steps, scenario, numUavs, numUsers, interference, loadAll]);

  const running = run?.status === "running" || run?.status === "queued";
  const nUav = traj?.num_uavs || 1;
  const step = traj?.trajectory?.[stepIndex];

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand-block">
          <span className="brand">ResQNet</span>
          <span className="subtitle">Disaster Communication Digital Twin</span>
        </div>
        <span className="spacer" />
        <div className="head-meta">
          <div className="item">
            <span className="lbl">Scenario</span>
            <span className="val">{traj?.scenario || meta?.scenario || "legacy"}</span>
          </div>
          <div className="item">
            <span className="lbl">System</span>
            <span className="val">
              <span className={`sdot ${health?.status === "ok" ? "ok" : "warn"}`} />
              {health?.status === "ok" ? "Online" : "Connecting"}
            </span>
          </div>
          <div className="item">
            <span className="lbl">Run</span>
            <span className="val">
              <span className={`sdot ${run ? run.status : "accent"}`} />
              {run ? run.status : "Idle"}
            </span>
          </div>
        </div>
      </header>

      <div className="content">
        {loadErr && (
          <Panel title="Startup">
            <ErrorState error={loadErr} hint="Start the backend:  python dashboard/backend/app.py" />
          </Panel>
        )}

        {loading && !meta && (
          <Panel title="Initialising"><Loading label="Fetching digital-twin state" /></Panel>
        )}

        {meta && traj && (
          <>
            <KpiStrip traj={traj} ns3={ns3} stepIndex={stepIndex} />

            <div className="mission-grid">
              <div className="stack">
                <Scene3D
                  variant={variant} meta={meta} traj={traj} stepIndex={stepIndex} ns3={ns3}
                  selected={selected} onSelect={setSelected}
                />
                <PlaybackControls
                  traj={traj} stepIndex={stepIndex} setStepIndex={setStepIndex}
                  playing={playing} setPlaying={setPlaying}
                />
              </div>

              <div className="stack">
                <ControlBar
                  variant={variant} setVariant={setVariant}
                  scenario={scenario} setScenario={setScenario} scenarios={scenarioList}
                  numUavs={numUavs} setNumUavs={setNumUavs}
                  numUsers={numUsers} setNumUsers={setNumUsers}
                  userCounts={userCounts} uavCounts={uavCounts}
                  steps={steps} setSteps={setSteps}
                  mode={mode} setMode={setMode}
                  interference={interference} setInterference={setInterference}
                  onRun={startRun} run={run} disabled={running}
                />
                <RunStages run={run} />
                <UavFleet
                  traj={traj} stepIndex={stepIndex} ns3={ns3}
                  selected={selected} onSelect={setSelected}
                />
              </div>
            </div>

            <PipelineDiagram />

            <UserCharts
              traj={traj} stepIndex={stepIndex}
              selected={selected} onSelectUser={setSelected}
            />

            <div className="row-2">
              <PpoCharts traj={traj} stepIndex={stepIndex} onStep={setStepIndex} />
              <ComparisonView traj={traj} ns3={ns3} />
            </div>

            <ResearchFindings traj={traj} ns3={ns3} />

            <Ns3Section ns3={ns3} selected={selected} onSelectUser={setSelected} />

            <PerUserTable
              traj={traj} stepIndex={stepIndex} ns3={ns3}
              selected={selected} onSelectUser={setSelected}
            />

            <div className="row-2">
              <TrainingResults />
              <ExperimentTable />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
