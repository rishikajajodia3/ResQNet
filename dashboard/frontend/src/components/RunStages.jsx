import { Panel } from "./Panel.jsx";
import { STAGE_DEFS, inferStages } from "../multiuav.js";

export function RunStages({ run }) {
  const st = inferStages(run);
  const active = run && (run.status === "running" || run.status === "queued");

  return (
    <Panel
      title="Pipeline run"
      right={
        run ? (
          <>
            <span className={`sdot ${run.status}`} /> {run.run_id}
          </>
        ) : (
          "no run this session"
        )
      }
    >
      {!run && (
        <p className="inline-note" style={{ marginTop: 0 }}>
          Configure an experiment and select <b>Run experiment</b>. Stage progress
          is inferred from the live run log — no simulated percentages.
        </p>
      )}

      {run && (
        <>
          <div className="stages">
            {STAGE_DEFS.map((s, i) => {
              const state = st[s.id]; // idle | active | done | failed
              return (
                <div key={s.id} className={`stage ${state}`}>
                  <span className="st-num">{i + 1}</span>
                  <span className="st-name">{s.name}</span>
                  <span className="st-detail">
                    {state === "done" ? "complete"
                      : state === "active" ? "running"
                      : state === "failed" ? "failed"
                      : s.detail}
                  </span>
                </div>
              );
            })}
          </div>

          {active && <div className="indeterminate" />}

          <div className="inline-note" style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            {run.scenario
              ? <span>{run.scenario} · {run.num_uavs} UAV · {run.num_users} users · {run.mode}</span>
              : <span>legacy 1-UAV · {run.mode} · {run.steps} steps</span>}
            {run.snapshot_dir && <span className="mono">snapshot: {shortPath(run.snapshot_dir)}</span>}
          </div>

          {run.log_tail?.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary className="inline-note" style={{ cursor: "pointer" }}>
                Run log ({run.log_tail.length} lines)
              </summary>
              <div className="log-box" style={{ marginTop: 6 }}>{run.log_tail.join("\n")}</div>
            </details>
          )}
        </>
      )}
    </Panel>
  );
}

function shortPath(p) {
  return String(p).split("/").slice(-2).join("/");
}
