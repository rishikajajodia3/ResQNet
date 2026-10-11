import { Panel, EmptyState } from "./Panel.jsx";
import { deriveFindings } from "../multiuav.js";

export function ResearchFindings({ traj, ns3 }) {
  const findings = deriveFindings(traj, ns3);

  return (
    <Panel
      title="Research findings"
      right="derived only from this run's returned values"
    >
      {findings.length === 0 ? (
        <EmptyState label="Run an experiment to generate findings." />
      ) : (
        <div className="findings">
          {findings.map((f, i) => (
            <div className={`finding ${f.kind}`} key={i}>
              <span className="f-mark" />
              <div className="f-title">{f.title}</div>
              <div className="f-body">{f.body}</div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
