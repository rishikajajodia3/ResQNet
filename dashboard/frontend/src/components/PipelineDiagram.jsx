import { Panel } from "./Panel.jsx";

const NODES = [
  { main: "3D disaster environment", kind: "scenario scene (city_damaged / baseline)" },
  { main: "Sionna RT", kind: "PathSolver ray tracing, K transmitters", cls: "sionna" },
  { main: "Wireless metrics", kind: "per-user × per-UAV: RSS / SNR / SINR / paths", cls: "sionna" },
  { main: "Multi-UAV PPO", kind: "MultiDiscrete([5]×K) policy" },
  { main: "UAV repositioning", kind: "K UAVs, discrete moves" },
  { main: "uav_positions.json", kind: "handoff: K UAVs + N users + associations" },
  { main: "NS-3", kind: "K UDP servers, per-user association, FlowMonitor", cls: "ns3" },
  { main: "Network validation", kind: "per-flow + per-UAV throughput / delay", cls: "ns3" },
];

export function PipelineDiagram() {
  return (
    <Panel title="Pipeline" hint="each stage runs the existing code unchanged">
      <div className="pipeline">
        {NODES.map((n, i) => (
          <div key={n.main}>
            <div className={`node ${n.cls || ""}`}>
              <span className="n-main">{n.main}</span>
              <span className="n-kind">{n.kind}</span>
            </div>
            {i < NODES.length - 1 && <div className="arrow">&darr;</div>}
            {i === 2 && (
              <div className="sep-label">
                metrics inform the agent &mdash; separate models below
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="inline-note">
        <span className="tag sionna">Sionna RT</span> propagation / radio analysis
        &nbsp;&middot;&nbsp;
        <span className="tag ns3">NS-3</span> network / MAC / traffic validation.
        NS-3 uses its own Wi-Fi channel model; it does not consume Sionna&rsquo;s
        ray-traced channel loss. Multi-UAV SINR is co-channel (all UAVs share one
        channel, full buffer); for 1 UAV, SINR = SNR. No inter-UAV interference is
        claimed beyond this shared-channel model.
      </p>
    </Panel>
  );
}
