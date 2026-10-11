import { Panel } from "./Panel.jsx";

const COLS = [
  {
    tag: "sionna",
    name: "SIONNA RT",
    sub: "Radio propagation",
    items: [
      "Shoot-and-bounce ray tracing (Mitsuba)",
      "K transmitters over the 3D city mesh",
      "Per-user × per-UAV RSS, path gain, delay",
      "SNR and co-channel SINR",
      "Best-RSS user→UAV association",
    ],
  },
  {
    tag: "ppo",
    name: "PPO",
    sub: "Reinforcement learning",
    items: [
      "MultiDiscrete([5]×K) policy",
      "Per-UAV actions: stay / ±X / ±Y",
      "Reward: coverage + connectivity + balance − overlap − move",
      "Trajectory of K UAV positions",
    ],
  },
  {
    tag: "ns3",
    name: "NS-3",
    sub: "Network validation",
    items: [
      "One UDP server per UAV",
      "Each user is a client of its associated UAV",
      "802.11n ad-hoc, YansWifiChannel",
      "FlowMonitor: throughput, delay, delivery",
    ],
  },
];

export function PipelineDiagram() {
  return (
    <Panel title="Methodology" flush right={<span className="hint">scenario → Sionna → PPO → JSON → NS-3</span>}>
      <div className="method-grid">
        {COLS.map((c) => (
          <div className="method-col" key={c.name}>
            <div className="mc-head">
              <span className={`tag ${c.tag}`}>{c.name}</span>
            </div>
            <div className="mc-sub">{c.sub}</div>
            <ul>{c.items.map((it) => <li key={it}>{it}</li>)}</ul>
          </div>
        ))}
      </div>
      <div className="method-flow">
        scenario <span className="arrow">→</span> sionna <span className="arrow">→</span> ppo
        <span className="arrow">→</span> uav_positions.json <span className="arrow">→</span> ns-3
        <span className="arrow">→</span> results
      </div>
      <div className="scene-note">
        Sionna and NS-3 use <b>different channel / network models</b>. NS-3 does <b>not</b>{" "}
        directly consume Sionna&rsquo;s ray-traced channel loss — only the geometry (UAV
        positions, user positions, association) is handed over. Multi-UAV SINR assumes all
        UAVs share one channel with a full buffer; for a single UAV, SINR = SNR.
      </div>
    </Panel>
  );
}
