import { useState } from "react";
import App from "./App.jsx";
import { ControlView } from "./control/ControlView.jsx";
import "./control/control.css";

// Two views: the new closed-loop control twin, and the existing pipeline
// results dashboard (App.jsx, unchanged).
const TABS = [
  ["control", "Closed-loop control (simulated)"],
  ["pipeline", "Pipeline results"],
];

function initialTab() {
  try {
    return localStorage.getItem("resqnet.tab") || "control";
  } catch {
    return "control";
  }
}

export default function Root() {
  const [tab, setTab] = useState(initialTab);
  const choose = (t) => {
    setTab(t);
    try { localStorage.setItem("resqnet.tab", t); } catch { /* ignore */ }
  };
  return (
    <>
      <nav className="root-tabs">
        <span className="brand">ResQNet</span>
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? "on" : ""} onClick={() => choose(id)}>{label}</button>
        ))}
      </nav>
      {tab === "control" ? <ControlView /> : <App />}
    </>
  );
}
