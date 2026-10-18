// node --test src/control/   (no extra dependencies)
// Checks that the view model reproduces the backend state exactly.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { sceneData, historySeries, beforeAfter, decisionRows, pdrColor, PDR_GOOD, PDR_MID, PDR_BAD,
  isActive } from "./model.js";

const FIXTURE = process.env.RESQNET_STATE_FIXTURE;

const mini = {
  status: "running",
  session: { process_alive: true },
  config: { objective: { pdr_target: 0.9, coverage_pdr_threshold: 0.5 } },
  uavs: [
    { id: "UAV-1", position: [500, -20, 50], measured_at: [500, -20, 50], status: "moved" },
    { id: "UAV-2", position: [470, -95, 30], measured_at: [470, -80, 40], status: "moving" },
  ],
  users: [
    { id: "S-1", position: [480, 0, 1.5], serving_uav: 1, pdr: 0.95 },
    { id: "S-2", position: [460, -100, 1.5], serving_uav: 2, pdr: 0.6 },
    { id: "S-3", position: [470, -120, 1.5], serving_uav: 2, pdr: 0.0 },
  ],
  history: [
    { step: 0, kind: "initial", num_users: 3, positions: [[500, -20, 40], [470, -80, 40]],
      metrics: { objective: 0.3, coverage: 0.67, pdr: 0.5, worst_decile_pdr: 0, mean_latency_ms: 90 } },
    { step: 1, kind: "reverted", num_users: 3, positions: [[500, -20, 40], [470, -80, 40]],
      metrics: { objective: 0.3, coverage: 0.67, pdr: 0.5, worst_decile_pdr: 0, mean_latency_ms: 90 },
      trial_metrics: { objective: 0.2 } },
    { step: 2, kind: "accepted", num_users: 3, positions: [[500, -20, 50], [470, -80, 40]],
      metrics: { objective: 0.4, coverage: 0.67, pdr: 0.52, worst_decile_pdr: 0, mean_latency_ms: 60 } },
  ],
  before: { metrics: { objective: 0.3, coverage: 0.67, pdr: 0.5, worst_decile_pdr: 0, mean_latency_ms: 90,
    users_pdr_ge_target: 1, users_no_delivery: 1 } },
  after: { metrics: { objective: 0.4, coverage: 0.67, pdr: 0.52, worst_decile_pdr: 0, mean_latency_ms: 60,
    users_pdr_ge_target: 1, users_no_delivery: 1 } },
  decisions: [
    { step: 1, uav_id: "UAV-1", outcome: "reverted" },
    { step: 1, uav_id: "UAV-2", outcome: "held" },
    { step: 2, uav_id: "UAV-1", outcome: "accepted" },
  ],
};

test("scene data mirrors UAV and user state", () => {
  const s = sceneData(mini);
  assert.deepEqual(s.uavs.map((u) => u.position), mini.uavs.map((u) => u.position));
  assert.deepEqual(s.users.map((u) => u.position), mini.users.map((u) => u.position));
  // links go to the position at which the link was measured
  assert.deepEqual(s.users[1].servingPos, [470, -80, 40]);
  assert.deepEqual(s.trails[0], [[500, -20, 40], [500, -20, 40], [500, -20, 50]]);
  assert.deepEqual(s.users.map((u) => u.color), [PDR_GOOD, PDR_MID, PDR_BAD]);
});

test("pdr colours", () => {
  assert.equal(pdrColor(null), PDR_BAD);
  assert.equal(pdrColor(0.9), PDR_GOOD);
  assert.equal(pdrColor(0.5), PDR_MID);
});

test("history series one row per interval with per-UAV altitude", () => {
  const h = historySeries(mini);
  assert.equal(h.length, 3);
  assert.equal(h[2].z1, 50);
  assert.equal(h[1].trial_objective, 0.2);
  assert.equal(h[2].mean_latency_ms, 60);
});

test("before/after verdicts respect metric direction", () => {
  const rows = Object.fromEntries(beforeAfter(mini).map((r) => [r.key, r]));
  assert.equal(rows.objective.verdict, "better");
  assert.equal(rows.mean_latency_ms.verdict, "better");   // lower latency is better
  assert.ok(Math.abs(rows.mean_latency_ms.delta + 30) < 1e-9);
  assert.equal(rows.coverage.verdict, "same");
});

test("decision filtering", () => {
  assert.equal(decisionRows(mini).length, 3);
  assert.equal(decisionRows(mini, { includeHolds: false }).length, 2);
  assert.deepEqual(decisionRows(mini, { uavId: "UAV-1" }).map((d) => d.step), [2, 1]);
  assert.ok(isActive(mini));
  assert.ok(!isActive({ ...mini, status: "finished" }));
});

test("real controller state (fixture from the backend, if provided)", { skip: !FIXTURE || !existsSync(FIXTURE) }, () => {
  const st = JSON.parse(readFileSync(FIXTURE, "utf8"));
  const s = sceneData(st);
  assert.equal(s.uavs.length, st.uavs.length);
  assert.equal(s.users.length, st.users.length);
  s.uavs.forEach((u, k) => assert.deepEqual(u.position, st.uavs[k].position));
  const last = st.history[st.history.length - 1];
  assert.deepEqual(s.trails.map((t) => t[t.length - 1]), last.positions);
  const ba = Object.fromEntries(beforeAfter(st).map((r) => [r.key, r]));
  assert.equal(ba.objective.after, st.after.metrics.objective);
  assert.equal(historySeries(st).length, st.history.length);
});

// ---- 3D PPO additions (policyInfo / stepActionRows / uavObstruction) ----
import { policyInfo, stepActionRows, uavObstruction } from "./model.js";

test("policy info: rule-based unless the backend says ppo", () => {
  assert.equal(policyInfo(mini).kind, "rule_based");
  assert.equal(policyInfo({ provenance: { policy: { policy: "rule_based" } } }).kind, "rule_based");
  const p = policyInfo({ provenance: { policy: { policy: "ppo", checkpoint: "/m/run/model.zip",
    checkpoint_sha256: "abcdef0123456789", run_id: "run", deterministic: true, latency_in_reward: false } } });
  assert.equal(p.kind, "ppo");
  const d = Object.fromEntries(p.details);
  assert.equal(d.checkpoint, "/m/run/model.zip");
  assert.equal(d.sha256, "abcdef012345");
  assert.equal(d["latency in reward"], "no");
});

test("step action rows reproduce backend actions and positions", () => {
  const st = { history: [
    { step: 0, kind: "initial", positions: [[0, 0, 40], [30, 0, 30]], actions: null },
    { step: 1, kind: "accepted", positions: [[0, 0, 50], [30, 0, 20]], actions: ["up", "down"] },
  ] };
  const rows = stepActionRows(st);
  assert.equal(rows.length, 4);
  assert.deepEqual(rows[2], { step: 1, kind: "accepted", uav: "UAV-1", k: 0, action: "up",
    from: [0, 0, 40], position: [0, 0, 50] });
  assert.equal(rows[3].action, "down");
  assert.equal(rows[0].action, null);
});

test("uav obstruction summarises the backend obstruction_db per serving UAV", () => {
  const st = { uavs: [{ id: "UAV-1" }, { id: "UAV-2" }], users: [
    { serving_uav: 1, obstruction_db: 0 }, { serving_uav: 1, obstruction_db: 20 },
    { serving_uav: 2, obstruction_db: 200 } ] };
  const o = uavObstruction(st);
  assert.equal(o[0].links, 2);
  assert.equal(o[0].mean_db, 10);
  assert.equal(o[0].share_over_10db, 0.5);
  assert.equal(o[1].no_path, 1);
  assert.equal(o[1].mean_db, 60);
});
