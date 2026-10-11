# ResQNet closed-loop UAV placement: audit and results

Source: `data/experiments_closed_loop/main_surrogate/`, which was read but not modified. 3 scenarios × 20 seeds × 9 methods,
30 users, 3 UAVs. Every number here comes from ns-3 simulation, **not real-world measurement**.
Regenerate the figures and tables with `python ml/analyze_closed_loop.py`. Full tables are in `tables.md` and
`metrics_by_method.csv`, and the machine-checked audit is in `audit.json`.

## 1. What the pipeline actually does

| Step | Where | What happens |
|---|---|---|
| Users | `ml/scenarios.py: generate_users` | Seeded user positions at 1.5 m height (seed = 0…19) |
| Start layout | `scenarios.initial_uav_positions` | Fixed geometric triangle; **z = middle of the scenario's z-range** (40 m wide/flood, 30 m urban) |
| Placement | `evaluate_closed_loop.main` → `ev.ppo_rollout`, `ps.SURROGATE_OPTIMISERS` | PPO and the surrogate optimisers move UAVs in **X/Y only**, scored by a *calibrated model*: a gradient-boosting model fitted to ns-3 runs of the 5 training scenarios (`ml/calibrated_model.py`) |
| Handoff | `handoff_for` → `multiuav_io.build_uav_positions` | Positions, users, and best-signal user→UAV association written to JSON |
| ns-3 | `ns3/uav-demo.cc` (via `evaluate_network.run_ns3`) | Simulates the network; returns per-user packets sent/received and delay from FlowMonitor |
| Feedback | `placement_search.ns3_refine_batch` | **Only `ns3_refine`**: measured ns-3 PDR picks the best start, then decides which single-UAV 20 m and 10 m moves to keep |
| Stop / final | `ns3_refine_batch` | One round at 20 m, one at 10 m (27 ns-3 runs). The best *measured* placement is kept and scored with fresh 8 s runs (RngRun 11 and 12) |

**Is it closed-loop?** Partly.
- `ns3_refine` is a genuine measurement-driven loop: each round's candidate moves depend on the previous ns-3 measurement. However, it is **shallow**: at most 2 accepted moves.
  - In 9/20 wide-area and 9/20 flood instances it was still improving when it stopped (fig 6).
  - Most of its benefit comes from picking the best starting layout, which is almost always the `coordinate` result (19/20 wide, 17/20 flood). The moves then add +0.019 (wide, 0.953→0.972) and +0.027 (flood, 0.912→0.939) of measured search PDR.
- PPO, greedy, coordinate and evolution never see ns-3. They are closed-loop only against the calibrated model, and ns-3 just grades their final answer.

## 2. Altitude (Z)

Altitude is **fixed**. The action space is `0 stay, ±X, ±Y` (`multiuav_env._apply`), and all optimisers edit only `pos[:, :2]`. The audit confirms one altitude per scenario for every method and seed: 40 m, 30 m, 40 m. **Do not claim 3D optimisation.**

ns-3 does receive Z, and it matters. In a 10-run probe, raising the same layout to 60 m cut the search PDR (wide 0.96→0.91, flood 0.99→0.96). The reason is that the ns-3 channel has no line-of-sight benefit from height, so higher only means farther away. Adding Z moves today would just push UAVs to the minimum altitude, which teaches nothing.

## 3. Network technology actually simulated

- **IEEE 802.11n Wi-Fi** (`WIFI_STANDARD_80211n`), **ad-hoc mode**, one shared `YansWifiChannel`. This is not LTE or 5G NR.
- Band: not set in code, so ns-3.47 uses the 802.11n default of **2.4 GHz** with a 20 MHz channel. Rate control is `IdealWifiManager` (HT MCS 0–7, 6.5–65 Mbit/s).
- Propagation: `LogDistancePropagationLossModel` with exponent 3 and 46.68 dB at 1 m, plus constant-speed delay. There is no fading, no buildings, and no LoS/NLoS in ns-3.
  - Note: 46.68 dB is the free-space loss at **5.15 GHz**. At 2.4 GHz it would be about 40 dB, so links are about 6.6 dB more pessimistic than the band implies.
- Tx power 30 dBm, RxSensitivity −90 dBm, CCA-ED −62 dBm, noise figure 7 dB.
- Traffic: **uplink only**. Each user sends UDP CBR, 512 B every 10 ms (≈0.43 Mbit/s at IP level) to its UAV, from 2 s to 10 s.
- **Ground↔UAV is the only link type.** There are no UAV↔UAV or backhaul links. All 3 UAVs share one channel, so more UAVs do not add airtime. Most losses come from hidden-node collisions and MAC-queue expiry.
- RF used by the optimisers in this run: the **free-space surrogate** at 2.1 GHz (`rf_mode=surrogate`). Sionna ray tracing was *not* used in this experiment.

## 4. Results: ranking by packet delivery (mean ± 95% CI, 20 seeds)

| Scenario | 1st | 2nd (strongest competitor) | 3rd | fixed |
|---|---|---|---|---|
| wide_area | evolution 0.974 ± 0.010 | ns3_refine 0.972 ± 0.010 | coordinate 0.965 ± 0.014 | 0.746 ± 0.023 |
| flood_shelters | ns3_refine 0.942 ± 0.040 | evolution 0.934 ± 0.037 | coordinate 0.901 ± 0.044 | 0.599 ± 0.045 |
| blocked_urban | all methods 0.999: placement does not matter here | | | |

Paired over seeds, `ns3_refine` − baseline, in PDR:
- **vs fixed: +0.226 (wide) and +0.343 (flood).** Wins on 20/20 seeds; Holm-corrected p = 5.7e-06. This is the solid claim.
- vs every PPO variant and greedy_joint: +0.09 to +0.28, significant (20/0/0 or 18/0/2 W/T/L).
- vs coordinate: +0.007 in wide (p = 0.083, *not significant*) and +0.041 in flood (p = 0.004 uncorrected).
- **vs evolution: −0.001 (wide, p = 0.84) and +0.008 (flood, p = 0.96). No difference.**

Other metrics, flood_shelters, fixed → ns3_refine:
- Loss 40.1% → 5.8%
- Worst-user goodput 0.002 → 0.292 Mbit/s
- Jain fairness 0.66 → 0.95
- Mean latency 121 → 42 ms
- p95 of per-user mean latency 460 → 105 ms
- Mean UAV move 67 m (evolution needed 103 m for similar PDR)

**Honest headline:** search-based placement cuts packet loss from 25% to about 3% (wide) and from 40% to about 6% (flood) compared with a fixed layout. However, the ns-3-in-the-loop method is *statistically tied* with a pure surrogate optimiser (evolution). The calibrated surrogate already captures most of the gain. The trained PPO policies are clearly weaker than simple search.

## 5. Correctness and reproducibility checks

| Check | Result |
|---|---|
| Final scores really from ns-3? | **Yes.** All 540 rows recomputed from the raw cached ns-3 outputs: 0 mismatches, 0 missing. The "calibrated PDR" column is a separate, labelled prediction. |
| Traffic duration | Final runs use the 8 s default. Every user sent exactly 800 packets. Search runs use 3 s. |
| Paired seeds | Same users and seeds for all 9 methods. `random` uses `seed+9999`. |
| Leakage | None found. Search uses RngRun 101; finals use 11 and 12. The calibrated model is trained on 5 training scenarios (seeds 5000+); eval scenarios appear only in its validation report (seeds 900+). Search vs final PDR: no winner's-curse bias (mean −0.003…0.000). The 3 s search ranks placements like the 8 s finals (Spearman 0.99 wide/flood). |
| Cache | Key = SHA-256 of (binary hash, handoff, args). Binary hash matches config.json. Duplicate placements (e.g. everything = fixed in blocked_urban) correctly share one simulation. |
| Code vs experiment | Config, scenario, and ns-3 source hashes match `config.json`. |

**Weaknesses found** (methodological; no result-changing bug):
1. The ns-3 loop is shallow (≤ 2 moves) and stops while still improving.
2. `ns3_refine` gets 27 extra ns-3 runs per instance, which the others don't. This is reported, but it is not an equal budget.
3. Latency is computed over *delivered* packets only, so methods that drop distant users can look faster (e.g. `random`). The "p95 latency" is the 95th percentile of per-user *mean* delay, not packet-level p95.
4. Only the primary metric (vs fixed) is multiple-comparison corrected. Treat the other p-values as exploratory.
5. `flood_shelters` was designed *because* a fixed grid fails there (stated in `configs/eval_network.json`), so its large gain is scenario-specific.
6. Reference-loss / band mismatch, no fading, single shared channel, uplink-only CBR (section 3).
7. The experiment code (`evaluate_closed_loop.py`, `placement_search.py`, `calibrated_model.py`, `checkpoint.py`, …) and models are **untracked in git**. Commit them so the result can be reproduced.

## 6. Change made and next experiment

`ns3_refine_iter` (opt-in) repeats each step size until ns-3 measures no gain (≤ 5 rounds). The default `ns3_refine` replays the stored experiment exactly (60/60 placements). `--reuse-cache` reuses finished simulations read-only.

Suggested follow-up. This was **not** run here. Estimated 1–4 h on 4 workers: ≈ 3.8 s wall per new simulation, roughly 1,000–3,800 new search runs. Everything else is answered from the existing cache.

```bash
python ml/evaluate_closed_loop.py --scenarios wide_area flood_shelters --seeds 20 \
  --methods fixed coordinate evolution ppo_calibrated ns3_refine ns3_refine_iter \
  --candidate ns3_refine_iter --workers 4 \
  --reuse-cache data/experiments_closed_loop/main_surrogate/ns3_runs \
  --out data/experiments_closed_loop/refine_iter
```

Question it answers: does deeper measured feedback beat the surrogate-only `evolution`, which it currently ties?

## Figures

- `fig1_delivery.png`: PDR and loss
- `fig2_latency.png`: latency
- `fig3_goodput_fairness.png`: goodput and Jain fairness
- `fig4_movement.png`: movement
- `fig5_paired_pdr.png`: paired ΔPDR vs every method
- `fig6_refine_trace.png`: the feedback loop round by round
