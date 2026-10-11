# Network evaluation — sionna_n30

RF model: **sionna** · users: 30 · UAVs: 3 · PPO steps: 12 · seeds per scenario: 5 · values are mean ± 95% CI over seeds.

## blocked_urban

| metric | fixed | random | ppo_legacy | ppo_network |
|---|---|---|---|---|
| RF coverage % (legacy PG>=-100 dB) | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |
| RF connected users (legacy thresholds) | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| predicted usable ns-3 links | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| ns-3 users with any delivery | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| ns-3 users with PDR >= 90% | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| ns-3 aggregate PDR | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 1.000 ± 0.000 |
| ns-3 mean per-user goodput (Mbps) | 0.432 ± 0.000 | 0.432 ± 0.000 | 0.432 ± 0.000 | 0.432 ± 0.000 |
| ns-3 worst-user goodput (Mbps) | 0.430 ± 0.001 | 0.430 ± 0.001 | 0.429 ± 0.003 | 0.430 ± 0.001 |
| ns-3 mean latency (ms) | 10.734 ± 0.473 | 10.674 ± 0.906 | 10.781 ± 0.580 | 10.734 ± 0.473 |
| network reward (state) | 0.899 ± 0.001 | 0.896 ± 0.005 | 0.899 ± 0.001 | 0.899 ± 0.001 |
| legacy reward (state) | 0.763 ± 0.015 | 0.758 ± 0.026 | 0.778 ± 0.005 | 0.763 ± 0.015 |
| mean UAV movement (m) | 0.000 ± 0.000 | 0.000 ± 0.000 | 8.943 ± 7.069 | 0.000 ± 0.000 |

Paired per-seed differences (A − B), wins/ties/losses for A in the metric's good direction, Wilcoxon signed-rank p:

| comparison | predicted usable ns-3 links | ns-3 users with any delivery | ns-3 users with PDR >= 90% | ns-3 aggregate PDR | ns-3 mean per-user goodput (Mbps) | ns-3 mean latency (ms) | network reward (state) |
|---|---|---|---|---|---|---|---|
| ppo_network − fixed | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) |
| ppo_network − random | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (2/0/3, p=1) | +0.000 (2/0/3, p=1) | +0.060 (2/0/3, p=1) | +0.003 (3/0/2, p=0.31) |
| ppo_network − ppo_legacy | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (3/1/1, p=0.5) | +0.000 (3/1/1, p=0.5) | -0.047 (2/1/2, p=0.88) | -0.000 (2/2/1, p=1) |

Network model vs ns-3: per-user usable-link agreement 1.000; Spearman(predicted PDR, ns-3 PDR) = None

## wide_area

| metric | fixed | random | ppo_legacy | ppo_network |
|---|---|---|---|---|
| RF coverage % (legacy PG>=-100 dB) | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |
| RF connected users (legacy thresholds) | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| predicted usable ns-3 links | 26.800 ± 2.221 | 24.600 ± 3.117 | 27.000 ± 2.323 | 26.800 ± 2.221 |
| ns-3 users with any delivery | 27.800 ± 1.039 | 24.800 ± 2.964 | 28.200 ± 1.039 | 27.800 ± 1.039 |
| ns-3 users with PDR >= 90% | 18.800 ± 1.842 | 21.400 ± 2.256 | 20.400 ± 1.883 | 18.800 ± 1.619 |
| ns-3 aggregate PDR | 0.760 ± 0.038 | 0.768 ± 0.078 | 0.773 ± 0.034 | 0.755 ± 0.045 |
| ns-3 mean per-user goodput (Mbps) | 0.328 ± 0.017 | 0.332 ± 0.034 | 0.334 ± 0.015 | 0.326 ± 0.019 |
| ns-3 worst-user goodput (Mbps) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| ns-3 mean latency (ms) | 112.743 ± 19.814 | 79.831 ± 18.989 | 103.202 ± 7.778 | 115.462 ± 23.105 |
| network reward (state) | 0.777 ± 0.065 | 0.712 ± 0.094 | 0.783 ± 0.067 | 0.777 ± 0.065 |
| legacy reward (state) | 0.754 ± 0.016 | 0.740 ± 0.022 | 0.768 ± 0.006 | 0.754 ± 0.016 |
| mean UAV movement (m) | 0.000 ± 0.000 | 0.000 ± 0.000 | 16.841 ± 12.185 | 1.333 ± 2.267 |

Paired per-seed differences (A − B), wins/ties/losses for A in the metric's good direction, Wilcoxon signed-rank p:

| comparison | predicted usable ns-3 links | ns-3 users with any delivery | ns-3 users with PDR >= 90% | ns-3 aggregate PDR | ns-3 mean per-user goodput (Mbps) | ns-3 mean latency (ms) | network reward (state) |
|---|---|---|---|---|---|---|---|
| ppo_network − fixed | +0.000 (0/5/0, p=1) | +0.000 (0/5/0, p=1) | +0.000 (1/3/1, p=1) | -0.005 (1/3/1, p=1) | -0.002 (1/3/1, p=1) | +2.720 (0/3/2, p=0.5) | +0.000 (1/3/1, p=1) |
| ppo_network − random | +2.200 (4/1/0, p=0.12) | +3.000 (4/1/0, p=0.12) | -2.600 (0/0/5, p=0.062) | -0.014 (2/0/3, p=0.62) | -0.006 (2/0/3, p=0.62) | +35.631 (1/0/4, p=0.12) | +0.065 (5/0/0, p=0.062) |
| ppo_network − ppo_legacy | -0.200 (0/4/1, p=1) | -0.400 (0/3/2, p=0.5) | -1.600 (0/0/5, p=0.062) | -0.018 (2/0/3, p=0.31) | -0.008 (2/0/3, p=0.31) | +12.261 (1/0/4, p=0.12) | -0.006 (2/0/3, p=0.81) |

Network model vs ns-3: per-user usable-link agreement 0.972; Spearman(predicted PDR, ns-3 PDR) = 0.332
