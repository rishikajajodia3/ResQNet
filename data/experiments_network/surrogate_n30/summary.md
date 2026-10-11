# Network evaluation — surrogate_n30

RF model: **surrogate** · users: 30 · UAVs: 3 · PPO steps: 12 · seeds per scenario: 20 · values are mean ± 95% CI over seeds.

## blocked_urban

| metric | fixed | random | ppo_legacy | ppo_network |
|---|---|---|---|---|
| RF coverage % (legacy PG>=-100 dB) | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |
| RF connected users (legacy thresholds) | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| predicted usable ns-3 links | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| ns-3 users with any delivery | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| ns-3 users with PDR >= 90% | 30.000 ± 0.000 | 29.950 ± 0.105 | 29.950 ± 0.105 | 30.000 ± 0.000 |
| ns-3 aggregate PDR | 0.999 ± 0.000 | 0.998 ± 0.003 | 0.998 ± 0.003 | 0.999 ± 0.000 |
| ns-3 mean per-user goodput (Mbps) | 0.432 ± 0.000 | 0.431 ± 0.001 | 0.431 ± 0.001 | 0.432 ± 0.000 |
| ns-3 worst-user goodput (Mbps) | 0.429 ± 0.001 | 0.409 ± 0.041 | 0.409 ± 0.041 | 0.429 ± 0.001 |
| ns-3 mean latency (ms) | 11.019 ± 0.259 | 11.315 ± 1.156 | 10.922 ± 0.326 | 11.019 ± 0.259 |
| network reward (state) | 0.900 ± 0.000 | 0.895 ± 0.003 | 0.900 ± 0.000 | 0.900 ± 0.000 |
| legacy reward (state) | 0.772 ± 0.007 | 0.746 ± 0.013 | 0.783 ± 0.003 | 0.772 ± 0.007 |
| mean UAV movement (m) | 0.000 ± 0.000 | 0.000 ± 0.000 | 5.648 ± 3.042 | 0.000 ± 0.000 |

Paired per-seed comparison of **ppo_network** (wins/ties/losses, Wilcoxon p; 'win' = better in the metric's good direction):

| metric | vs fixed | vs random | vs ppo_legacy |
|---|---|---|---|
| RF coverage % (legacy PG>=-100 dB) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| RF connected users (legacy thresholds) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| predicted usable ns-3 links | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| ns-3 users with any delivery | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| ns-3 users with PDR >= 90% | Δ +0.000 (0/20/0, p=1) | Δ +0.050 (1/19/0, p=0.677) | Δ +0.050 (1/19/0, p=0.677) |
| ns-3 aggregate PDR | Δ +0.000 (0/20/0, p=1) | Δ +0.002 (10/2/8, p=0.985) | Δ +0.002 (5/11/4, p=0.703) |
| ns-3 mean per-user goodput (Mbps) | Δ +0.000 (0/20/0, p=1) | Δ +0.001 (10/2/8, p=0.985) | Δ +0.001 (5/11/4, p=0.732) |
| ns-3 worst-user goodput (Mbps) | Δ +0.000 (0/20/0, p=1) | Δ +0.019 (9/2/9, p=0.94) | Δ +0.020 (5/11/4, p=0.62) |
| ns-3 mean latency (ms) | Δ +0.000 (0/20/0, p=1) | Δ -0.297 (9/0/11, p=0.756) | Δ +0.097 (4/9/7, p=0.336) |
| network reward (state) | Δ +0.000 (0/20/0, p=1) | Δ +0.005 (15/5/0, p=0.000266) | Δ +0.000 (1/19/0, p=0.677) |
| legacy reward (state) | Δ +0.000 (0/20/0, p=1) | Δ +0.026 (17/0/3, p=0.00422) | Δ -0.011 (1/9/10, p=0.00623) |
| mean UAV movement (m) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ -5.648 (11/9/0, p=0.00184) |

Network model vs ns-3: per-user usable-link agreement 1.000; Spearman(predicted PDR, ns-3 PDR) = None

## wide_area

| metric | fixed | random | ppo_legacy | ppo_network |
|---|---|---|---|---|
| RF coverage % (legacy PG>=-100 dB) | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |
| RF connected users (legacy thresholds) | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 | 30.000 ± 0.000 |
| predicted usable ns-3 links | 27.800 ± 0.580 | 23.700 ± 2.126 | 27.750 ± 0.586 | 27.800 ± 0.637 |
| ns-3 users with any delivery | 27.750 ± 0.586 | 23.700 ± 2.126 | 27.700 ± 0.590 | 27.750 ± 0.642 |
| ns-3 users with PDR >= 90% | 18.550 ± 0.736 | 18.150 ± 2.160 | 19.500 ± 0.990 | 18.500 ± 0.767 |
| ns-3 aggregate PDR | 0.746 ± 0.023 | 0.682 ± 0.067 | 0.749 ± 0.025 | 0.750 ± 0.021 |
| ns-3 mean per-user goodput (Mbps) | 0.322 ± 0.010 | 0.295 ± 0.029 | 0.323 ± 0.011 | 0.324 ± 0.009 |
| ns-3 worst-user goodput (Mbps) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| ns-3 mean latency (ms) | 115.236 ± 6.696 | 92.301 ± 19.526 | 110.752 ± 7.817 | 114.870 ± 7.654 |
| network reward (state) | 0.806 ± 0.017 | 0.683 ± 0.061 | 0.806 ± 0.018 | 0.806 ± 0.019 |
| legacy reward (state) | 0.759 ± 0.007 | 0.744 ± 0.011 | 0.769 ± 0.003 | 0.759 ± 0.007 |
| mean UAV movement (m) | 0.000 ± 0.000 | 0.000 ± 0.000 | 12.551 ± 7.696 | 1.167 ± 1.700 |

Paired per-seed comparison of **ppo_network** (wins/ties/losses, Wilcoxon p; 'win' = better in the metric's good direction):

| metric | vs fixed | vs random | vs ppo_legacy |
|---|---|---|---|
| RF coverage % (legacy PG>=-100 dB) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| RF connected users (legacy thresholds) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| predicted usable ns-3 links | Δ +0.000 (1/18/1, p=1) | Δ +4.100 (20/0/0, p=7.94e-05) | Δ +0.050 (4/14/2, p=0.546) |
| ns-3 users with any delivery | Δ +0.000 (1/18/1, p=1) | Δ +4.050 (20/0/0, p=8.01e-05) | Δ +0.050 (4/14/2, p=0.546) |
| ns-3 users with PDR >= 90% | Δ -0.050 (1/18/1, p=0.984) | Δ +0.350 (9/3/8, p=0.793) | Δ -1.000 (4/8/8, p=0.114) |
| ns-3 aggregate PDR | Δ +0.004 (2/18/0, p=0.425) | Δ +0.067 (13/0/7, p=0.0696) | Δ +0.001 (6/8/6, p=0.821) |
| ns-3 mean per-user goodput (Mbps) | Δ +0.002 (2/18/0, p=0.425) | Δ +0.029 (13/0/7, p=0.0696) | Δ +0.001 (6/8/6, p=0.821) |
| ns-3 worst-user goodput (Mbps) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) | Δ +0.000 (0/20/0, p=1) |
| ns-3 mean latency (ms) | Δ -0.366 (1/18/1, p=0.984) | Δ +22.569 (5/0/15, p=0.04) | Δ +4.117 (4/8/8, p=0.259) |
| network reward (state) | Δ +0.000 (1/18/1, p=0.984) | Δ +0.123 (20/0/0, p=1.91e-06) | Δ +0.001 (7/8/5, p=0.679) |
| legacy reward (state) | Δ +0.000 (1/18/1, p=0.984) | Δ +0.015 (15/0/5, p=0.0362) | Δ -0.010 (1/8/11, p=0.00335) |
| mean UAV movement (m) | Δ +1.167 (0/18/2, p=0.425) | Δ +1.167 (0/18/2, p=0.425) | Δ -11.384 (11/8/1, p=0.00451) |

Network model vs ns-3: per-user usable-link agreement 0.999; Spearman(predicted PDR, ns-3 PDR) = 0.47
