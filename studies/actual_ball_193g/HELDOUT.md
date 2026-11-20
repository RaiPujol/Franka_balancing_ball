# Held-out starts and assumed joint-torque noise

These are offline MuJoCo trials on the 233 mm plate and measured 74.5 mm, 193 g ball, with gravity-referenced hybrid MPC/PD from the preceding study. They use six fixed random starts sampled with NumPy seed 20261120, with initial radial offsets uniformly drawn between 20 and 55 mm. The tasks are 20 mm-radius circle tracking and centering. Each task-start pair uses the same seed and settings across conditions. Nominal sliding friction is 0.3, and the ball uses the uniform solid-sphere inertia assumption.

| Delay and independent white joint-torque noise std | Circle | Centering |
| --- | ---: | ---: |
| 20 ms, 0.0001 N·m | 6/6 | 6/6 |
| 50 ms, 0.0005 N·m | 6/6 | 6/6 |
| 80 ms, 0.001 N·m | 6/6 | 6/6 |
| 50 ms, 0.006 N·m | 6/6 | 0/6 |
| 80 ms, 0.006 N·m | 6/6 | 1/6 |

Circle passes use less than 5 mm RMS tracking error on the plateau. Centering passes require the ball offset to remain within 3 mm for the final second. Other checks require completion, contact, containment, and path tracking. All 60 runs completed with no samples lacking ball-tray contact after warmup. The smallest modeled ball-footprint margin was 21.50 mm.

The 0.006 N·m stress value is roughly the size of the real arm's stationary joint 5 and 6 per-sample torque variations. The simulator injects independent white noise into all seven joints, while real robot signals have correlated drift, unknown latency and model error. This is a synthetic stress test, not a calibration or hardware reliability measurement. The model uses estimated ball position for feedback, and simulator truth only for scoring.

Reproduce from repository root with the installed dependencies:

    python studies/actual_ball_193g/heldout_study.py

The published results are in results/heldout.csv. The command writes new results under output/heldout_robustness_20261120/.

Six exploratory filter tunings on two starts did not consistently improve centering under 0.006 N·m noise. These are exploratory results only and were not promoted to the controller. The tested values are in results/heldout_tuning_exploratory.csv. Reproduce with `python studies/actual_ball_193g/heldout_tuning.py`.
