# Balancing the measured 193 g ball

This study uses the 74.5 mm diameter, 193 g ball and the outward-facing 233 × 233 mm, 162.5 g printed tray. Hybrid MPC/PD with a gravity-referenced level orientation passed **60/60** nominal-friction trials across four starts, five tasks and three sensor settings. The original LQR shape runs missed their tracking target even with good ball position estimates. Under load, they had treated a slightly tilted tray as level.

## Controller comparison

Each column includes two starting positions, five tasks and all three sensor settings. The 20/50/80 ms delays use joint-torque noise standard deviations of 0.0001/0.0005/0.001 N·m, respectively. Joint-velocity noise is 0.0001 rad/s. The first starts are (+18, −12) mm and (−16, +18) mm. The additional starts are (+24, +16) mm and (−27, −20) mm, each with a new noise seed.

| Controller | Original reference | Gravity reference | Gravity reference, additional starts |
| --- | ---: | ---: | ---: |
| PD | 1/30 | 7/30 | — |
| Kalman + PD | 5/30 | 17/30 | — |
| Kalman + LQR | 9/30 | 27/30 | 27/30 |
| Kalman + hybrid MPC/PD | 12/30 | 30/30 | 30/30 |

The shape runner captured its reference orientation after the heavy ball had tilted the tray by 0.36–0.39° during startup. The correction preserves the heading from forward kinematics and takes the reference normal from gravity. With the same LQR gains, shape tracking improved from 10.42–11.68 mm RMS to 0.90–2.75 mm RMS. Ball position and velocity feedback come from the delayed torque-based estimator. Ground truth supplies the evaluation metrics and the orange video overlay.

The selected controller blends 90% of its PD/feedforward tilt with 10% of the predictive tilt. MPC replans every 50 ms over a 14-step horizon. Across its 60 nominal-friction trials, shape RMS error ranged from 1.09 to 2.74 mm, the minimum ball-footprint margin was 41.49 mm, and the ball maintained modeled support contact after warmup. Centering and carrying required a final full second within 3 mm. Shapes required less than 5 mm tracking RMS. All tasks also required completion, tray position error below 20 mm, containment, contact and the existing simulation checks.

## Contact assumptions

The nominal trials use sliding friction 0.3 and a uniform sphere. At 50 ms we tested sliding friction coefficients 0.1, 0.3 and 0.6, and three ball inertia factors I/(mr²). These were 0.2 for a mass distribution weighted toward the center, 0.4 for a solid uniform sphere and 2/3 for a thin shell. The controller retained its original model and gains across the sweep.

At friction 0.3 and 0.6, both gravity-referenced LQR and hybrid MPC/PD passed all 60 trials per controller across the three inertia factors. At 0.1, each passed 0/30. All of those runs had samples without modeled ball contact. The real friction, ball inertia, mounting transform, bolt mass, torque noise and delay remain unmeasured in this study. The orientation correction assumes a known gravity direction in the robot base frame and a known orientation of the tray relative to the hand. The study exercised simulation only.

## Reproduce

From the repository root, use the Python environment specified in the root README.

```sh
python studies/actual_ball_193g/ball_study.py --suite all --workers 6
python studies/actual_ball_193g/render_videos.py
```

The first command writes the 200 original-reference cases, 200 gravity-reference cases and 140 challenge cases to `output/actual_ball_193g/`. The second writes five MP4s to `output/actual_ball_193g/videos/`. Pass `--publish-dir "/path/to/video folder"` to copy the verified MP4s to another location. The videos use gravity-referenced MPC/PD, delay 50 ms, seed 401 and the starting offset (+18, −12) mm. The simulator checks the actual ball diameter, mass and inertia in both the plant and the unloaded observer. The original golf-ball presets remain available through their existing commands.

[Original trials](results/original.csv) · [Gravity-referenced trials](results/gravity.csv) · [Additional starts and MPC sensitivity](results/challenge.csv) · [Grouped results](results/summary.json) · [Video scores](results/video_index.csv) · [Simulation source hashes](results/source_hashes.json)

For six additional initial positions and a higher assumed noise level, see [held-out robustness](HELDOUT.md).
