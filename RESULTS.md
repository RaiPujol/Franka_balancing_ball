# Revised tray simulation

The revised print has a 233 × 233 mm deck. With the standard Franka hand gripping the tab and the plate facing outward, Kalman + LQR completed all five tasks from both starting positions at 20 ms delay. It passed 30 of 50 trials across all conditions. The previous 150 mm model passed 42 of 50 with this tuning.

## Geometry and feedback

All 396 oriented STL triangles matched the 3MF object's triangles within 0.000005 mm after undoing the slicer's recentering. Including the mounting tab, the part measures 233 × 254 × 21 mm. Holding the tab at the same place moves the deck center 41.5 mm farther out than on the first print. The simulation uses the revised mesh for appearance and inertia. The contact surface fills the 7 mm underside rib envelope and omits the small center dimple. The contact model also fills the tab hole and undercuts.

We used the old print's measured 92.77 g and the ratio of revised to original CAD volumes (256.824 to 113.801 cm³) to estimate the new print at **209.36 g**. The new print, fasteners and center of mass have yet to be measured. The nominal golf ball is 42.67 mm in diameter and weighs 45.93 g.

The controller sees robot motion and a delayed simulated joint-torque signal. It estimates the ball's position from the residual after subtracting the modeled empty-tray load. The simulator's ball position scores each run and supplies the orange video trace. Tray dimensions are available to all four controllers. The starting ball offset is withheld.

## Five recorded tasks

These clips use seed 401, an initial offset of (+18, −12) mm, 50 ms delay, joint-torque noise of 0.0005 N·m standard deviation and joint-velocity noise of 0.0001 rad/s standard deviation. Shapes require less than 5 mm RMS tracking error during the constant-amplitude portion. Centering and carrying require the ball to stay within 3 mm of center throughout the final second. A pass also requires completion, contact, a positive or zero edge margin, tray path error below 20 mm and no unexpected contacts or simulation warnings.

| Task | Result | Measurement |
| --- | --- | --- |
| Circle | Passed | 2.90 mm RMS |
| Rounded square | Stopped | At 20.06 s of 43 s, the estimated normal force fell below threshold |
| Figure eight | Passed | 2.71 mm RMS |
| Centering | Passed | 1.02 mm maximum offset in the final second |
| Carrying | Passed | 1.45 mm maximum offset in the final second |

A separate nominal rounded-square video uses the same initial offset and seed, 20 ms delay and 0.0001 N·m torque noise. It completed the full 43 s at 2.12 mm RMS. [Video filenames and scores](results/video_index.csv)

## Noise, delay and controller choice

Each row uses two seeds for each of five tasks. A force-threshold stop records an unreliable measurement. It does not establish that the ball left the tray.

| Condition | Kalman + LQR passes |
| --- | ---: |
| 20 ms, 0.0001 N·m torque noise | 10/10 |
| 50 ms, 0.0005 N·m torque noise | 8/10 |
| 80 ms, 0.001 N·m torque noise | 4/10 |
| 50 ms, 0.001 N·m wrist torque bias | 8/10 |
| 50 ms, observer assumes 10% excess tray mass | 0/10 |

In the 80 ms circle with seed 401, the feedback estimate dipped to 0.152 N against the 0.200 N threshold. The simulated ball still exerted 0.266 N on the tray at that sample. In the 10% mass-error case, the first rejected reading was 0.170 N while the ball exerted 0.450 N. These contact measurements were recorded for diagnosis only. [Force readings at the stops](results/wrench_stop_diagnostics.json)

Fifteen LQR trials stopped at the normal-force check. Five more finished outside the performance criteria. The four methods below used the same seeds, starts, tasks and disturbances. On the revised plate the controller receives its actual dimensions for the carrying and centering tasks. This also corrects a stale size assumption in the earlier MPC comparison.

| Controller | All trials passed | Circle RMS in the matched 50 ms video |
| --- | ---: | ---: |
| PD | 10/50 | 9.09 mm |
| Kalman + PD | 26/50 | 1.76 mm |
| Kalman + LQR | 30/50 | 2.90 mm |
| Kalman + hybrid MPC/PD | 22/50 | 3.41 mm |

Kalman + LQR passed the most cases with these tunings. Kalman + PD tracked the single recorded circle most accurately. The independent LQR runs matched the LQR comparison outcomes across all 50 cases. [Main trials](results/stl_main_trials.csv) · [All four methods](results/method_all_trials.csv) · [Source hashes](results/method_scene_manifest.json)

## Print mass, friction and available deck area

- **Print mass.** Two new seeds tested circle and carrying at 50 ms. When the observer used the correct simulated mass, 4/4 passed at 157.02 g, 3/4 at 209.36 g and 2/4 at 261.70 g. Keeping the observer fixed at 209.36 g gave 0/4 at each off-nominal mass. The heavier plant with the fixed observer completed all four runs but missed the error targets. [Mass trials](results/print_mass_sensitivity.csv)
- **Friction and fasteners.** A 48-trial circle and centering sweep used friction coefficients of 0.1, 0.3 and 0.6, modeled bolt masses of 0 and 10 g, two delays and two new seeds. The outcomes were 0/16, 12/16 and 12/16 respectively. All 24 failures stopped at the normal-force check. [Material trials](results/material_sensitivity.csv)
- **Larger patterns.** At 50 ms, circles of 60 and 80 mm radius each stopped in both seeds even with the same steady tangential speed as the 20 mm circle. The simulated ball footprint still had at least 27 mm of edge margin in the recorded samples. Centering from (70, 0) mm passed twice, ending within 1.03 mm. Centering from (−80, 60) mm stopped in both trials. [Expanded deck trials](results/expanded_deck_trials.csv)

## Clearance and hardware

The smallest modeled plate-to-finger gap was **0.158 mm** during the sampled circle and carrying runs. The smallest modeled ball-to-arm gaps were 66.76 mm for the circle and 72.50 mm for carrying. The mount is rigid in simulation. Real finger compliance, bolts, cables and surrounding obstacles can change these distances. [Sampled distances](results/distance_audit.json)

The separate `torque_balance/` bench models a 254 mm square paddle and a different attachment. Its default joint pose holds the revised CAD about 90° from level. The outward home used in these trials holds the revised CAD level. Before physical testing, weigh the print and bolts, measure the installed mount and clearances, and record empty-tray and known ball-position torque readings at the intended level pose.
