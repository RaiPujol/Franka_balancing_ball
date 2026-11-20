# Revised tray simulation with the measured 162.5 g print

With the 233 × 233 mm deck facing outward, Kalman + LQR passed all five tasks from both starting positions in the 20 ms and 50 ms noise tests. It passed 33/50 across all conditions. The earlier run with identical CAD and a provisional mass of 209.36 g passed 30/50. We kept the gains, seeds and pass criteria fixed for the rerun.

## Geometry and feedback

All 396 oriented STL triangles matched the 3MF object's triangles within 0.000005 mm after undoing the slicer's recentering. Including the mounting tab, the part measures 233 × 254 × 21 mm. Holding the tab at the same place moves the deck center 41.5 mm farther out than on the first print. The simulation uses the revised mesh for appearance and inertia. The contact surface fills the 7 mm underside rib envelope and omits the small center dimple. The contact model also fills the tab hole and undercuts.

The team measured the revised print at **162.5 g**. We used this value for both the simulated plate and the nominal empty-tray model. The nominal run assumes zero added bolt mass. The fastener mass and the assembled center of mass still need to be measured. The nominal golf ball is 42.67 mm in diameter and weighs 45.93 g.

The controller sees robot motion and a delayed simulated joint-torque signal. It estimates the ball's position from the residual after subtracting the modeled empty-tray load. The simulator's ball position scores each run and supplies the orange video trace. Tray dimensions are available to all four controllers. The starting ball offset is withheld.

## Five recorded tasks

These clips use seed 401, an initial offset of (+18, −12) mm, 50 ms delay, joint-torque noise of 0.0005 N·m standard deviation and joint-velocity noise of 0.0001 rad/s standard deviation. Shapes require less than 5 mm RMS tracking error during the constant-amplitude portion. Centering and carrying require the ball to stay within 3 mm of center throughout the final second. A pass also requires completion, contact, a positive or zero edge margin, tray path error below 20 mm and no unexpected contacts or simulation warnings.

| Task | Result | Measurement |
| --- | --- | --- |
| Circle | Passed | 2.80 mm RMS |
| Rounded square | Passed | 2.56 mm RMS over the constant-amplitude portion of the full 43 s run |
| Figure eight | Passed | 2.67 mm RMS |
| Centering | Passed | 1.02 mm maximum offset in the final second |
| Carrying | Passed | 1.46 mm maximum offset in the final second |

A separate nominal rounded-square video uses the same initial offset and seed, 20 ms delay and 0.0001 N·m torque noise. It completed the full 43 s at 2.12 mm RMS. [Video filenames and scores](results/video_index.csv)

## Noise, delay and controller choice

Each row uses two seeds for each of five tasks. A force-threshold stop records an unreliable measurement. It does not establish that the ball left the tray.

| Condition | Kalman + LQR passes |
| --- | ---: |
| 20 ms, 0.0001 N·m torque noise | 10/10 |
| 50 ms, 0.0005 N·m torque noise | 10/10 |
| 80 ms, 0.001 N·m torque noise | 5/10 |
| 50 ms, 0.001 N·m wrist torque bias | 8/10 |
| 50 ms, observer assumes 10% excess tray mass | 0/10 |

In the 80 ms circle with seed 401, the feedback estimate dipped to 0.157 N against the 0.200 N threshold. The simulated ball still exerted 0.265 N on the tray at that sample. In the 10% mass-error case, the first rejected reading was 0.194 N while the ball exerted 0.370 N. These contact measurements were recorded for diagnosis only. [Force readings at the stops](results/wrench_stop_diagnostics.json)

Thirteen LQR trials stopped at the normal-force check. Four more finished outside the performance criteria. The four methods below used the same seeds, starts, tasks and disturbances. On the revised plate the controller receives its actual dimensions for the carrying and centering tasks. This also corrects a stale size assumption in the earlier MPC comparison.

| Controller | All trials passed | Circle RMS in the matched 50 ms video |
| --- | ---: | ---: |
| PD | 10/50 | 8.84 mm |
| Kalman + PD | 27/50 | 1.71 mm |
| Kalman + LQR | 33/50 | 2.80 mm |
| Kalman + hybrid MPC/PD | 22/50 | 3.35 mm |

Kalman + LQR passed the most cases with these tunings. Kalman + PD tracked the single recorded circle most accurately. The independent LQR runs matched the LQR comparison outcomes across all 50 cases. [Main trials](results/stl_main_trials.csv) · [All four methods](results/method_all_trials.csv) · [Source hashes](results/method_scene_manifest.json)

## Print mass, friction and available deck area

- **Print mass.** Two new seeds tested circle and carrying at 50 ms. When the observer used the correct simulated mass, 4/4 passed at 121.875 g, 4/4 at 162.5 g and 3/4 at 203.125 g. Keeping the observer fixed at 162.5 g gave 0/4 at each off-nominal mass. With the plant at 162.5 g and the observer still using the previous 209.36 g estimate, all four trials stopped on insufficient estimated normal force. At rest on a level tray, that 46.86 g error corresponds to 0.460 N, close to the modeled ball’s 0.451 N weight. [Mass trials](results/print_mass_sensitivity.csv)
- **Friction and fasteners.** A 48-trial circle and centering sweep used friction coefficients of 0.1, 0.3 and 0.6, modeled bolt masses of 0 and 10 g, two delays and two new seeds. The outcomes were 0/16, 12/16 and 12/16 respectively. All 24 failures stopped at the normal-force check. [Material trials](results/material_sensitivity.csv)
- **Larger patterns.** At 50 ms, circles of 60 and 80 mm radius each stopped in both seeds even with the same steady tangential speed as the 20 mm circle. The simulated ball footprint still had at least 27 mm of edge margin in the recorded samples. Centering from (70, 0) mm passed twice, ending within 1.03 mm. Centering from (−80, 60) mm stopped in both trials. [Expanded deck trials](results/expanded_deck_trials.csv)

## Clearance and hardware

The smallest modeled plate-to-finger gap was **0.158 mm** during the sampled circle and carrying runs. The smallest modeled ball-to-arm gaps were 66.78 mm for the circle and 72.50 mm for carrying. The mount is rigid in simulation. The quoted finger gap is for the deck edge. The tab and finger contact pair is excluded to represent the grip. Real finger compliance, bolts, cables and surrounding obstacles can change these distances. [Sampled distances](results/distance_audit.json)

The separate `torque_balance/` bench models a 254 mm square paddle and a different attachment. Its default joint pose holds the revised CAD about 90° from level. The outward home used in these trials holds the revised CAD level. For physical testing, the outstanding measurements are the added bolt mass, installed mount and clearances, and torque readings with the tray empty and the ball at known positions at the intended level pose.
