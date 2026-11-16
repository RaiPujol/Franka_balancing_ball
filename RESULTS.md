# Outward-facing platform: simulation results

The platform extends away from the robot base at the default home pose, with the tab at the gripper and the top surface facing up. The user-supplied STL is used for visual geometry, and the printed PLA part is modeled at the supplied mass of 92.77 g. The golf ball is modeled with a nominal 42.67 mm diameter and 45.93 g mass. The [R&A equipment rules](https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls) specify the minimum diameter and maximum mass, so measure the actual ball.

## Model and controller

The demonstrations use a Kalman filter and linear-quadratic regulator. Feedback uses a delayed, noisy simulated joint-torque-derived wrench and estimated robot motion. The ball's simulated position is used only for scoring and the orange trajectory overlay, not as controller input. In the videos, blue is the target and green is the estimate.

The attachment transform, noise, delay, friction and mass distribution are assumptions. The hand rigidly holds the tab by construction; bolt geometry and detailed mechanical fit are not modeled. The nonconvex STL is rendered, while the collision model uses a flat tray and conservative tab boxes; it omits the center dimple and tab hole. Inertia assumes uniform solid density, and bolt mass defaults to zero. These results do not establish hardware observability or reliability.

## Noise, delay and modeling error

`results/stl_main_trials.csv` contains 50 trials over five tasks and two seeds/initial offsets per condition. The shape criteria include less than 5 mm RMS tracking error, while centering and carrying include less than 3 mm ball offset over the final second; both also require the other simulation checks. These are numerical simulation criteria, not real-world success rates.

| Test condition | Passed |
| --- | ---: |
| Nominal simulated noise, 20 ms delay | 10/10 |
| Higher simulated noise, 50 ms delay | 10/10 |
| Higher simulated noise, 80 ms delay | 10/10 |
| Hypothetical 0.001 N m wrist torque bias, 50 ms delay | 8/10 |
| Estimator assumes 10% too much tray mass | 4/10 |
| Total | 42/50 |

Four failed trials stopped on the insufficient estimated normal-force check; this does not itself show the ball fell. Four carrying trials finished but did not meet the numerical success criteria. Only two seeds were tested per condition.

`results/material_sensitivity.csv` contains 48 additional circle and centering trials across hypothetical sliding friction values of 0.1, 0.3 and 0.6, bolt masses of 0 and 10 g, delays of 20 and 50 ms, and two new seeds. The plant and estimator used the same bolt mass. Overall 32/48 passed: 0/16 at friction 0.1, 16/16 at 0.3, and 16/16 at 0.6. All 16 failed trials stopped on the normal-force check; that alone does not prove the ball fell. Friction and bolt mass were not measured.

## Matched controller comparison

Each method was evaluated on the same 50 task, condition, and seed combinations used in the main study. These are the pass counts for the specific tested tunings and model, not a general ranking of the algorithms.

| Controller | Passed |
| --- | ---: |
| PD | 10/50 |
| Kalman + PD | 27/50 |
| Kalman + LQR | 42/50 |
| Kalman + hybrid MPC/PD | 28/50 |

The complete data and breakdown are in `results/method_all_trials.csv` and `results/method_aggregate.csv`. The LQR trial results match the independent main study. `results/method_scene_manifest.json` records the sources used for the comparison; the comparison wrapper was subsequently extended to write a manifest on future runs, without changing the trial computation. Solver timing reflects this computer and concurrent workload, not a real-time hardware guarantee.

## Matched circle videos

The four comparison videos use the same outward tray, circle path, 50 ms simulated delay, noise and seed 401. In this single scenario, the tracking RMS errors were: PD 9.14 mm, Kalman + PD 1.37 mm, Kalman + LQR 2.10 mm, and hybrid MPC/PD 2.94 mm. The PD video completed but missed the 5 mm tracking criterion; the other three passed. This one example does not establish which method is most robust.

## Clearance and real-robot use

`results/distance_audit.json` records sampled modeled distances during representative circle and carrying trials. Minimum modeled plate-to-finger clearance was about 0.158 mm. The mount pose is unmeasured, the collision geometry is approximate, and sampling cannot guarantee continuous clearance; this number is not evidence of a safe physical fit. Tables, shelves, people, cables and actual bolts were not modeled. A previous exploratory approximation of the dimple produced numerical contact artifacts, so the dimple's effect remains unresolved.

Measure the installed tray-to-hand transform, fit and clearances, complete payload mass and center of mass, ball properties, actual wrench noise and delay, and sensor observability before attempting a hardware controller. The `hardware/` tools collect stationary readings only. This repository does not provide a validated real-robot balancing controller.
