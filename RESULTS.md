# Actual-platform simulation results

The runs use the supplied STL for appearance, the 92.77 g mass supplied for the printed PLA part, a nominal 42.67 mm diameter and 45.93 g golf ball, and a hypothetical rigid attachment to the standard Franka hand. The original STL is in `inputs/`. This is an offline simulation, not a physical demonstration or a validated model of the bolted connection. Ball dimensions and mass should be measured for the ball used in the lab. The nominal diameter and mass correspond to the minimum diameter and maximum mass in the [R&A equipment rules](https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls).

## Controller and measurements

The demonstrated controller uses a Kalman filter and linear-quadratic regulator. It receives a delayed, noisy simulated joint-torque-derived wrench and estimated robot motion, not the simulated ball position. The simulator's ball state is used only to score the trials and draw the orange trajectory in the videos. Blue is the reference and green is the estimated position. The sensor noise, delay, load model, friction and attachment transform are assumptions; none establish that the real Franka can estimate the ball position accurately enough for this task.

The STL is nonconvex. For collision, the default model uses a flat tray and conservative tab boxes. It omits the small center dimple and the tab hole; bolts and the actual mechanical grip are not modeled. The computed tray inertia assumes a uniform solid and the optional bolt mass defaults to zero. Changing any of these assumptions can change the results.

## Main study

`results/stl_main_trials.csv` contains all 50 trials: five tasks, five conditions, and two seeds/initial offsets. The shape task criteria include less than 5 mm RMS tracking error and the other simulation checks. The centering and carrying criteria include less than 3 mm final-second ball offset and the simulation checks. The scores are numerical criteria, not real-world success rates.

| Condition | Passed |
| --- | ---: |
| Nominal simulated noise, 20 ms delay | 10/10 |
| Increased simulated noise, 50 ms delay | 10/10 |
| Increased simulated noise, 80 ms delay | 9/10 |
| Hypothetical 0.001 N m wrist torque bias, 50 ms delay | 8/10 |
| Estimator assumes 10% too much tray mass | 0/10 |
| Total | 37/50 |

The ten mass-mismatch cases stopped at the insufficient estimated normal force check; that stop does not by itself prove that the ball fell. The study uses only two seeds per condition and does not establish the best possible algorithm or its hardware reliability.

`results/material_sensitivity.csv` contains 48 additional circle and centering trials across hypothetical sliding friction values 0.1, 0.3 and 0.6, bolt masses 0 and 10 g, delays of 20 and 50 ms, and two new seeds. The plant and estimator assumed the same bolt mass. Overall, 28/48 passed: 0/16 at friction 0.1, 14/16 at 0.3, and 14/16 at 0.6. All 20 failures stopped on insufficient estimated normal force; they should not be interpreted as confirmed falls. Friction and bolt mass have not been measured.

## Clearance and next measurements

`results/distance_audit.json` records sampled modeled distances in representative circle and carrying trials. The minimum modeled plate-to-finger clearance was about 0.158 mm. This result depends on an unmeasured mount pose and simplified geometry and is not a physical clearance measurement. Tables, shelf structures, people, cables and actual bolts were not modeled. The center dimple is not represented in the default collision model; an exploratory high-detail contact approximation produced numerical artifacts, so its effects remain unresolved.

Measure the installed tray-to-hand transform, fit and clearances, complete payload mass and center of mass, ball properties, actual wrench noise and delay, and sensor observability before attempting a hardware controller. The read-only tools in `hardware/` can help collect stationary measurements; the repository does not provide a validated real-robot balancing controller.
